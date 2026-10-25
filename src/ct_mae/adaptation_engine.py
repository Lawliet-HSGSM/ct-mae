"""Twin-view optimization with group-wide CE normalization and local SupCon pools."""

import itertools
import time

import numpy as np
import torch
from torch.nn import functional as F

from ct_mae.adaptation_loss import supcon_loss, weighted_ce_sum
from ct_mae.engine import classification_metrics, learning_rate


def make_adaptation_optimizer(model, recipe):
    buckets = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            multiplier = 10 if name.startswith('projector.') else 1
            decay = 0. if parameter.ndim == 1 or name.endswith('.bias') else .05
            buckets.setdefault((multiplier, decay), []).append(parameter)
    return torch.optim.AdamW([
        {'params': parameters, 'lr': recipe['lr'] * multiplier,
         'lr_multiplier': multiplier, 'weight_decay': decay}
        for (multiplier, decay), parameters in buckets.items()
    ], betas=(.9, .999), eps=1e-8)


def train_adaptation_epoch(model, loader, optimizer, scaler, recipe, epoch, weights,
                           *, device, amp, contrastive=False):
    model.train()
    weights = weights.to(device)
    iterator = iter(loader)
    seen = count = updates = skipped = 0
    raw_total = weighted_total = weight_total = sup_total = lambda_total = scaled_sup_total = 0.
    labels, predicted, gradient_norms = [], [], []
    started = time.perf_counter()
    while group := list(itertools.islice(iterator, recipe['accumulation'])):
        progress = epoch + seen / len(loader)
        rate = learning_rate(progress, recipe, recipe['min_lr'])
        coefficient = .05 * min(1., progress / recipe.get('lambda_warmup_epochs', 5)) if (
            contrastive) else 0.
        for bucket in optimizer.param_groups:
            bucket['lr'] = rate * bucket.get('lr_multiplier', 1)
        size = sum(len(batch['label']) for batch in group)
        # One denominator for the complete optimizer update, not separate weighted means.
        denominator = 2 * sum(float(weights[batch['label'].to(device)].sum()) for batch in group)
        optimizer.zero_grad(set_to_none=True)
        for batch in group:
            inputs = batch['views'].flatten(0, 1).to(device, non_blocking=True)
            target = batch['label'].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits, projection = model.adaptation_forward(inputs)
            numerator, assigned = weighted_ce_sum(logits, target, weights)
            sup = supcon_loss(projection, target.repeat_interleave(2)) if contrastive else (
                logits.new_zeros((), dtype=torch.float32)
            )
            loss = numerator / denominator + coefficient * sup * (len(target) / size)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite adaptation loss')
            scaler.scale(loss).backward()
            raw_total += float(F.cross_entropy(logits.float(), target.repeat_interleave(2),
                                               reduction='sum').detach())
            weighted_total += float(numerator.detach())
            weight_total += float(assigned)
            sup_total += float(sup.detach()) * len(target)
            scaled_sup_total += coefficient * float(sup.detach()) * len(target)
            lambda_total += coefficient * len(target)
            labels.extend(target.cpu().tolist())
            probabilities = logits.detach().float().softmax(-1).reshape(len(target), 2, -1)
            predicted.extend(probabilities.mean(1).argmax(-1).cpu().tolist())
            count += len(target)
            seen += 1
        scaler.unscale_(optimizer)
        norm = torch.stack([p.grad.detach().float().norm() for p in model.parameters()
                            if p.grad is not None]).norm()
        if torch.isfinite(norm):
            gradient_norms.append(float(norm))
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped += int(scaler.get_scale() < scale)
        updates += 1
    if not count:
        raise ValueError('Empty adaptation loader')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    ce, wce = raw_total / (2 * count), weighted_total / weight_total
    result = {'loss': wce + scaled_sup_total / count,
              'ce': ce, 'weighted_ce': wce, 'supcon': sup_total / count,
              'lambda': lambda_total / count, 'lr': rate, 'seconds': time.perf_counter() - started,
              'samples': count, 'views': count * 2, 'optimizer_steps': updates - skipped,
              'skipped_steps': skipped, 'gradient_norm': float(np.mean(gradient_norms))
              if gradient_norms else 0.}
    result.update(classification_metrics(labels, predicted, len(weights)))
    return result
