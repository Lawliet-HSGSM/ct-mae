"""Warm-start full/partial classifiers; validation designs the study, test runs last."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ct_mae.adaptation_data import PAIRS, TwinViewDataset, class_weights, paired_batches
from ct_mae.adaptation_engine import make_adaptation_optimizer, train_adaptation_epoch
from ct_mae.adaptation_loss import AdaptationClassifier
from ct_mae.config import config_digest, write_json
from ct_mae.data import OrganDataset, read_split, select_labeled_indices
from ct_mae.download import fetch_asset, file_md5, official_asset
from ct_mae.engine import (
    capture_rng,
    evaluate,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
    seed_everything,
)
from ct_mae.pipeline import environment_info, make_loader, sha256_file
from ct_mae.tracking import RunTracker

VARIANTS = ('A', 'B', 'C', 'D', 'E')
RECIPE = dict(epochs=50, lr=3e-5, min_lr=1e-6, warmup_epochs=5,
              batch_size=32, accumulation=2)


def build_model(sources, scope, variant, *, seed=42, device=None):
    device = device or torch.device('cpu')
    expected = json.loads((sources / 'source_manifest.json').read_text('utf-8'))
    source = sources / f'{scope}.pt'
    if sha256_file(source) != expected['checkpoints'][scope]:
        raise ValueError('Source checkpoint checksum changed')
    saved = torch.load(source, map_location='cpu', weights_only=True)
    baseline = json.loads((sources / 'baseline.json').read_text('utf-8'))
    if saved['config']['model'] != baseline['model']:
        raise ValueError('Source model architecture differs from the approved baseline')
    if saved['subset_sha256'] != expected['subset_sha256']:
        raise ValueError('Source checkpoint uses different training labels')
    model = AdaptationClassifier(saved['config']['model'], partial=scope == 'partial',
                                 contrastive=variant in ('D', 'E'), seed=seed)
    missing, unexpected = model.load_state_dict(saved['model'], strict=False)
    if unexpected or any(not key.startswith('projector.') for key in missing):
        raise ValueError('Source encoder or classifier head did not load exactly')
    return model.to(device), saved


def scalar_class_metrics(metrics):
    return {**metrics, **{f'f1_c{i}': value
                         for i, value in enumerate(metrics['per_class_f1'])}}


def training_loader(dataset, selected_labels, recipe, variant, epoch, workers):
    dataset.set_epoch(epoch)
    if variant == 'A':
        order = np.random.default_rng(42 + epoch).permutation(len(dataset))
        batches = order.reshape(-1, recipe['batch_size']).tolist()
    else:
        batches, _ = paired_batches(selected_labels, recipe['batch_size'], seed=42 + epoch)
    return DataLoader(dataset, batch_sampler=batches, num_workers=workers,
                      pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(42 + epoch))


def profile_recipe(sources, dataset, device):
    recipe = copy.deepcopy(RECIPE)
    if device.type != 'cuda':
        return recipe
    for batch_size, accumulation in ((32, 2), (16, 4)):
        recipe.update(batch_size=batch_size, accumulation=accumulation)
        seed_everything(42)
        model, _ = build_model(sources, 'full', 'E', device=device)
        optimizer = make_adaptation_optimizer(model, recipe)
        scaler = torch.amp.GradScaler('cuda')
        try:
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
            batch = next(iter(loader))
            train_adaptation_epoch(model, [batch], optimizer, scaler,
                                   {**recipe, 'accumulation': 1}, 5, torch.ones(11),
                                   device=device, amp=True, contrastive=True)
            return recipe
        except torch.cuda.OutOfMemoryError:
            if batch_size == 16:
                raise
        finally:
            del model, optimizer, scaler
            torch.cuda.empty_cache()
    raise RuntimeError('No common batch size fitted')


def run_variant(sources, output, scope, variant, dataset, selected_labels, validation,
                clean_train, initial, weights, recipe, *, device, workers, smoke=False,
                stop_after=None):
    directory = output / scope / variant
    directory.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((sources / 'source_manifest.json').read_text('utf-8'))
    baseline = json.loads((sources / 'baseline.json').read_text('utf-8'))
    config = {'seed': 42, 'wandb': {**baseline['wandb'], 'run_name_suffix': 'adaptation5pct'},
              'model': baseline['model'],
              'scope': scope, 'variant': variant, 'recipe': recipe, 'smoke': smoke,
              'class_weights': weights.tolist(), 'source_manifest': source_manifest}
    fingerprint = config_digest(config)
    last_path = directory / 'last.pt'
    seed_everything(42)
    model, source_state = build_model(sources, scope, variant, device=device)
    optimizer = make_adaptation_optimizer(model, recipe)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history, start, best_epoch, best_score = [], 0, 0, initial['macro_f1']
    run_id = None
    if last_path.exists():
        saved = torch.load(last_path, map_location='cpu', weights_only=True)
        if saved['config_digest'] != fingerprint:
            raise ValueError('Adaptation resume configuration differs')
        record = json.loads((directory / 'run.json').read_text('utf-8'))
        if record['status'] in ('completed', 'smoke_completed'):
            return record
        load_checkpoint(last_path, model, optimizer, scaler)
        history, start = saved['history'], saved['epoch'] + 1
        best_epoch, best_score = saved['best_epoch'], saved['best_score']
        run_id = saved['wandb_run_id']
    state = capture_rng()
    tracker = RunTracker(config, f'adapt-{scope}-{variant}', directory, run_id=run_id, smoke=smoke)
    restore_rng(state)
    metadata = {'status': 'running', 'config': config, 'config_digest': fingerprint,
                'source': scope, 'variant': variant, 'recipe': recipe, 'initial_val': initial,
                'source_epoch': source_state['epoch'] + 1, 'completed_epochs': start,
                'history': history, 'best_epoch': best_epoch, 'best_score': best_score,
                'wandb_run_id': tracker.id, 'wandb_url': tracker.url,
                'environment': environment_info(device),
                'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}

    def save_best(epoch, val):
        torch.save({'model': model.state_dict(), 'config': config, 'epoch': epoch,
                    'source': scope, 'variant': variant, 'val': val}, directory / 'best.pt')

    if start == 0:
        save_best(0, initial)
    effective_weights = torch.tensor(weights if variant in ('C', 'E') else np.ones(11),
                                     dtype=torch.float32)
    status = 'failed'
    try:
        for epoch in range(start, min(recipe['epochs'], stop_after or recipe['epochs'])):
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            loader = training_loader(dataset, selected_labels, recipe, variant, epoch, workers)
            train = train_adaptation_epoch(model, loader, optimizer, scaler, recipe, epoch,
                                           effective_weights, device=device, amp=amp,
                                           contrastive=variant in ('D', 'E'))
            val = evaluate(model, validation, device=device, pretrain=False, seed=42, amp=amp)
            if val['macro_f1'] > best_score:
                best_score, best_epoch = val['macro_f1'], epoch + 1
                save_best(best_epoch, val)
            resources = {'peak_vram_bytes': torch.cuda.max_memory_allocated(device)
                         if device.type == 'cuda' else 0}
            row = {'epoch': epoch + 1, 'train': train, 'val': val, 'resource': resources}
            if (epoch + 1) % 10 == 0 or epoch + 1 == recipe['epochs']:
                row['train_clean'] = evaluate(model, clean_train, device=device, pretrain=False,
                                              seed=42, amp=amp)
            history.append(row)
            metadata.update(history=history, best_epoch=best_epoch, best_score=best_score,
                            completed_epochs=epoch + 1)
            tracker.log_epoch(epoch + 1, train, scalar_class_metrics(val), resources)
            save_checkpoint(last_path, model, optimizer, scaler, {
                'epoch': epoch, 'history': history, 'best_epoch': best_epoch,
                'best_score': best_score, 'config_digest': fingerprint,
                'wandb_run_id': tracker.id, 'scheduler': {'next_epoch': epoch + 1},
            })
            write_json(directory / 'run.json', metadata)
            print(f'ADAPT {scope}/{variant} {epoch+1}/{recipe["epochs"]} '
                  f'val_f1={val["macro_f1"]:.6f} best={best_score:.6f}', flush=True)
        status = ('smoke_completed' if smoke else 'completed') if (
            metadata['completed_epochs'] == recipe['epochs']) else 'interrupted'
        frozen = [key for key, p in model.named_parameters() if not p.requires_grad]
        if any(not torch.equal(model.state_dict()[key].cpu(), source_state['model'][key])
               for key in frozen):
            raise RuntimeError('Adaptation changed a frozen source tensor')
        metadata['freeze_audit'] = {'passed': True, 'unchanged_tensors': len(frozen)}
        if status in ('completed', 'smoke_completed'):
            metadata['status'] = status
            write_json(directory / 'run.json', metadata)
            tracker.checkpoint(directory, metadata['completed_epochs'])
    except BaseException:
        status = 'failed'
        raise
    finally:
        metadata['status'] = status
        write_json(directory / 'run.json', metadata)
        tracker.finish(status, {'best_val/macro_f1': best_score, 'best_epoch': best_epoch})
    return metadata


def run_study(sources, archive, output, device, *, smoke=False, workers=2):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((sources / 'subset.json').read_text('utf-8'))
    selected = np.array(manifest['indices'], dtype=np.int64)
    digest = hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest()
    if digest != manifest['indices_sha256'] or file_md5(archive) != manifest['md5']:
        raise ValueError('Dataset or 5% labeled subset checksum differs')
    images, labels = read_split(archive, 'train', supervised=True)
    val_images, val_labels = read_split(archive, 'val', supervised=True)
    if smoke:
        selected = selected[select_labeled_indices(labels[selected], 128, 42)]
        val_indices = select_labeled_indices(val_labels, 128, 42)
    else:
        if len(selected) != 1728:
            raise ValueError('Production adaptation requires the original 1,728 labels')
        val_indices = np.arange(len(val_labels))
    dataset = TwinViewDataset(images, labels, selected)
    validation = make_loader(OrganDataset(val_images, val_labels, val_indices), 64, workers,
                             seed=42, shuffle=False)
    clean_train = make_loader(OrganDataset(images, labels, selected), 64, workers,
                              seed=42, shuffle=False)
    recipe = profile_recipe(sources, dataset, device)
    if smoke:
        recipe.update(epochs=2, warmup_epochs=0)
    counts = np.bincount(labels[selected], minlength=11)
    initial, weights = {}, {}
    for scope in ('full', 'partial'):
        model, _ = build_model(sources, scope, 'A', device=device)
        metrics, predictions = evaluate(model, validation, device=device, pretrain=False,
                                        seed=42, amp=device.type == 'cuda', return_predictions=True)
        initial[scope] = metrics
        weights[scope] = class_weights(metrics['per_class_f1'], counts)
        directory = output / scope
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory / 'initial_val_predictions.npz', **predictions)
        write_json(directory / 'initial.json', metrics)
        del model
    source_manifest = json.loads((sources / 'source_manifest.json').read_text('utf-8'))
    protocol = {'seed': 42, 'smoke': smoke, 'recipe': recipe, 'pairs': PAIRS,
                'source_manifest': source_manifest,
                'subset_sha256': digest, 'actual_train_indices': selected.tolist(),
                'train_class_counts': counts.tolist(), 'initial_validation': initial,
                'class_weights': {key: value.tolist() for key, value in weights.items()},
                'view_count': 2, 'temperature': .1, 'lambda_max': .05, 'lambda_warmup_epochs': 5,
                'validation_role': 'design weights/pairs and select checkpoints; no gradients'}
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text('utf-8')) != json.loads(
            json.dumps(protocol)):
        raise ValueError('Output already holds another adaptation protocol')
    write_json(protocol_path, protocol)
    for scope in ('full', 'partial'):
        for variant in VARIANTS:
            write_json(output / 'progress.json', {'status': 'training', 'source': scope,
                                                   'variant': variant})
            run_variant(sources, output, scope, variant, dataset, labels[selected], validation,
                        clean_train, initial[scope], weights[scope], recipe,
                        device=device, workers=workers, smoke=smoke)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    write_json(output / 'progress.json', {'status': 'training_completed', 'smoke': smoke})
    if not smoke:
        from ct_mae.adaptation_evaluate import evaluate_study
        evaluate_study(sources, archive, output, device, workers=workers)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    archive = fetch_asset(official_asset(128), args.data_dir)
    run_study(args.sources, archive, args.output, torch.device(args.device),
              smoke=args.smoke, workers=args.workers)


if __name__ == '__main__':
    main()
