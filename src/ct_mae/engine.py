"""Training primitives with sample-weighted accumulation and resumable RNG state."""

import itertools
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def classification_metrics(labels, predicted, num_classes: int) -> dict:
    labels, predicted = np.asarray(labels, dtype=int), np.asarray(predicted, dtype=int)
    confusion = np.bincount(
        labels * num_classes + predicted, minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * confusion.diagonal(), denominator,
                   out=np.zeros(num_classes, dtype=float), where=denominator != 0)
    return {"accuracy": float(confusion.trace() / max(1, confusion.sum())),
            "macro_f1": float(f1.mean()), "per_class_f1": f1.tolist(),
            "confusion_matrix": confusion.tolist()}


def make_optimizer(model: nn.Module, recipe: dict, eps: float):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim == 1 or name.endswith(".bias") else decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": recipe["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=recipe["lr"], betas=tuple(recipe["betas"]), eps=eps)


def learning_rate(progress: float, recipe: dict, minimum: float) -> float:
    warmup = recipe["warmup_epochs"]
    if progress < warmup:
        return recipe["lr"] * progress / warmup
    fraction = min(1.0, max(0.0, (progress - warmup) / (recipe["epochs"] - warmup)))
    return minimum + (recipe["lr"] - minimum) * (1 + math.cos(math.pi * fraction)) / 2


def fixed_noise(indices: torch.Tensor, patches: int, seed: int) -> torch.Tensor:
    return torch.stack([
        torch.rand(patches, generator=torch.Generator().manual_seed(seed + int(index)))
        for index in indices
    ])


def train_epoch(model, loader, optimizer, scaler, *, device, recipe, epoch, min_lr,
                pretrain: bool, amp: bool, num_classes: int = 11) -> dict:
    model.train()
    iterator = iter(loader)
    batches_seen, count, total_loss = 0, 0, 0.0
    labels, predicted = [], []
    updates, skipped = 0, 0
    started = time.perf_counter()
    while group := list(itertools.islice(iterator, recipe["accumulation"])):
        rate = learning_rate(epoch + batches_seen / len(loader), recipe, min_lr)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        group_count = sum(len(batch["image"]) for batch in group)
        for batch in group:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                if pretrain:
                    loss, _, _ = model(images)
                else:
                    logits = model(images)
                    target = batch["label"].to(device, non_blocking=True)
                    loss = F.cross_entropy(logits, target)
                    labels.extend(batch["label"].tolist())
                    predicted.extend(logits.detach().argmax(-1).cpu().tolist())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            # Weight by samples, including a short last microbatch/accumulation group.
            # https://docs.pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
            scaler.scale(loss * (len(images) / group_count)).backward()
            count += len(images)
            total_loss += loss.detach().item() * len(images)
            batches_seen += 1
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped += int(scaler.get_scale() < scale_before)
        updates += 1
    if not count:
        raise ValueError("Empty training loader")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {"loss": total_loss / count, "lr": rate,
              "seconds": time.perf_counter() - started, "samples": count,
              "optimizer_steps": updates - skipped, "skipped_steps": skipped,
              "amp_scale": scaler.get_scale()}
    if not pretrain:
        result.update(classification_metrics(labels, predicted, num_classes))
    return result


@torch.inference_mode()
def evaluate(model, loader, *, device, pretrain: bool, seed: int, amp: bool,
             num_classes: int = 11, return_predictions: bool = False):
    model.eval()
    total_loss, count = 0.0, 0
    labels, predicted, probabilities, indices = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            if pretrain:
                patches = (model.config["image_size"] // model.config["patch_size"])**2
                loss, _, _ = model(images, noise=fixed_noise(batch["index"], patches, seed))
            else:
                logits = model(images)
                loss = F.cross_entropy(logits, batch["label"].to(device, non_blocking=True))
                labels.extend(batch["label"].tolist())
                predicted.extend(logits.argmax(-1).cpu().tolist())
                if return_predictions:
                    probabilities.extend(logits.float().softmax(-1).cpu().tolist())
                    indices.extend(batch["index"].tolist())
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss")
        total_loss += loss.item() * len(images)
        count += len(images)
    if not count:
        raise ValueError("Empty evaluation loader")
    result = {"loss": total_loss / count, "samples": count}
    if not pretrain:
        result.update(classification_metrics(labels, predicted, num_classes))
    if return_predictions:
        return result, {"index": indices, "label": labels, "prediction": predicted,
                        "probability": probabilities}
    return result


def capture_rng() -> dict:
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(path: Path, model, optimizer, scaler, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({**state, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "rng": capture_rng()}, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, model, optimizer=None, scaler=None,
                    *, restore_random: bool = True) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_random and "rng" in checkpoint:
        restore_rng(checkpoint["rng"])
    return checkpoint
