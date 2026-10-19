"""Stage orchestration: shared data protocol, checkpoint selection and final-only tests."""

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ct_mae.config import config_digest, write_json
from ct_mae.data import OrganDataset, read_split
from ct_mae.engine import (
    capture_rng,
    evaluate,
    load_checkpoint,
    make_optimizer,
    restore_rng,
    save_checkpoint,
    seed_everything,
    train_epoch,
)
from ct_mae.models import MaskedAutoencoder, OrganClassifier
from ct_mae.tracking import RunTracker


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_info(device: torch.device) -> dict:
    versions = {}
    for package in ("torch", "torchvision", "wandb", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = os.environ.get("CT_MAE_COMMIT", "unknown")
    return {"python": platform.python_version(), "versions": versions, "git_commit": revision,
            "device": str(device), "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "vram_bytes": torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda" else None}


def make_loader(dataset, batch_size: int, workers: int, *, seed: int, shuffle: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False,
                      generator=torch.Generator().manual_seed(seed))


def build_model(stage: str, config: dict, output: Path, device: torch.device):
    if stage == "pretrain":
        return MaskedAutoencoder(config["model"]).to(device), None
    model = OrganClassifier(config["model"], config["dataset"]["num_classes"],
                            frozen=stage == "linear", seed=config["seed"])
    source_hash = None
    if stage in ("linear", "finetune"):
        source = output / "pretrain" / "best.pt"
        record = json.loads(source.with_name("run.json").read_text("utf-8"))
        expected_status = "smoke_completed" if config.get("run_kind") == "smoke" else "completed"
        if (record["status"] != expected_status
                or record["completed_epochs"] != config["stages"]["pretrain"]["epochs"]):
            raise ValueError("SSL transfer requires completed pretraining at the configured budget")
        if record["config_digest"] != config_digest(config):
            raise ValueError("SSL pretraining configuration differs from the transfer protocol")
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        if (checkpoint["config"].get("run_kind", "full")
                != config.get("run_kind", "full")):
            raise ValueError("Smoke and full experiment checkpoints cannot be mixed")
        if checkpoint["config"]["model"] != config["model"]:
            raise ValueError("SSL checkpoint architecture does not match the classifier")
        model.encoder.load_state_dict({key.removeprefix("encoder."): value
                                       for key, value in checkpoint["model"].items()
                                       if key.startswith("encoder.")})
        source_hash = sha256_file(source)
    return model.to(device), source_hash


def check_batch_fits(model, dataset, recipe, config, device, pretrain: bool):
    """Probe a complete optimizer step and undo it before collecting training results."""
    if device.type != "cuda":
        return recipe, []
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    rng = capture_rng()
    recipe, changes = copy.deepcopy(recipe), []
    while True:
        optimizer = make_optimizer(model, recipe, config["training"]["adam_eps"])
        scaler = torch.amp.GradScaler("cuda", enabled=config["training"]["amp"])
        try:
            loader = make_loader(dataset, min(len(dataset), recipe["batch_size"]), 0,
                                 seed=config["seed"], shuffle=False)
            batch = next(iter(loader))
            one = {**recipe, "accumulation": 1, "warmup_epochs": 0}
            train_epoch(model, [batch], optimizer, scaler, device=device, recipe=one, epoch=0,
                        min_lr=config["training"]["min_lr"], pretrain=pretrain,
                        amp=config["training"]["amp"],
                        num_classes=config["dataset"]["num_classes"])
        except torch.cuda.OutOfMemoryError:
            if recipe["batch_size"] <= 1:
                raise
            old_batch = recipe["batch_size"]
            if old_batch % 2:
                raise RuntimeError(
                    "Cannot halve an odd batch while preserving effective batch"
                ) from None
            recipe["batch_size"] //= 2
            recipe["accumulation"] *= 2
            changes.append({"reason": "CUDA OOM", "batch_size": recipe["batch_size"],
                            "accumulation": recipe["accumulation"]})
        else:
            break
        finally:
            model.zero_grad(set_to_none=True)
            model.load_state_dict(initial)
            restore_rng(rng)
            del optimizer, scaler
            torch.cuda.empty_cache()
    return recipe, changes


def run_stage(stage: str, config: dict, archive: Path, manifest: dict, output: Path,
              device: torch.device, *, resume: bool = False, stop_after_epoch: int | None = None,
              smoke: bool = False, train_limit: int | None = None,
              val_limit: int | None = None) -> dict:
    directory = output / stage
    directory.mkdir(parents=True, exist_ok=True)
    last_path = directory / "last.pt"
    if last_path.exists() and not resume:
        raise FileExistsError(f"Use --resume or a new output directory: {directory}")
    fingerprint = config_digest(config)
    if resume:
        previous = torch.load(last_path, map_location="cpu", weights_only=True)
        if previous["config_digest"] != fingerprint:
            raise ValueError("Resume configuration differs from the recorded configuration")
        if previous["subset_sha256"] != manifest["indices_sha256"]:
            raise ValueError("Resume labeled subset differs from the original run")
        if previous.get("smoke", False) != smoke:
            raise ValueError("A smoke checkpoint cannot be resumed as a full experiment")
        saved_metadata = json.loads((directory / "run.json").read_text("utf-8"))
        if (previous["epoch"] + 1 >= config["stages"][stage]["epochs"]
                and saved_metadata["status"] == "completed"):
            return saved_metadata
        run_id = previous["wandb_run_id"]
    else:
        previous, run_id = None, None
    seed_everything(config["seed"])
    pretrain = stage == "pretrain"
    model, source_hash = build_model(stage, config, output, device)
    images, labels = read_split(archive, "train", supervised=not pretrain)
    selected = np.arange(len(images)) if pretrain else np.array(manifest["indices"])
    if train_limit is not None:
        selected = selected[:train_limit]
    train_data = OrganDataset(images, labels, selected, augment=True, seed=config["seed"],
                              **config["augmentation"])
    val_images, val_labels = read_split(archive, "val", supervised=not pretrain)
    val_indices = np.arange(min(len(val_images), val_limit or len(val_images)))
    val_data = OrganDataset(val_images, val_labels, val_indices)
    recipe = copy.deepcopy(previous["recipe"] if previous else config["stages"][stage])
    recipe, adjustments = check_batch_fits(model, train_data, recipe, config, device, pretrain)
    optimizer = make_optimizer(model, recipe, config["training"]["adam_eps"])
    amp = config["training"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, start, best_score, best_epoch = [], 0, None, None
    if previous:
        load_checkpoint(last_path, model, optimizer, scaler)
        history, start = previous["history"], previous["epoch"] + 1
        best_score, best_epoch = previous["best_score"], previous["best_epoch"]
    runtime = environment_info(device)
    sessions = saved_metadata.get("sessions", []) if previous else []
    sessions.append({"start_epoch": start + 1, "environment": runtime, "recipe": recipe})
    # W&B setup must not consume the model's training RNG, including during resume.
    training_rng = capture_rng()
    tracker = RunTracker(config, stage, directory, run_id=run_id, smoke=smoke)
    restore_rng(training_rng)
    metadata = {
        "stage": stage, "status": "running", "smoke": smoke, "config": config,
        "config_digest": fingerprint, "recipe": recipe, "batch_adjustments": adjustments,
        "subset_sha256": manifest["indices_sha256"], "labeled_count": manifest["labeled_count"],
        "train_samples": len(train_data), "val_samples": len(val_data),
        "environment": runtime, "sessions": sessions, "ssl_checkpoint_sha256": source_hash,
        "parameters": {"total": sum(p.numel() for p in model.parameters()),
                       "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                       "encoder": sum(p.numel() for p in model.encoder.parameters())},
        "wandb_run_id": tracker.id, "wandb_url": tracker.url,
        "history": history, "completed_epochs": start,
    }
    write_json(directory / "run.json", metadata)
    if tracker.run is not None:
        tracker.run.config.update({"actual_recipe": recipe, "parameters": metadata["parameters"],
                                   "subset_sha256": manifest["indices_sha256"],
                                   "environment": metadata["environment"]}, allow_val_change=True)
    workers = config["training"]["num_workers"]
    validation = make_loader(val_data, recipe["batch_size"], workers,
                             seed=config["seed"], shuffle=False)
    stop = min(recipe["epochs"], stop_after_epoch or recipe["epochs"])
    started = time.perf_counter()
    status = "failed"
    try:
        for epoch in range(start, stop):
            train_data.set_epoch(epoch)
            loader = make_loader(train_data, recipe["batch_size"], workers,
                                 seed=config["seed"] + epoch, shuffle=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train = train_epoch(model, loader, optimizer, scaler, device=device, recipe=recipe,
                                epoch=epoch, min_lr=config["training"]["min_lr"],
                                pretrain=pretrain, amp=amp,
                                num_classes=config["dataset"]["num_classes"])
            val = evaluate(model, validation, device=device, pretrain=pretrain,
                           seed=config["seed"], amp=amp,
                           num_classes=config["dataset"]["num_classes"])
            score = -val["loss"] if pretrain else val["macro_f1"]
            improved = best_score is None or score > best_score
            if improved:
                best_score, best_epoch = score, epoch + 1
                temporary = directory / "best.tmp"
                torch.save({"model": model.state_dict(), "config": config, "stage": stage,
                            "epoch": epoch, "val": val,
                            "subset_sha256": manifest["indices_sha256"]},
                           temporary)
                temporary.replace(directory / "best.pt")
            resources = {"peak_vram_bytes": torch.cuda.max_memory_allocated(device)
                         if device.type == "cuda" else 0}
            history.append({"epoch": epoch + 1, "train": train, "val": val,
                            "resource": resources})
            metadata.update(completed_epochs=epoch + 1, best_score=best_score,
                            best_epoch=best_epoch)
            write_json(directory / "run.json", metadata)
            tracker.log_epoch(epoch + 1, train, val, resources)
            state = {"config": config, "config_digest": fingerprint, "stage": stage,
                     "epoch": epoch, "history": history, "best_score": best_score,
                     "best_epoch": best_epoch, "recipe": recipe,
                     "subset_sha256": manifest["indices_sha256"], "wandb_run_id": tracker.id,
                     "smoke": smoke,
                     "scheduler": {"kind": "warmup_cosine", "next_epoch": epoch + 1}}
            save_checkpoint(last_path, model, optimizer, scaler, state)
            if pretrain and (epoch + 1) % config["training"]["preview_interval"] == 0:
                from ct_mae.visuals import save_reconstruction
                preview = directory / "reconstruction.png"
                save_reconstruction(model, val_data, device, config, preview)
                tracker.image(preview, "reconstruction", "Original / masked / decoder / composite")
            if (epoch + 1) % config["training"]["checkpoint_interval"] == 0:
                tracker.checkpoint(directory, epoch + 1)
            print(f"{stage} {epoch + 1}/{recipe['epochs']} "
                  f"train={train['loss']:.5f} val={val['loss']:.5f} "
                  f"val_macro_f1={val.get('macro_f1', 0):.4f} "
                  f"seconds={train['seconds']:.1f}", flush=True)
        status = "smoke_completed" if smoke else (
            "completed" if stop >= recipe["epochs"] else "partial"
        )
        metadata.update(status=status, seconds_this_session=time.perf_counter() - started)
        write_json(directory / "run.json", metadata)
        tracker.checkpoint(directory, metadata["completed_epochs"])
        return metadata
    except Exception as error:
        metadata.update(status="failed", error_type=type(error).__name__)
        write_json(directory / "run.json", metadata)
        raise
    finally:
        tracker.finish(status, {"best_epoch": best_epoch, "best_validation_score": best_score})


def evaluate_stage(stage: str, config: dict, archive: Path, manifest: dict, output: Path,
                   device: torch.device) -> dict:
    directory = output / stage
    metadata = json.loads((directory / "run.json").read_text("utf-8"))
    if metadata["status"] != "completed" or metadata["smoke"]:
        raise ValueError("Test evaluation requires a completed full run")
    if metadata["config_digest"] != config_digest(config):
        raise ValueError("Evaluation configuration differs from the completed run")
    if metadata["subset_sha256"] != manifest["indices_sha256"]:
        raise ValueError("Evaluation subset differs from the completed run")
    target = directory / "test.json"
    if target.exists():
        return json.loads(target.read_text("utf-8"))
    model, _ = build_model(stage, config, output, device)
    load_checkpoint(directory / "best.pt", model, restore_random=False)
    images, labels = read_split(archive, "test", supervised=True)
    loader = make_loader(OrganDataset(images, labels), metadata["recipe"]["batch_size"],
                         config["training"]["num_workers"], seed=config["seed"], shuffle=False)
    result, predictions = evaluate(
        model, loader, device=device, pretrain=False, seed=config["seed"],
        amp=config["training"]["amp"] and device.type == "cuda",
        num_classes=config["dataset"]["num_classes"], return_predictions=True,
    )
    np.savez_compressed(directory / "test_predictions.npz", **predictions)
    result.update(selected_epoch=metadata["best_epoch"], checkpoint_sha256=sha256_file(
        directory / "best.pt"), subset_sha256=manifest["indices_sha256"])
    tracker = RunTracker(config, stage, directory, run_id=metadata["wandb_run_id"])
    if tracker.run is not None:
        tracker.run.summary.update({f"test/{key}": value for key, value in result.items()
                                   if isinstance(value, (float, int))})
    tracker.finish("completed")
    write_json(target, result)
    return result
