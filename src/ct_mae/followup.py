"""Controlled partial fine-tuning and read-only feature extraction from baseline checkpoints."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from ct_mae.config import config_digest, load_config, write_json
from ct_mae.data import OrganDataset, prepare_dataset, read_split
from ct_mae.engine import evaluate, load_checkpoint, seed_everything
from ct_mae.kaggle import verify_runs
from ct_mae.models import OrganClassifier
from ct_mae.pipeline import evaluate_stage, make_loader, run_stage, sha256_file


def followup_config(baseline: Path, blocks: int = 3) -> dict:
    config = load_config(baseline / "resolved_config.json")
    checkpoint = baseline / "pretrain/best.pt"
    digest = sha256_file(checkpoint)
    reference = json.loads((baseline / "finetune/run.json").read_text("utf-8"))
    if reference["ssl_checkpoint_sha256"] != digest:
        raise ValueError("Baseline full fine-tuning used a different SSL checkpoint")
    config["stages"]["partial"] = copy.deepcopy(config["stages"]["finetune"])
    config["followup"] = {
        "partial_blocks": blocks, "baseline_directory": str(baseline.resolve()),
        "baseline_config_digest": config_digest(config_without_partial(config)),
        "pretrain_sha256": digest,
    }
    return config


def config_without_partial(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.pop("followup", None)
    result["stages"].pop("partial", None)
    return result


def audit_frozen(checkpoint: Path, source: Path, blocks: int) -> dict:
    initial = torch.load(source, map_location="cpu", weights_only=True)
    current = torch.load(checkpoint, map_location="cpu", weights_only=True)
    depth = current["config"]["model"]["encoder_depth"]
    changing = tuple(f"encoder.blocks.{i}." for i in range(depth - blocks, depth))
    checked, changed = [], []
    for key, value in current["model"].items():
        if not key.startswith("encoder."):
            continue
        if key.startswith((*changing, "encoder.norm.")):
            if not torch.equal(value, initial["model"][key]):
                changed.append(key)
        else:
            if not torch.equal(value, initial["model"][key]):
                raise ValueError(f"A frozen parameter changed: {key}")
            checked.append(key)
    if not changed:
        raise ValueError("No selected encoder parameter was updated")
    return {"frozen_tensors_unchanged": len(checked), "updated_tensors": len(changed),
            "trainable_blocks": list(range(depth - blocks, depth)),
            "checkpoint_sha256": sha256_file(checkpoint), "passed": True}


def train_partial(baseline: Path, data_dir: Path, output: Path, device: torch.device,
                  *, blocks: int = 3, smoke: bool = False) -> dict:
    config = followup_config(baseline, blocks)
    if smoke:
        config["followup"]["diagnostic"] = True
        config["stages"]["partial"].update(epochs=2, warmup_epochs=0)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", config)
    manifest = prepare_dataset(data_dir, output, config)
    original = json.loads((baseline / "subset.json").read_text("utf-8"))
    if manifest != original:
        raise ValueError("Follow-up changed the labeled subset or official data")
    record = run_stage(
        "partial", config, data_dir / manifest["filename"], manifest, output, device,
        smoke=smoke, train_limit=128 if smoke else None, val_limit=128 if smoke else None,
        resume=(output / "partial/last.pt").exists(),
    )
    audit = audit_frozen(output / "partial/best.pt", baseline / "pretrain/best.pt", blocks)
    write_json(output / "freeze_audit.json", audit)
    if not smoke:
        evaluate_stage("partial", config, data_dir / manifest["filename"], manifest,
                       output, device)
    return record


def analysis_model(stage: str, baseline: Path, output: Path, device: torch.device):
    config = load_config(baseline / "resolved_config.json")
    seed_everything(config["seed"])
    model = OrganClassifier(config["model"], config["dataset"]["num_classes"],
                            seed=config["seed"])
    source = None
    if stage == "mae":
        source = baseline / "pretrain/best.pt"
        state = torch.load(source, map_location="cpu", weights_only=True)["model"]
        model.encoder.load_state_dict({key.removeprefix("encoder."): value
                                       for key, value in state.items()
                                       if key.startswith("encoder.")})
    elif stage != "random":
        source = (output if stage == "partial" else baseline) / stage / "best.pt"
        load_checkpoint(source, model, restore_random=False)
    model.eval()
    return model.to(device), {"checkpoint_sha256": sha256_file(source) if source else None,
                              "random_seed": config["seed"] if source is None else None}


@torch.inference_mode()
def extract_features(model, images: np.ndarray, device: torch.device, *, batch_size: int,
                     workers: int, seed: int) -> np.ndarray:
    loader = make_loader(OrganDataset(images), batch_size, workers, seed=seed, shuffle=False)
    features = []
    for batch in loader:
        inputs = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            tokens, _, _ = model.encoder(inputs)
            pooled = tokens[:, 1:].mean(dim=1)
        features.append(pooled.float().cpu().numpy())
    return np.concatenate(features)


def extract_analysis(baseline: Path, archive: Path, output: Path, device: torch.device,
                     *, stages: tuple[str, ...] = ("random", "mae", "scratch", "finetune",
                                                  "partial")) -> dict:
    config = load_config(baseline / "resolved_config.json")
    manifest = json.loads((baseline / "subset.json").read_text("utf-8"))
    settings = config["training"]
    directory = output / "embeddings"
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {"pooling": "mean of all patch tokens after final LayerNorm; no CLS",
                  "augmentation": False, "model_mode": "eval",
                  "dimension": config["model"]["encoder_dim"],
                  "train_labels_used_for_embeddings": False,
                  "amp_fp16": device.type == "cuda", "stages": {}}
    for stage in stages:
        started = time.perf_counter()
        model, source = analysis_model(stage, baseline, output, device)
        for split in ("train", "test"):
            path = directory / f"{stage}_{split}.npz"
            if not path.exists():
                images, _ = read_split(archive, split, supervised=False)
                values = extract_features(model, images, device, batch_size=64,
                                          workers=settings["num_workers"], seed=config["seed"])
                np.savez_compressed(path, features=values, index=np.arange(len(values)))
                del images, values
            with np.load(path, allow_pickle=False) as saved:
                shape = list(saved["features"].shape)
            source[split] = {"shape": shape, "sha256": sha256_file(path)}
            print(f"EMBEDDING {stage}/{split}: {shape}", flush=True)
        provenance["stages"][stage] = {**source, "seconds": time.perf_counter() - started}
        write_json(output / "embedding_provenance.json", provenance)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    diagnostics = {}
    for stage in ("scratch", "linear", "finetune", "partial"):
        model, source = analysis_model(stage, baseline, output, device)
        measurements = {}
        for split, augment in (("train", False), ("train", True), ("val", False)):
            images, labels = read_split(archive, split, supervised=True)
            indices = np.array(manifest["indices"]) if split == "train" else None
            dataset = OrganDataset(images, labels, indices, augment=augment, seed=config["seed"])
            loader = make_loader(dataset, 64, settings["num_workers"],
                                 seed=config["seed"], shuffle=False)
            values, predictions = evaluate(
                model, loader, device=device, pretrain=False, seed=config["seed"],
                amp=device.type == "cuda", return_predictions=True,
            )
            name = f"{split}_{'augmented' if augment else 'clean'}"
            measurements[name] = values
            np.savez_compressed(output / f"{stage}_{name}_predictions.npz", **predictions)
            del images, labels, dataset, loader
        diagnostics[stage] = {"source": source, "measurements": measurements,
                              "all_measurements_at_fixed_checkpoint_in_eval_mode": True,
                              "augmentation_seed": config["seed"], "augmentation_epoch": 0}
        write_json(output / "loss_diagnostics.json", diagnostics)
        print(f"LOSS DIAGNOSTIC {stage}: {measurements}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["train", "smoke", "extract", "all"], default="all")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the full follow-up")
    config = load_config(args.baseline / "resolved_config.json")
    if args.mode in ("train", "smoke", "all"):
        smoke = args.mode == "smoke"
        train_partial(args.baseline, args.data_dir, args.output, device,
                      blocks=args.blocks, smoke=smoke)
        if config["wandb"]["mode"] == "online":
            import wandb
            api = wandb.Api()
            verify_runs(args.output, api, config["wandb"]["entity"] or api.default_entity,
                        config["wandb"]["project"], ["partial"], smoke=smoke)
    if args.mode in ("extract", "all"):
        original = json.loads((args.baseline / "subset.json").read_text("utf-8"))
        extract_analysis(args.baseline, args.data_dir / original["filename"], args.output, device)


if __name__ == "__main__":
    main()
