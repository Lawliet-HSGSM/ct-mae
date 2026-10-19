"""Run the frozen CT-MAE recipe, diagnostics, final evaluation or offline reporting."""

import argparse
import copy
import json
from pathlib import Path

import torch

from ct_mae.config import config_digest, load_config, write_json
from ct_mae.data import prepare_dataset
from ct_mae.models import MaskedAutoencoder
from ct_mae.pipeline import evaluate_stage, run_stage
from ct_mae.training_report import build_report
from ct_mae.visuals import save_input_preview, save_reconstruction


def diagnostic_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result["run_kind"] = "smoke"
    for recipe in result["stages"].values():
        recipe.update(epochs=2, warmup_epochs=0)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "pretrain", "scratch", "linear", "finetune",
                                           "evaluate", "report", "all"], required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/baseline"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args(argv)
    if args.stop_after_epoch is not None and (
        args.stop_after_epoch < 1 or args.stage not in ("pretrain", "scratch", "linear", "finetune")
    ):
        parser.error("--stop-after-epoch requires one training stage and a positive epoch")
    config = load_config(args.config)
    if args.wandb_mode:
        config["wandb"]["mode"] = args.wandb_mode
    output = args.output / "smoke" if args.smoke else args.output
    if args.smoke:
        config = diagnostic_config(config)
    if args.stage == "report":
        recorded = output / "resolved_config.json"
        if args.config is None and recorded.exists():
            config = load_config(recorded)
        print(build_report(output, config))
        return
    output.mkdir(parents=True, exist_ok=True)
    recorded = output / "resolved_config.json"
    if recorded.exists() and config_digest(json.loads(recorded.read_text("utf-8"))) != (
        config_digest(config)
    ):
        raise ValueError("Output already contains a different configuration; use a new output path")
    write_json(recorded, config)
    manifest = prepare_dataset(args.data_dir, output, config)
    archive = args.data_dir / manifest["filename"]
    if not (output / "input_preview.png").exists():
        save_input_preview(archive, manifest, config, output / "input_preview.png")
    if not (output / "model_summary.json").exists():
        model = MaskedAutoencoder(config["model"])
        encoder = sum(parameter.numel() for parameter in model.encoder.parameters())
        total = sum(parameter.numel() for parameter in model.parameters())
        write_json(output / "model_summary.json", {"encoder_parameters": encoder,
                   "decoder_parameters": total - encoder, "total_parameters": total})
        del model
    if args.stage == "prepare":
        print(build_report(output, config))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; use --device cpu only for local diagnostics")
    stages = ["scratch", "pretrain", "linear", "finetune"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "evaluate":
            break
        run_stage(stage, config, archive, manifest, output, device,
                  resume=args.resume and (output / stage / "last.pt").exists(),
                  smoke=args.smoke, stop_after_epoch=args.stop_after_epoch,
                  train_limit=512 if args.smoke else None, val_limit=128 if args.smoke else None)
        if stage == "pretrain":
            from ct_mae.data import OrganDataset, read_split
            from ct_mae.engine import load_checkpoint
            model = MaskedAutoencoder(config["model"]).to(device)
            load_checkpoint(output / "pretrain" / "best.pt", model, restore_random=False)
            images, _ = read_split(archive, "val", supervised=False)
            save_reconstruction(model, OrganDataset(images), device, config,
                                output / "pretrain" / "reconstruction.png")
            del model
        build_report(output, config)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.stage in ("all", "evaluate") and not args.smoke:
        for stage in ("scratch", "linear", "finetune"):
            print(stage, evaluate_stage(stage, config, archive, manifest, output, device))
    print(build_report(output, config))


if __name__ == "__main__":
    main()
