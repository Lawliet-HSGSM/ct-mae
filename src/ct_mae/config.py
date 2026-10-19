"""One recorded configuration shared by local checks and Kaggle runs."""

import hashlib
import json
from pathlib import Path


def load_config(path: Path | None = None) -> dict:
    config = json.loads((path or Path(__file__).with_name("baseline.json")).read_text("utf-8"))
    model = config["model"]
    if model["image_size"] % model["patch_size"] or model["in_channels"] != 1:
        raise ValueError("Expected square grayscale images divisible by patch size")
    if model["image_size"] != config["dataset"]["size"]:
        raise ValueError("Dataset and model resolution must agree")
    for part in ("encoder", "decoder"):
        if model[f"{part}_dim"] % model[f"{part}_heads"] or model[f"{part}_dim"] % 4:
            raise ValueError("Embedding dimensions must divide heads and support 2D sin/cos")
    if not 0 < model["mask_ratio"] < 1:
        raise ValueError("Pretraining requires visible and masked patches")
    for stage in config["stages"].values():
        if min(stage["epochs"], stage["batch_size"], stage["accumulation"]) < 1:
            raise ValueError("Epochs, batch size and accumulation must be positive")
        if not 0 <= stage["warmup_epochs"] < stage["epochs"]:
            raise ValueError("Warmup must finish before the training budget")
    return config


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    temporary.replace(path)
