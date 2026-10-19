"""Supervision boundaries, reproducible sampling and modest spatial augmentation."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, v2

from ct_mae.config import write_json
from ct_mae.download import METADATA, fetch_asset, official_asset


def select_labeled_indices(labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Allocate the exact budget proportionally, using largest remainders."""
    labels = np.asarray(labels).reshape(-1)
    if not 0 < count <= len(labels):
        raise ValueError("Labeled count must be between one and the number of train images")
    classes, frequencies = np.unique(labels, return_counts=True)
    quotas = frequencies * (count / len(labels))
    allocation = np.floor(quotas).astype(int)
    remainder = count - allocation.sum()
    priority = np.argsort(-(quotas - allocation), kind="stable")
    allocation[priority[:remainder]] += 1
    rng = np.random.default_rng(seed)
    indices = np.concatenate([
        rng.choice(np.flatnonzero(labels == label), size=int(size), replace=False)
        for label, size in zip(classes, allocation, strict=True)
    ])
    return np.sort(indices).astype(np.int64)


class OrganDataset(Dataset):
    """An SSL instance never receives labels; targets are the complete augmented image."""

    def __init__(
        self, images: np.ndarray, labels: np.ndarray | None = None,
        indices: np.ndarray | None = None, *, augment: bool = False, seed: int = 42,
        degrees: float = 5.0, translate: float = 0.05,
    ):
        if images.dtype != np.uint8 or images.ndim != 3:
            raise ValueError("Expected uint8 grayscale images with shape [N,H,W]")
        self.images = images
        self.labels = None if labels is None else np.asarray(labels).reshape(-1)
        if self.labels is not None and len(self.labels) != len(images):
            raise ValueError("Image and label counts must agree")
        self.indices = np.arange(len(images)) if indices is None else np.asarray(indices)
        self.seed, self.epoch = seed, 0
        # https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomAffine.html
        self.transform = v2.RandomAffine(
            degrees=degrees, translate=(translate, translate),
            interpolation=InterpolationMode.BILINEAR, fill=0,
        ) if augment else None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict:
        index = int(self.indices[position])
        image = torch.from_numpy(self.images[index].copy()).unsqueeze(0).float().div_(255)
        if self.transform is not None:
            # Per-image/epoch randomness also survives a change in batch size or worker order.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed + self.epoch * len(self.images) + index)
                image = self.transform(image)
        item = {"image": image, "index": index}
        if self.labels is not None:
            item["label"] = int(self.labels[index])
        return item


def prepare_dataset(data_dir: Path, output_dir: Path, config: dict) -> dict:
    asset = official_asset(config["dataset"]["size"])
    path = fetch_asset(asset, data_dir)
    with np.load(path, allow_pickle=False) as archive:
        labels = archive["train_labels"].reshape(-1)
        indices = select_labeled_indices(labels, config["dataset"]["labeled_count"], config["seed"])
        counts = {split: len(archive[f"{split}_labels"]) for split in ("train", "val", "test")}
    manifest = {
        "dataset": "OrganAMNIST", "size": config["dataset"]["size"],
        "filename": asset["filename"], "md5": asset["md5"], "source": METADATA["record"],
        "seed": config["seed"], "split_counts": counts,
        "labeled_count": len(indices), "train_label_fraction": len(indices) / len(labels),
        "indices": indices.tolist(),
        "indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "labeled_class_counts": np.bincount(
            labels[indices], minlength=config["dataset"]["num_classes"]
        ).tolist(),
        "label_budget_scope": "train only; full official labeled validation and test retained",
    }
    write_json(output_dir / "subset.json", manifest)
    return manifest


def read_split(path: Path, split: str, *, supervised: bool) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as archive:
        images = archive[f"{split}_images"]
        labels = archive[f"{split}_labels"].reshape(-1) if supervised else None
    return images, labels
