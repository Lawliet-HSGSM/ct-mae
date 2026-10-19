"""Fixed-scale previews; reconstruction composites explicitly distinguish copied pixels."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ct_mae.data import OrganDataset, read_split
from ct_mae.download import METADATA
from ct_mae.engine import fixed_noise
from ct_mae.models import patchify, random_masking, unpatchify


def image_grid(rows, row_labels: list[str], path: Path, title: str, columns=None) -> None:
    count = len(rows[0])
    figure, axes = plt.subplots(len(rows), count, figsize=(count * 2.1, len(rows) * 2.25),
                                squeeze=False, layout="constrained")
    for row, label in enumerate(row_labels):
        for column in range(count):
            axes[row, column].imshow(rows[row][column], cmap="gray", vmin=0, vmax=1,
                                      interpolation="nearest")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if column == 0:
                axes[row, column].set_ylabel(label)
            if row == 0 and columns:
                axes[row, column].set_title(columns[column], fontsize=10)
    figure.suptitle(title, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def pixel_mask(mask: torch.Tensor, config: dict) -> torch.Tensor:
    patch_pixels = config["patch_size"]**2 * config["in_channels"]
    return unpatchify(mask[:, :, None].expand(-1, -1, patch_pixels),
                      config["patch_size"], config["in_channels"])


def save_input_preview(archive: Path, manifest: dict, config: dict, path: Path) -> None:
    images, labels = read_split(archive, "train", supervised=True)
    indices = np.array(manifest["indices"])
    preferred = [3, 4, 6, 9, 10]
    selected = [int(indices[np.flatnonzero(labels[indices] == label)[0]])
                for label in preferred if np.any(labels[indices] == label)]
    original = OrganDataset(images, indices=np.array(selected))
    augmented = OrganDataset(images, indices=np.array(selected), augment=True,
                             seed=config["seed"], **config["augmentation"])
    raw = torch.stack([original[i]["image"] for i in range(len(original))])
    target = torch.stack([augmented[i]["image"] for i in range(len(augmented))])
    patches = patchify(target, config["model"]["patch_size"])
    noise = fixed_noise(torch.tensor(selected), patches.shape[1], config["seed"])
    _, mask, _ = random_masking(patches, config["model"]["mask_ratio"], noise)
    masked = target * (1 - pixel_mask(mask, config["model"]))
    image_grid([raw[:, 0], target[:, 0], masked[:, 0]],
               ["Original", "Affine target", "75% masked input"], path,
               "Real train samples: light augmentation before masking",
               [f"{METADATA['labels'][labels[index]]}\ntrain #{index}" for index in selected])


@torch.inference_mode()
def save_reconstruction(model, dataset: OrganDataset, device, config: dict, path: Path) -> None:
    positions = np.random.default_rng(config["seed"]).choice(
        len(dataset), size=min(5, len(dataset)), replace=False,
    )
    items = [dataset[int(position)] for position in positions]
    images = torch.stack([item["image"] for item in items]).to(device)
    indices = torch.tensor([item["index"] for item in items])
    patches = (config["model"]["image_size"] // config["model"]["patch_size"])**2
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=config["training"]["amp"] and device.type == "cuda"):
        loss, predicted, mask = model(images, fixed_noise(indices, patches, config["seed"]))
    restored = unpatchify(predicted.float(), config["model"]["patch_size"], 1)
    missing = pixel_mask(mask, config["model"])
    masked = images * (1 - missing)
    composite = masked + restored * missing
    image_grid([image[:, 0].float().cpu() for image in (images, masked, restored, composite)],
               ["Original", "Masked input", "Decoder output", "Visible + predicted"], path,
               f"Validation reconstruction | masked MSE {loss.item():.5f}\n"
               "Bottom row copies visible pixels; only missing pixels are predictions",
               [f"val #{index}" for index in indices.tolist()])
