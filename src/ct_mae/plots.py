"""Static inspection plots with an explicit, fixed grayscale display range."""

import base64
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def image_array(example: dict, size: int) -> np.ndarray:
    encoded = example["images"][str(size)]["png"].split(",", 1)[1]
    return np.array(Image.open(BytesIO(base64.b64decode(encoded))))


def write_plots(data: dict, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    examples = data["examples"]["train"]
    first = {example["label"]: example for example in examples[::-1]}
    size = 64 if 64 in data["sizes"] else data["sizes"][-1]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), layout="constrained")
    for class_id, axis in enumerate(axes.flat):
        axis.set_axis_off()
        if class_id not in first:
            continue
        example = first[class_id]
        axis.imshow(
            image_array(example, size), cmap="gray", vmin=0, vmax=255, interpolation="nearest"
        )
        axis.set_title(
            f"{data['labels_vi'][class_id]} · {data['metadata']['labels'][class_id]}\n"
            f"train #{example['index']} · {size}×{size}"
        )
    fig.suptitle("OrganAMNIST — 11 lớp có nhãn supervised", fontsize=17)
    fig.savefig(output / "organ_samples.png", dpi=150)
    plt.close(fig)

    classes = [3, 4, 6, 9]
    fig, axes = plt.subplots(
        len(data["sizes"]),
        4,
        figsize=(12, 3 * len(data["sizes"])),
        squeeze=False,
        layout="constrained",
    )
    for row, resolution in enumerate(data["sizes"]):
        for column, class_id in enumerate(classes):
            axis = axes[row, column]
            example = first[class_id]
            axis.imshow(
                image_array(example, resolution),
                cmap="gray",
                vmin=0,
                vmax=255,
                interpolation="nearest",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"{data['labels_vi'][class_id]} · train #{example['index']}")
            if column == 0:
                axis.set_ylabel(f"{resolution}×{resolution}\n{resolution**2:,} pixels")
    fig.suptitle("Cùng index ở các bản chính thức — cùng kích thước hiển thị", fontsize=15)
    fig.savefig(output / "resolution_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    baseline = data["resolutions"][str(data["sizes"][0])]["splits"]
    x = np.arange(11)
    for offset, split in enumerate(("train", "val", "test")):
        axes[0].bar(
            x + (offset - 1) * 0.25, baseline[split]["class_counts"], width=0.25, label=split
        )
    axes[0].set_xticks(x, data["metadata"]["labels"], rotation=65, ha="right")
    axes[0].set_ylabel("Số ảnh có nhãn")
    axes[0].set_title("Phân bố 11 lớp theo split")
    axes[0].legend()
    for resolution in data["sizes"]:
        histogram = np.array(data["resolutions"][str(resolution)]["splits"]["train"]["histogram"])
        axes[1].plot(
            np.arange(256), histogram / histogram.sum(), label=f"{resolution}×{resolution}"
        )
    axes[1].set_xlabel("Pixel gốc (0–255)")
    axes[1].set_ylabel("Tỷ lệ pixel")
    axes[1].set_title("Histogram toàn bộ train — chưa chuẩn hóa")
    axes[1].legend()
    fig.savefig(output / "dataset_overview.png", dpi=150)
    plt.close(fig)
