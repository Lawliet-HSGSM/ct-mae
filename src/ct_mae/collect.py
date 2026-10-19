"""Read verified NPZ archives and collect measurements and reproducible examples."""

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from ct_mae.download import METADATA, file_md5, official_asset
from ct_mae.inspect import image_properties, sample_indices, summarize_split, verify_alignment

SPLITS = ("train", "val", "test")
LABELS_VI = [
    "Bàng quang",
    "Xương đùi trái",
    "Xương đùi phải",
    "Tim",
    "Thận trái",
    "Thận phải",
    "Gan",
    "Phổi trái",
    "Phổi phải",
    "Tụy",
    "Lách",
]


def png_data_url(image: np.ndarray) -> str:
    stream = BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def collect_dataset(data_dir: Path, sizes: list[int], per_class: int, seed: int) -> dict:
    if not sizes:
        raise ValueError("At least one resolution is required")
    result = {
        "metadata": METADATA,
        "labels_vi": LABELS_VI,
        "sizes": sizes,
        "seed": seed,
        "samples_per_class": per_class,
        "resolutions": {},
        "examples": {split: [] for split in SPLITS},
        "warnings": [],
        "label_alignment": len(sizes) > 1,
    }
    reference_labels = {}
    chosen_indices = {}
    for size in sizes:
        asset = official_asset(size)
        path = data_dir / asset["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run python -m ct_mae.download first.")
        if path.stat().st_size != asset["bytes"] or file_md5(path) != asset["md5"]:
            raise ValueError(f"Checksum mismatch: {path}")
        resolution = {"asset": asset, "checksum_verified": True, "splits": {}}
        print(f"Inspecting verified {path.name}", flush=True)
        with np.load(path, allow_pickle=False) as archive:
            for split in SPLITS:
                images, labels = archive[f"{split}_images"], archive[f"{split}_labels"]
                stats = summarize_split(images, labels)
                if stats["image_shape"] != [size, size]:
                    raise ValueError(f"Unexpected stored shape in {path.name}: {stats['shape']}")
                if split in reference_labels:
                    verify_alignment(reference_labels[split], labels)
                else:
                    reference_labels[split] = labels.copy()
                    chosen_indices[split] = sample_indices(labels, per_class, seed)
                    result["examples"][split] = [
                        {"index": index, "label": int(labels.reshape(-1)[index]), "images": {}}
                        for index in chosen_indices[split]
                    ]
                expected = METADATA["expected_samples"][split]
                if stats["samples"] != expected:
                    result["warnings"].append(
                        f"{size}/{split}: archive has {stats['samples']} images; "
                        f"metadata {METADATA['metadata_version']} says {expected}. "
                        "Measurements use the unchanged archive."
                    )
                for example in result["examples"][split]:
                    image = images[example["index"]]
                    example["images"][str(size)] = {
                        "png": png_data_url(image),
                        "stats": image_properties(image),
                    }
                resolution["splits"][split] = stats
                print(
                    f"  {split}: {stats['shape']}, {stats['dtype']}, "
                    f"labels {stats['labels_shape']}, pixels {stats['min']}..{stats['max']}",
                    flush=True,
                )
                del images, labels
        result["resolutions"][str(size)] = resolution
    return result
