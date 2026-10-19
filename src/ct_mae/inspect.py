"""Measure stored pixels without fitting or applying model preprocessing."""

import numpy as np


def _pixel_stats(histogram: np.ndarray) -> dict:
    values = np.arange(256, dtype=np.float64)
    total = int(histogram.sum())
    mean = float(np.dot(histogram, values) / total)
    cumulative = histogram.cumsum()

    def percentile(q: float) -> float:
        rank = (total - 1) * q
        low = int(np.searchsorted(cumulative, int(np.floor(rank)), side="right"))
        high = int(np.searchsorted(cumulative, int(np.ceil(rank)), side="right"))
        return low + (high - low) * (rank - np.floor(rank))

    occupied = np.flatnonzero(histogram)
    return {
        "min": int(occupied[0]),
        "max": int(occupied[-1]),
        "mean": mean,
        "std": float(np.sqrt(np.dot(histogram, (values - mean) ** 2) / total)),
        "p01": percentile(0.01),
        "p50": percentile(0.50),
        "p99": percentile(0.99),
        "zero_fraction": int(histogram[0]) / total,
        "max_fraction": int(histogram[255]) / total,
        "histogram": histogram.tolist(),
    }


def summarize_split(images: np.ndarray, labels: np.ndarray) -> dict:
    if images.ndim != 3 or images.shape[0] == 0 or images.shape[1] != images.shape[2]:
        raise ValueError("Expected nonempty square grayscale images with shape [N,H,W]")
    if images.dtype != np.uint8:
        raise ValueError(f"Expected official uint8 image values, got {images.dtype}")
    if labels.shape not in [(len(images),), (len(images), 1)]:
        raise ValueError("Expected exactly one label per image, shape [N] or [N,1]")
    flat_labels = labels.reshape(-1)
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected integer organ labels")
    if np.any((flat_labels < 0) | (flat_labels > 10)):
        raise ValueError("Organ label IDs must be in [0,10]")
    histogram = np.zeros(256, dtype=np.int64)
    for start in range(0, len(images), 256):
        histogram += np.bincount(images[start : start + 256].ravel(), minlength=256)
    return {
        "samples": len(images),
        "shape": list(images.shape),
        "image_shape": list(images.shape[1:]),
        "all_images_same_shape": True,
        "channels": 1,
        "dtype": str(images.dtype),
        "labels_shape": list(labels.shape),
        "labels_dtype": str(labels.dtype),
        "labeled_samples": len(flat_labels),
        "class_counts": np.bincount(flat_labels, minlength=11).tolist(),
        "image_bytes": int(images[0].nbytes),
        "array_bytes": int(images.nbytes),
        "float32_array_bytes": int(images.size * 4),
        **_pixel_stats(histogram),
    }


def sample_indices(labels: np.ndarray, per_class: int, seed: int) -> list[int]:
    if per_class < 1:
        raise ValueError("samples-per-class must be positive")
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(11):
        candidates = np.flatnonzero(labels.reshape(-1) == class_id)
        selected.extend(
            rng.choice(candidates, min(per_class, len(candidates)), replace=False).tolist()
        )
    return selected


def verify_alignment(reference: np.ndarray, other: np.ndarray) -> None:
    if not np.array_equal(reference.reshape(-1), other.reshape(-1)):
        raise ValueError("Cross-resolution label/index alignment failed; cannot compare samples")


def image_properties(image: np.ndarray) -> dict:
    if image.ndim != 2 or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("Expected a nonempty uint8 grayscale image")
    return {
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "dtype": str(image.dtype),
        "bytes": int(image.nbytes),
        **_pixel_stats(np.bincount(image.ravel(), minlength=256)),
    }
