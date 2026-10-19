import numpy as np
import pytest

from ct_mae.inspect import image_properties, sample_indices, summarize_split, verify_alignment


def test_statistics_use_original_uint8_pixels_and_count_every_label():
    images = np.array([[[0, 0], [128, 255]], [[255, 128], [64, 0]]], dtype=np.uint8)
    labels = np.array([[0], [10]], dtype=np.int64)
    stats = summarize_split(images, labels)
    assert stats["shape"] == [2, 2, 2]
    assert stats["all_images_same_shape"] is True
    assert stats["image_shape"] == [2, 2]
    assert stats["dtype"] == "uint8"
    assert stats["labels_shape"] == [2, 1]
    assert stats["class_counts"] == [1] + [0] * 9 + [1]
    assert stats["mean"] == pytest.approx(images.mean())
    assert stats["std"] == pytest.approx(images.std())
    assert stats["p01"] == pytest.approx(np.percentile(images, 1))
    assert stats["p50"] == pytest.approx(np.percentile(images, 50))
    assert stats["p99"] == pytest.approx(np.percentile(images, 99))
    assert stats["histogram"][0] == 3
    assert stats["histogram"][255] == 2
    assert sum(stats["histogram"]) == images.size
    assert stats["image_bytes"] == 4


@pytest.mark.parametrize("labels", [np.array([[0, 1]]), np.array([0]), np.array([0, 11])])
def test_rejects_wrong_label_shape_count_or_class(labels):
    with pytest.raises(ValueError):
        summarize_split(np.zeros((2, 2, 2), dtype=np.uint8), labels)


def test_selected_samples_are_reproducible_and_cover_available_classes():
    labels = np.array([0] * 10 + [6] * 10 + [10] * 10)
    selected = sample_indices(labels, per_class=3, seed=42)
    assert selected == sample_indices(labels, per_class=3, seed=42)
    assert len(set(selected)) == 9
    assert np.bincount(labels[selected], minlength=11).tolist() == [3] + [0] * 5 + [3] + [0] * 3 + [
        3
    ]


def test_resolution_comparison_rejects_reordered_labels():
    verify_alignment(np.array([[0], [6]]), np.array([0, 6]))
    with pytest.raises(ValueError, match="alignment"):
        verify_alignment(np.array([0, 6]), np.array([6, 0]))


def test_image_stats_preserve_range_without_autocontrast():
    stats = image_properties(np.array([[100, 101], [102, 103]], dtype=np.uint8))
    assert stats["min"] == 100
    assert stats["max"] == 103
    assert stats["mean"] == 101.5
