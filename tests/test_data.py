import numpy as np
import torch

from ct_mae.data import OrganDataset, select_labeled_indices


def test_subset_is_exact_stratified_reproducible_and_without_replacement():
    labels = np.repeat(np.arange(11), np.arange(1, 12) * 100)
    selected = select_labeled_indices(labels, 330, seed=42)
    assert len(selected) == len(set(selected.tolist())) == 330
    np.testing.assert_array_equal(selected, select_labeled_indices(labels, 330, seed=42))
    proportions = np.bincount(labels[selected], minlength=11)
    assert np.max(np.abs(proportions - np.bincount(labels) * 330 / len(labels))) <= 1


def test_ssl_dataset_returns_only_image_and_index_with_preserved_pixel_scale():
    images = np.array([[[0, 128], [255, 64]]], dtype=np.uint8)
    item = OrganDataset(images)[0]
    assert set(item) == {"image", "index"}
    assert item["image"].shape == (1, 2, 2)
    assert item["image"].dtype == torch.float32
    torch.testing.assert_close(item["image"][0], torch.tensor(images[0] / 255, dtype=torch.float32))


def test_augmentation_varies_by_epoch_and_resumes_independently_of_loader_order():
    images = np.arange(32 * 32, dtype=np.uint8).reshape(1, 32, 32)
    dataset = OrganDataset(images, augment=True, seed=42)
    first = dataset[0]["image"]
    dataset.set_epoch(1)
    second = dataset[0]["image"]
    assert not torch.equal(first, second)
    restored = OrganDataset(images, augment=True, seed=42)
    restored.set_epoch(1)
    torch.testing.assert_close(second, restored[0]["image"])
    assert second.shape == (1, 32, 32)
    assert 0 <= second.min() <= second.max() <= 1
