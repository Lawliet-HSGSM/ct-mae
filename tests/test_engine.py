import random

import numpy as np
import torch

from ct_mae.engine import (
    classification_metrics,
    fixed_noise,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    train_epoch,
)


def test_accumulation_matches_large_batch_including_uneven_final_microbatch():
    seed_everything(42)
    large = torch.nn.Linear(3, 2)
    micro = torch.nn.Linear(3, 2)
    micro.load_state_dict(large.state_dict())
    images, labels = torch.randn(3, 3), torch.tensor([0, 1, 1])
    recipe = {"lr": 0.03, "epochs": 2, "warmup_epochs": 0, "accumulation": 1}
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    train_epoch(large, [{"image": images, "label": labels}],
                torch.optim.SGD(large.parameters(), lr=0.03), scaler,
                device=torch.device("cpu"), recipe=recipe, epoch=0, min_lr=1e-6,
                pretrain=False, amp=False, num_classes=2)
    recipe["accumulation"] = 2
    train_epoch(micro, [{"image": images[:2], "label": labels[:2]},
                        {"image": images[2:], "label": labels[2:]}],
                torch.optim.SGD(micro.parameters(), lr=0.03), scaler,
                device=torch.device("cpu"), recipe=recipe, epoch=0, min_lr=1e-6,
                pretrain=False, amp=False, num_classes=2)
    for expected, actual in zip(large.parameters(), micro.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)


def test_checkpoint_restores_optimizer_rng_and_next_update_exactly(tmp_path):
    seed_everything(42)
    model = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(3, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    images = torch.ones(3, 3)

    def update():
        optimizer.zero_grad()
        model(images).square().mean().backward()
        optimizer.step()

    update()
    save_checkpoint(tmp_path / "last.pt", model, optimizer, scaler, {"epoch": 0})
    expected_random = (random.random(), np.random.rand(), torch.rand(2))
    update()
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    load_checkpoint(tmp_path / "last.pt", model, optimizer, scaler)
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    torch.testing.assert_close(torch.rand(2), expected_random[2], rtol=0, atol=0)
    update()
    for key, actual in model.state_dict().items():
        torch.testing.assert_close(expected[key], actual, rtol=0, atol=0)


def test_validation_masks_do_not_depend_on_batching_or_global_rng():
    first = fixed_noise(torch.tensor([7, 3]), 256, seed=42)
    torch.rand(100)
    second = torch.cat([fixed_noise(torch.tensor([7]), 256, seed=42),
                        fixed_noise(torch.tensor([3]), 256, seed=42)])
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_macro_f1_counts_missing_predicted_classes_and_returns_confusion():
    metrics = classification_metrics([0, 0, 1, 2], [0, 0, 0, 0], 3)
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == 2 / 9
    assert np.array(metrics["confusion_matrix"]).sum() == 4
