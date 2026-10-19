import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ct_mae.config import load_config
from ct_mae.pipeline import build_model, evaluate_stage, run_stage

torch.set_num_threads(2)


def experiment_fixture(tmp_path):
    config = load_config()
    config["model"].update(image_size=16, patch_size=4, encoder_dim=32, encoder_depth=1,
                           encoder_heads=4, decoder_dim=16, decoder_depth=1, decoder_heads=4)
    config["dataset"].update(size=16, labeled_count=22)
    config["wandb"]["mode"] = "disabled"
    config["training"].update(num_workers=0, amp=False, preview_interval=100)
    for recipe in config["stages"].values():
        recipe.update(epochs=2, batch_size=4, accumulation=2, warmup_epochs=0)
    rng = np.random.default_rng(42)
    arrays = {}
    for split, count in [("train", 44), ("val", 11), ("test", 11)]:
        arrays[f"{split}_images"] = rng.integers(0, 256, (count, 16, 16), dtype=np.uint8)
        arrays[f"{split}_labels"] = (np.arange(count) % 11).reshape(-1, 1)
    archive = tmp_path / "test-fixture.npz"
    np.savez(archive, **arrays)
    manifest = {"indices": list(range(22)), "indices_sha256": "unit-test-subset",
                "labeled_count": 22, "split_counts": {"train": 44, "val": 11, "test": 11}}
    return config, archive, manifest


def test_pipeline_resume_requires_same_protocol_and_test_waits_for_completion(tmp_path):
    config, archive, manifest = experiment_fixture(tmp_path)
    output = tmp_path / "experiment"
    first = run_stage("scratch", config, archive, manifest, output,
                      torch.device("cpu"), stop_after_epoch=1)
    assert first["status"] == "partial"
    with pytest.raises(ValueError, match="completed"):
        evaluate_stage("scratch", config, archive, manifest, output, torch.device("cpu"))
    changed = copy.deepcopy(config)
    changed["stages"]["scratch"]["lr"] *= 2
    with pytest.raises(ValueError, match="configuration"):
        run_stage("scratch", changed, archive, manifest, output, torch.device("cpu"), resume=True)
    completed = run_stage("scratch", config, archive, manifest, output,
                          torch.device("cpu"), resume=True)
    assert completed["status"] == "completed"
    assert completed["completed_epochs"] == 2
    assert len(completed["history"]) == 2
    result = evaluate_stage("scratch", config, archive, manifest, output, torch.device("cpu"))
    assert result["samples"] == 11
    assert (output / "scratch" / "test_predictions.npz").is_file()


def test_both_transfer_experiments_load_identical_best_ssl_weights(tmp_path):
    config, archive, manifest = experiment_fixture(tmp_path)
    output = tmp_path / "experiment"
    for recipe in config["stages"].values():
        recipe.update(epochs=1)
    run_stage("pretrain", config, archive, manifest, output, torch.device("cpu"))
    linear = run_stage("linear", config, archive, manifest, output, torch.device("cpu"))
    finetune = run_stage("finetune", config, archive, manifest, output, torch.device("cpu"))
    assert linear["ssl_checkpoint_sha256"] == finetune["ssl_checkpoint_sha256"]
    assert linear["parameters"]["trainable"] < finetune["parameters"]["trainable"]


def test_transfer_requires_the_completed_pretraining_budget(tmp_path):
    config, archive, manifest = experiment_fixture(tmp_path)
    output = tmp_path / "experiment"
    run_stage("pretrain", config, archive, manifest, output,
              torch.device("cpu"), stop_after_epoch=1)
    for stage in ("linear", "finetune"):
        with pytest.raises(ValueError, match="completed pretraining"):
            build_model(stage, config, output, torch.device("cpu"))


def test_wandb_resume_accepts_changed_runtime_metadata(tmp_path, monkeypatch):
    from wandb.sdk.wandb_config import Config

    import ct_mae.pipeline as pipeline

    shared_config = Config()

    class LocalTracker:
        id, url = "local-resume-test", None

        def __init__(self, *args, **kwargs):
            self.run = SimpleNamespace(config=shared_config)

        def log_epoch(self, *args):
            pass

        def checkpoint(self, *args):
            pass

        def finish(self, *args):
            pass

    config, archive, manifest = experiment_fixture(tmp_path)
    output = tmp_path / "experiment"
    monkeypatch.setattr(pipeline, "RunTracker", LocalTracker)
    monkeypatch.setattr(pipeline, "environment_info", lambda _: {"gpu": "P100"})
    run_stage("scratch", config, archive, manifest, output,
              torch.device("cpu"), stop_after_epoch=1)
    monkeypatch.setattr(pipeline, "environment_info", lambda _: {"gpu": "T4"})
    completed = run_stage("scratch", config, archive, manifest, output,
                          torch.device("cpu"), resume=True)
    assert completed["status"] == "completed"
    assert shared_config["environment"] == {"gpu": "T4"}
