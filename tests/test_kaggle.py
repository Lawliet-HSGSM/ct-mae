import json
from types import SimpleNamespace

import pytest

from ct_mae.kaggle import verify_runs


def test_kaggle_gate_requires_finite_losses_real_updates_and_online_proof(tmp_path):
    directory = tmp_path / "scratch"
    directory.mkdir()
    record = {
        "status": "smoke_completed", "wandb_run_id": "local", "completed_epochs": 2,
        "history": [{"train": {"loss": 2.4, "optimizer_steps": 1, "seconds": 1},
                     "val": {"loss": 2.3}, "resource": {"peak_vram_bytes": 1024}}],
    }
    path = directory / "run.json"
    path.write_text(json.dumps(record))
    remote = SimpleNamespace(url="https://wandb.ai/example/run", summary={
        "preflight/ok": 1, "status": "smoke_completed",
    })
    api = SimpleNamespace(run=lambda _: remote)
    result = verify_runs(tmp_path, api, "example", "ct-mae", ["scratch"], smoke=True)
    assert result[0]["peak_vram_bytes"] == 1024
    record["history"][0]["train"]["optimizer_steps"] = 0
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="optimizer updates"):
        verify_runs(tmp_path, api, "example", "ct-mae", ["scratch"], smoke=True)
    record["history"][0]["train"].update(optimizer_steps=1, loss=float("nan"))
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="non-finite"):
        verify_runs(tmp_path, api, "example", "ct-mae", ["scratch"], smoke=True)
    record["history"][0]["train"]["loss"] = 2.4
    path.write_text(json.dumps(record))
    remote.summary["status"] = "running"
    with pytest.raises(RuntimeError, match="W&B"):
        verify_runs(tmp_path, api, "example", "ct-mae", ["scratch"], smoke=True)
