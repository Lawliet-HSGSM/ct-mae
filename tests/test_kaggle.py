import ast
import json
from types import SimpleNamespace

import pytest

from ct_mae.kaggle import verify_runs


def test_generated_bundle_quotes_secret_label_and_retains_licenses(tmp_path, monkeypatch):
    import base64
    import io
    import zipfile

    from ct_mae import kaggle

    checkout = tmp_path / 'checkout'
    package = checkout / 'src/ct_mae'
    package.mkdir(parents=True)
    (package / 'example.py').write_text('value = 1', 'utf-8')
    (checkout / 'tests').mkdir()
    (checkout / 'licenses').mkdir()
    for name in ('LICENSE', 'THIRD_PARTY_NOTICES.md', 'licenses/SupContrast-BSD-2-Clause.txt'):
        (checkout / name).write_text('License fixture', 'utf-8')
    # The Kaggle bundle is flat, unlike a source checkout. Test the packager with
    # its own synthetic checkout so this same test also runs inside that bundle.
    monkeypatch.setattr(kaggle, '__file__', str(package / 'kaggle.py'))
    monkeypatch.setattr(kaggle.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(stdout='a' * 40))

    path = tmp_path / 'notebook.py'
    label = "MY_WANDB_KEY'custom"
    kaggle.build_script(path, secret_label=label, smoke=True)
    tree = ast.parse(path.read_text('utf-8'))
    assignments = [node for node in tree.body if isinstance(node, ast.If)
                   and isinstance(node.test, ast.Compare)
                   and isinstance(node.test.left, ast.Constant)
                   and node.test.left.value == label]
    assert len(assignments) == 1
    assert assignments[0].body[0].value.value == label
    payload = next(node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'payload' for t in node.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload.value.args[0].value))) as bundle:
        assert 'LICENSE' in bundle.namelist()
        assert 'THIRD_PARTY_NOTICES.md' in bundle.namelist()
        assert 'licenses/SupContrast-BSD-2-Clause.txt' in bundle.namelist()


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
