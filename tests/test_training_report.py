import json

from ct_mae.config import load_config
from ct_mae.training_report import build_report, configuration_rows


def test_report_explains_config_and_does_not_invent_results_for_pending_runs(tmp_path):
    config = load_config()
    build_report(tmp_path, config)
    page = (tmp_path / "training_report.html").read_text("utf-8")
    assert "Chưa có kết quả test" in page
    assert "gradient accumulation" in page
    assert "<table" in page
    assert (tmp_path / "training_report.md").exists()
    config["model"]["mask_ratio"] = 0.5
    assert any("50%" in row["value"] for row in configuration_rows(config))


def test_report_displays_logged_metrics_only_for_completed_full_runs(tmp_path):
    config = load_config()
    stage = tmp_path / "scratch"
    stage.mkdir()
    metadata = {
        "stage": "scratch",
        "status": "partial",
        "smoke": False,
        "config": config,
        "recipe": config["stages"]["scratch"],
        "history": [],
        "completed_epochs": 1,
    }
    (stage / "run.json").write_text(json.dumps(metadata))
    (stage / "test.json").write_text(
        json.dumps({"accuracy": 0.375, "macro_f1": 0.25, "per_class_f1": [0.25] * 11})
    )
    build_report(tmp_path, config)
    assert "37.50%" not in (tmp_path / "training_report.html").read_text("utf-8")
    metadata.update(status="completed", completed_epochs=100)
    (stage / "run.json").write_text(json.dumps(metadata))
    build_report(tmp_path, config)
    assert "37.50%" in (tmp_path / "training_report.html").read_text("utf-8")
