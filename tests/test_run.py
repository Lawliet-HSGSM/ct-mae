from ct_mae.config import load_config
from ct_mae.run import diagnostic_config


def test_diagnostic_budget_cannot_mutate_the_full_experiment_recipe():
    original = load_config()
    diagnostic = diagnostic_config(original)
    assert original["stages"]["pretrain"]["epochs"] == 200
    assert diagnostic["stages"]["pretrain"]["epochs"] == 2
    assert diagnostic["model"] == original["model"]
    assert diagnostic["run_kind"] == "smoke"
