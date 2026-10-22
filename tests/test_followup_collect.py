from pathlib import Path

import pytest

from ct_mae.followup_collect import selected_path


@pytest.mark.parametrize('filename', [
    '../ct-mae-followup/config.json', 'ct-mae-followup/../config.json',
    '/ct-mae-followup/config.json', 'ct-mae-followup/C:/config.json',
    'ct-mae-followup/wandb/key.json', 'ct-mae-followup\\key.json',
    'ct-mae-followup/smoke/partial/last.pt', 'ct-mae-followup-source/baseline/best.pt',
])
def test_collector_rejects_unrelated_or_escaping_paths(filename):
    assert selected_path(filename) is None


def test_collector_retains_evidence_and_resume_checkpoint():
    assert selected_path('ct-mae-followup/partial/last.pt') == Path('partial/last.pt')
    assert selected_path('ct-mae-followup/embeddings/mae_test.npz') == Path(
        'embeddings/mae_test.npz',
    )


def test_collector_uses_explicit_study_root_without_accepting_other_outputs():
    name = 'ct-mae-labels10pct/partial/best.pt'
    assert selected_path(name) is None
    assert selected_path(name, root_name='ct-mae-labels10pct') == Path('partial/best.pt')
    assert selected_path('ct-mae-followup/partial/best.pt',
                         root_name='ct-mae-labels10pct') is None
