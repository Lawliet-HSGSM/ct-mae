import json

import numpy as np
import torch
from test_pipeline import experiment_fixture

from ct_mae.config import write_json
from ct_mae.followup import extract_analysis, train_partial
from ct_mae.pipeline import evaluate_stage, run_stage


def test_followup_trains_selected_blocks_and_extracts_without_retraining_baselines(
    tmp_path, monkeypatch,
):
    config, archive, manifest = experiment_fixture(tmp_path)
    for recipe in config['stages'].values():
        recipe.update(epochs=1)
    baseline = tmp_path / 'baseline'
    write_json(baseline / 'resolved_config.json', config)
    manifest['filename'] = archive.name
    write_json(baseline / 'subset.json', manifest)
    device = torch.device('cpu')
    for stage in ('pretrain', 'scratch', 'linear', 'finetune'):
        run_stage(stage, config, archive, manifest, baseline, device)
    before = (baseline / 'finetune/best.pt').read_bytes()

    def prepare(data_dir, output, selected_config):
        write_json(output / 'subset.json', manifest)
        return manifest

    monkeypatch.setattr('ct_mae.followup.prepare_dataset', prepare)
    output = tmp_path / 'followup'
    record = train_partial(baseline, archive.parent, output, device, blocks=1)
    assert record['completed_epochs'] == 1
    assert record['status'] == 'completed'
    assert json.loads((output / 'freeze_audit.json').read_text())['passed']
    assert (output / 'partial/test_predictions.npz').exists()
    provenance = extract_analysis(baseline, archive, output, device)
    assert provenance['dimension'] == config['model']['encoder_dim']
    assert set(provenance['stages']) == {'random', 'mae', 'scratch', 'finetune', 'partial'}
    for stage in provenance['stages']:
        with np.load(output / f'embeddings/{stage}_test.npz') as features:
            assert features['features'].shape == (11, config['model']['encoder_dim'])
            assert np.isfinite(features['features']).all()
            np.testing.assert_array_equal(features['index'], np.arange(11))
    diagnostics = json.loads((output / 'loss_diagnostics.json').read_text())
    assert diagnostics['partial']['measurements']['train_clean']['samples'] == 22
    assert diagnostics['partial']['measurements']['val_clean']['samples'] == 11
    assert (baseline / 'finetune/best.pt').read_bytes() == before
    expected = evaluate_stage('finetune', config, archive, manifest, baseline, device)
    assert expected['samples'] == 11
