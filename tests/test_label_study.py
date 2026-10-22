import hashlib
import json

import numpy as np
import pytest
import torch
from test_pipeline import experiment_fixture

from ct_mae.config import write_json
from ct_mae.data import select_labeled_indices
from ct_mae.label_study import (
    build_study_model,
    nested_indices,
    run_label_study,
    study_config,
)
from ct_mae.pipeline import run_stage, sha256_file


def test_larger_budget_preserves_parent_and_stratified_proportions():
    labels = np.array([0] * 30 + [1] * 15 + [2] * 5)
    parent = select_labeled_indices(labels, 10, 42)
    chosen = nested_indices(labels, parent, 20, 42)
    assert len(chosen) == len(np.unique(chosen)) == 20
    assert set(parent) <= set(chosen)
    np.testing.assert_array_equal(np.bincount(labels[chosen]), [12, 6, 2])
    np.testing.assert_array_equal(chosen, nested_indices(labels, parent, 20, 42))
    np.testing.assert_array_equal(parent, nested_indices(labels, parent, 10, 42))


@pytest.mark.parametrize('parent,count', [([0, 0], 20), ([-1], 20), ([0], 51),
                                         (list(range(15)), 20)])
def test_nested_sampling_rejects_invalid_or_incompatible_parent(parent, count):
    with pytest.raises(ValueError):
        nested_indices(np.array([0] * 30 + [1] * 15 + [2] * 5), np.array(parent), count, 42)


def test_new_budget_reuses_verified_ssl_for_all_transfer_branches(tmp_path):
    config, archive, manifest = experiment_fixture(tmp_path)
    for recipe in config['stages'].values():
        recipe.update(epochs=1)
    baseline = tmp_path / 'baseline'
    manifest['filename'] = archive.name
    manifest['indices_sha256'] = hashlib.sha256(
        np.array(manifest['indices'], dtype='<i8').tobytes(),
    ).hexdigest()
    with np.load(archive) as data:
        manifest['labeled_class_counts'] = np.bincount(
            data['train_labels'].reshape(-1)[manifest['indices']], minlength=11,
        ).tolist()
    write_json(baseline / 'resolved_config.json', config)
    write_json(baseline / 'subset.json', manifest)
    device = torch.device('cpu')
    run_stage('pretrain', config, archive, manifest, baseline, device)
    original_bytes = (baseline / 'pretrain/best.pt').read_bytes()
    output = tmp_path / 'label10'
    results = run_label_study(baseline, archive, output, device, count=33, blocks=1)
    assert set(results) == {'scratch', 'linear', 'finetune', 'partial'}
    selected = json.loads((output / 'subset.json').read_text())
    assert len(selected['indices']) == 33
    assert set(manifest['indices']) <= set(selected['indices'])
    for stage in results:
        record = json.loads((output / stage / 'run.json').read_text())
        assert record['train_samples'] == 33
        assert record['completed_epochs'] == 1
        assert record['subset_sha256'] == selected['indices_sha256']
        assert (output / stage / 'test_predictions.npz').is_file()
        if stage != 'scratch':
            assert record['ssl_checkpoint_sha256'] == sha256_file(baseline / 'pretrain/best.pt')
    assert (baseline / 'pretrain/best.pt').read_bytes() == original_bytes
    assert json.loads((output / 'freeze_audits.json').read_text())['partial']['passed']
    changed = study_config(baseline, count=33, blocks=1)
    changed['augmentation']['degrees'] = 15
    with pytest.raises(ValueError, match='protocol'):
        build_study_model('finetune', changed, output, device)
