import json

import numpy as np
import torch
from test_pipeline import experiment_fixture

from ct_mae.config import write_json
from ct_mae.label_report import collect_comparison, write_comparison
from ct_mae.label_study import index_hash, run_label_study
from ct_mae.pipeline import run_stage


def test_report_matches_saved_predictions_and_nested_budgets(tmp_path):
    config, archive, manifest = experiment_fixture(tmp_path)
    for recipe in config['stages'].values():
        recipe.update(epochs=1)
    baseline = tmp_path / 'pretraining'
    manifest['filename'] = archive.name
    manifest['indices_sha256'] = index_hash(manifest['indices'])
    with np.load(archive) as saved:
        manifest['labeled_class_counts'] = np.bincount(
            saved['train_labels'].reshape(-1)[manifest['indices']], minlength=11,
        ).tolist()
    write_json(baseline / 'resolved_config.json', config)
    write_json(baseline / 'subset.json', manifest)
    device = torch.device('cpu')
    run_stage('pretrain', config, archive, manifest, baseline, device)
    smaller, larger = tmp_path / 'smaller', tmp_path / 'larger'
    run_label_study(baseline, archive, smaller, device, count=22, blocks=1)
    run_label_study(baseline, archive, larger, device, count=33, blocks=1)
    report = tmp_path / 'report'
    comparison = collect_comparison(smaller, smaller, larger, archive, report)
    write_comparison(comparison, report)
    assert comparison['budgets']['5']['count'] == 22
    assert comparison['budgets']['10']['count'] == 33
    for stage in ('scratch', 'linear', 'finetune', 'partial'):
        result = json.loads((larger / stage / 'test.json').read_text())
        assert comparison['budgets']['10']['methods'][stage]['accuracy'] == result['accuracy']
        assert sum(comparison['paired'][stage].values()) == 11
    assert (report / 'label_efficiency_report.html').is_file()
    assert (report / 'label_efficiency_report.md').is_file()
    assert (report / 'figures/label_comparison.png').is_file()
