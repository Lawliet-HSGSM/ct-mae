import json

import numpy as np
import pytest
import torch

from ct_mae.adaptation_data import TwinViewDataset
from ct_mae.adaptation_evaluate import calibration, freeze_selection
from ct_mae.adaptation_study import RECIPE, build_model, matched_recipe, run_variant
from ct_mae.config import load_config, write_json
from ct_mae.data import OrganDataset
from ct_mae.engine import evaluate, learning_rate
from ct_mae.models import OrganClassifier
from ct_mae.pipeline import make_loader, sha256_file


def fixture(tmp_path):
    sources = tmp_path / 'sources'
    sources.mkdir()
    config = load_config()
    config['model'].update(image_size=16, patch_size=4, encoder_dim=16,
                           encoder_depth=4, encoder_heads=4)
    config['wandb']['mode'] = 'disabled'
    torch.manual_seed(42)
    original = OrganClassifier(config['model'], 11)
    for scope in ('full', 'partial'):
        torch.save({'model': original.state_dict(), 'config': config, 'epoch': 4,
                    'subset_sha256': 'fixture'}, sources / f'{scope}.pt')
    write_json(sources / 'source_manifest.json', {
        'checkpoints': {scope: sha256_file(sources / f'{scope}.pt')
                        for scope in ('full', 'partial')}, 'subset_sha256': 'fixture',
    })
    write_json(sources / 'baseline.json', config)
    return sources, original


def test_sources_preserve_head_and_partial_frozen_tensors(tmp_path):
    sources, original = fixture(tmp_path)
    adapted, _ = build_model(sources, 'partial', 'E')
    original.eval()
    adapted.eval()
    inputs = torch.randn(2, 1, 16, 16)
    torch.testing.assert_close(original(inputs), adapted(inputs), rtol=0, atol=0)
    assert not adapted.encoder.blocks[0].attention.qkv.weight.requires_grad
    assert adapted.encoder.blocks[-1].attention.qkv.weight.requires_grad
    assert adapted.projector[0].weight.requires_grad


def test_full_adaptation_resume_preserves_next_update_including_projector(tmp_path):
    sources, _ = fixture(tmp_path)
    images = np.random.default_rng(42).integers(0, 256, (32, 16, 16), dtype=np.uint8)
    labels = np.arange(32) % 3
    dataset = TwinViewDataset(images, labels, np.arange(32))
    validation = make_loader(OrganDataset(images, labels), 16, 0, seed=42, shuffle=False)
    model, _ = build_model(sources, 'full', 'A')
    initial = evaluate(model, validation, device=torch.device('cpu'), pretrain=False,
                       seed=42, amp=False)
    recipe = dict(epochs=2, warmup_epochs=0, batch_size=16, accumulation=2, lr=1e-4, min_lr=1e-6)

    def run(path, stop=None):
        return run_variant(sources, path, 'full', 'E', dataset, labels, validation, validation,
                           initial, np.linspace(.5, 1.5, 11), recipe,
                           device=torch.device('cpu'), workers=0, stop_after=stop)

    continuous, interrupted = tmp_path / 'continuous', tmp_path / 'interrupted'
    run(continuous)
    run(interrupted, 1)
    run(interrupted)
    a = torch.load(continuous / 'full/E/last.pt', weights_only=True)
    b = torch.load(interrupted / 'full/E/last.pt', weights_only=True)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    assert json.loads((interrupted / 'full/E/run.json').read_text())['completed_epochs'] == 2


def test_calibration_handles_confidence_one_and_empty_bins():
    result = calibration([0, 1], [[1, 0], [0, 1]])
    assert result == {'nll': 0., 'ece_15': 0.}


def test_nested_ten_percent_matches_optimizer_budget_and_lr_contrastive_schedule():
    old, new = matched_recipe(RECIPE, 1728), matched_recipe(RECIPE, 3456)
    assert 50 * 1728 // 64 == new['epochs'] * 3456 // 64 == 1350
    for update in range(1350):
        old_progress, new_progress = update / 27, update / 54
        assert np.isclose(learning_rate(old_progress, old, 1e-6),
                          learning_rate(new_progress, new, 1e-6), rtol=0, atol=1e-18)
        assert np.isclose(min(old_progress / 5, 1),
                          min(new_progress / new['lambda_warmup_epochs'], 1))
    assert RECIPE['epochs'] == 50


@pytest.mark.parametrize('epochs', [25, 50])
def test_selection_preserves_epoch_zero_and_earliest_variant_ties(tmp_path, epochs):
    protocol = {'recipe': {'epochs': epochs}, 'initial_validation': {s: {'macro_f1': score}
                                      for s, score in [('full', .9), ('partial', .95)]}}
    scores = dict(zip('ABCDE', [.91, .91, .9, .909, .88], strict=True))
    for scope in ('full', 'partial'):
        for variant, score in scores.items():
            write_json(tmp_path / scope / variant / 'run.json', {
                'status': 'completed', 'completed_epochs': epochs,
                'best_score': score, 'best_epoch': 7,
            })
    selection = freeze_selection(tmp_path, protocol)
    assert selection == {'full': {'variant': 'A', 'epoch': 7, 'val_macro_f1': .91},
                         'partial': {'variant': 'source', 'epoch': 0, 'val_macro_f1': .95}}
    write_json(tmp_path / 'full/B/run.json', {
        'status': 'completed', 'completed_epochs': epochs, 'best_score': .92, 'best_epoch': 8,
    })
    with pytest.raises(ValueError, match='after it was frozen'):
        freeze_selection(tmp_path, protocol)
