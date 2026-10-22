import copy

import pytest
import torch
from test_models import small_config
from test_pipeline import experiment_fixture

from ct_mae.config import config_digest
from ct_mae.models import OrganClassifier
from ct_mae.pipeline import build_model, run_stage, sha256_file


@pytest.mark.parametrize('trainable_count', [2, 4])
def test_partial_updates_only_last_blocks_final_norm_and_head(trainable_count):
    config = small_config()
    config.update(encoder_depth=4, classifier_drop_path=0.5)
    model = OrganClassifier(config, num_classes=11, trainable_blocks=trainable_count)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model.train()
    assert not model.encoder.patch_embed.training
    first = 4 - trainable_count
    assert all(not block.training for block in model.encoder.blocks[:first])
    assert all(block.training for block in model.encoder.blocks[first:])
    assert model.encoder.norm.training and model.head.training
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01,
    )
    logits = model(torch.rand(5, 1, 16, 16))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3, 4])).backward()
    optimizer.step()
    allowed = tuple(f'encoder.blocks.{index}.' for index in range(first, 4)) + (
        'encoder.norm.', 'head.',
    )
    changed = []
    for key, value in model.state_dict().items():
        if key.startswith(allowed):
            if not torch.equal(before[key], value):
                changed.append(key)
        else:
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert any(key.startswith('encoder.blocks.2.') for key in changed)
    assert any(key.startswith('encoder.blocks.3.') for key in changed)
    assert any(key.startswith('encoder.norm.') for key in changed)
    assert any(key.startswith('head.') for key in changed)
    model.eval()
    assert all(not module.training for module in model.modules())


@pytest.mark.parametrize('count', [-1, 5])
def test_partial_rejects_invalid_block_budget(count):
    with pytest.raises(ValueError, match='blocks'):
        OrganClassifier(small_config(), num_classes=11, trainable_blocks=count)


def test_partial_transfers_original_ssl_without_mutating_baseline_protocol(tmp_path):
    config, archive, manifest = experiment_fixture(tmp_path)
    for recipe in config['stages'].values():
        recipe.update(epochs=1)
    baseline = tmp_path / 'baseline'
    run_stage('pretrain', config, archive, manifest, baseline, torch.device('cpu'))
    expected = sha256_file(baseline / 'pretrain/best.pt')
    followup = copy.deepcopy(config)
    followup['stages']['partial'] = copy.deepcopy(config['stages']['finetune'])
    followup['followup'] = {'baseline_directory': str(baseline), 'partial_blocks': 1,
                          'pretrain_sha256': expected,
                          'baseline_config_digest': config_digest(config)}
    model, actual = build_model('partial', followup, tmp_path / 'followup', torch.device('cpu'))
    reference = torch.load(baseline / 'pretrain/best.pt', weights_only=True)['model']
    assert actual == expected
    for key, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value, reference[f'encoder.{key}'], rtol=0, atol=0)
    damaged = copy.deepcopy(followup)
    damaged['augmentation']['degrees'] = 15
    with pytest.raises(ValueError, match='protocol'):
        build_model('partial', damaged, tmp_path / 'followup', torch.device('cpu'))
    damaged = copy.deepcopy(followup)
    damaged['followup']['pretrain_sha256'] = 'invalid'
    with pytest.raises(ValueError, match='checksum'):
        build_model('partial', damaged, tmp_path / 'followup', torch.device('cpu'))
