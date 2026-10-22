"""Compare label budgets while reusing the original, verified MAE pretraining."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ct_mae.config import config_digest, load_config, write_json
from ct_mae.data import select_labeled_indices
from ct_mae.download import fetch_asset, file_md5, official_asset
from ct_mae.followup import audit_frozen
from ct_mae.kaggle import verify_runs
from ct_mae.models import OrganClassifier
from ct_mae.pipeline import evaluate_stage, run_stage, sha256_file

STAGES = ('scratch', 'linear', 'finetune', 'partial')


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def nested_indices(labels, parent, count, seed):
    labels, parent = np.asarray(labels).reshape(-1), np.asarray(parent)
    if (parent.ndim != 1 or not np.issubdtype(parent.dtype, np.integer)
            or len(np.unique(parent)) != len(parent) or (parent < 0).any()
            or (parent >= len(labels)).any() or not len(parent) <= count <= len(labels)):
        raise ValueError('Invalid parent indices or label budget')
    # Reuse the canonical largest-remainder allocation; sample only the added rows.
    allocation = np.bincount(labels[select_labeled_indices(labels, count, seed)],
                             minlength=int(labels.max()) + 1)
    existing = np.bincount(labels[parent], minlength=len(allocation))
    if (existing > allocation).any():
        raise ValueError('Parent cannot fit the proportional class quotas at this budget')
    rng = np.random.default_rng(seed)
    result = [parent]
    available = np.ones(len(labels), dtype=bool)
    available[parent] = False
    for label, extra in enumerate(allocation - existing):
        result.append(rng.choice(np.flatnonzero(available & (labels == label)),
                                 size=int(extra), replace=False))
    return np.sort(np.concatenate(result)).astype(np.int64)


def study_config(baseline: Path, *, count=3456, blocks=3):
    config = load_config(baseline / 'resolved_config.json')
    source_digest = config_digest(config)
    config['dataset']['labeled_count'] = count
    config['stages']['partial'] = copy.deepcopy(config['stages']['finetune'])
    config['wandb']['run_name_suffix'] = f'labels{count}'
    config['label_study'] = {
        'baseline_directory': str(baseline.resolve()), 'source_config_digest': source_digest,
        'source_checkpoint_sha256': sha256_file(baseline / 'pretrain/best.pt'),
        'partial_blocks': blocks, 'sampling': 'stratified extension containing old subset',
        'budget_policy': '100 epochs for both budgets; more labels imply more updates',
    }
    return config


def validate_protocol(config):
    study = config['label_study']
    baseline = Path(study['baseline_directory'])
    original = load_config(baseline / 'resolved_config.json')
    record = json.loads((baseline / 'pretrain/run.json').read_text('utf-8'))
    if (config_digest(original) != study['source_config_digest']
            or record['config_digest'] != config_digest(original)
            or record['status'] != 'completed'
            or record['completed_epochs'] != original['stages']['pretrain']['epochs']):
        raise ValueError('Original SSL protocol is incomplete or inconsistent')
    expected = copy.deepcopy(original)
    expected['dataset']['labeled_count'] = config['dataset']['labeled_count']
    expected['stages']['partial'] = copy.deepcopy(expected['stages']['finetune'])
    expected['wandb']['run_name_suffix'] = f'labels{config["dataset"]["labeled_count"]}'
    expected['label_study'] = study
    if study.get('diagnostic'):
        for stage in STAGES:
            expected['stages'][stage].update(epochs=2, warmup_epochs=0)
    if config_digest(config) != config_digest(expected):
        raise ValueError('Label study changed another part of the baseline protocol')
    source = baseline / 'pretrain/best.pt'
    if sha256_file(source) != study['source_checkpoint_sha256']:
        raise ValueError('SSL source checkpoint checksum changed')
    return source


def build_study_model(stage, config, output, device):
    if stage not in STAGES:
        raise ValueError('Label-budget study only trains downstream classifiers')
    source = validate_protocol(config)
    model = OrganClassifier(
        config['model'], config['dataset']['num_classes'], frozen=stage == 'linear',
        seed=config['seed'], trainable_blocks=config['label_study']['partial_blocks']
        if stage == 'partial' else None,
    )
    source_hash = None
    if stage != 'scratch':
        checkpoint = torch.load(source, map_location='cpu', weights_only=True)
        if checkpoint['config']['model'] != config['model']:
            raise ValueError('SSL architecture differs from the study encoder')
        model.encoder.load_state_dict({key.removeprefix('encoder.'): value
                                       for key, value in checkpoint['model'].items()
                                       if key.startswith('encoder.')})
        source_hash = sha256_file(source)
    return model.to(device), source_hash


def prepare_study_manifest(archive, baseline, config):
    parent = json.loads((baseline / 'subset.json').read_text('utf-8'))
    if index_hash(parent['indices']) != parent['indices_sha256']:
        raise ValueError('Original labeled subset checksum is inconsistent')
    if 'md5' in parent and file_md5(archive) != parent['md5']:
        raise ValueError('Dataset checksum differs from the original data')
    with np.load(archive, allow_pickle=False) as saved:
        labels = saved['train_labels'].reshape(-1)
    indices = nested_indices(labels, np.array(parent['indices']),
                              config['dataset']['labeled_count'], config['seed'])
    manifest = copy.deepcopy(parent)
    manifest.update(indices=indices.tolist(), indices_sha256=index_hash(indices),
                    labeled_count=len(indices), train_label_fraction=len(indices) / len(labels),
                    labeled_class_counts=np.bincount(labels[indices], minlength=11).tolist(),
                    parent_labeled_count=parent['labeled_count'],
                    parent_indices_sha256=parent['indices_sha256'],
                    contains_original_subset=True, sampling=config['label_study']['sampling'])
    return manifest


def run_label_study(baseline, archive, output, device, *, count=3456, blocks=3, smoke=False):
    config = study_config(baseline, count=count, blocks=blocks)
    if smoke:
        config['label_study']['diagnostic'] = True
        for stage in STAGES:
            config['stages'][stage].update(epochs=2, warmup_epochs=0)
    manifest = prepare_study_manifest(archive, baseline, config)
    output.mkdir(parents=True, exist_ok=True)
    previous = output / 'subset.json'
    if previous.exists() and json.loads(previous.read_text('utf-8')) != manifest:
        raise ValueError('Study output already contains another subset')
    write_json(output / 'subset.json', manifest)
    write_json(output / 'resolved_config.json', config)
    results = {}
    for stage in STAGES:
        write_json(output / 'progress.json', {'status': 'running', 'stage': stage,
                                              'labeled_count': count})
        record = run_stage(stage, config, archive, manifest, output, device,
                           resume=(output / stage / 'last.pt').exists(), smoke=smoke,
                           train_limit=128 if smoke else None, val_limit=128 if smoke else None,
                           model_factory=build_study_model)
        results[stage] = record
        if not smoke:
            evaluate_stage(stage, config, archive, manifest, output, device,
                           model_factory=build_study_model)
    source = baseline / 'pretrain/best.pt'
    initial = torch.load(source, map_location='cpu', weights_only=True)['model']
    linear = torch.load(output / 'linear/best.pt', map_location='cpu', weights_only=True)['model']
    for key, value in linear.items():
        if key.startswith('encoder.') and not torch.equal(value, initial[key]):
            raise ValueError('Linear probing changed a frozen encoder tensor')
    write_json(output / 'freeze_audits.json', {
        'linear': {'passed': True, 'source_sha256': sha256_file(source)},
        'partial': audit_frozen(output / 'partial/best.pt', source, blocks),
    })
    write_json(output / 'progress.json', {'status': 'completed', 'labeled_count': count})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--count', type=int, default=3456)
    parser.add_argument('--blocks', type=int, default=3)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    config = load_config(args.baseline / 'resolved_config.json')
    archive = fetch_asset(official_asset(config['dataset']['size']), args.data_dir)
    run_label_study(args.baseline, archive, args.output, torch.device(args.device),
                    count=args.count, blocks=args.blocks, smoke=args.smoke)
    if config['wandb']['mode'] == 'online':
        import wandb
        api = wandb.Api()
        verify_runs(args.output, api, config['wandb']['entity'] or api.default_entity,
                    config['wandb']['project'], list(STAGES), smoke=args.smoke)


if __name__ == '__main__':
    main()
