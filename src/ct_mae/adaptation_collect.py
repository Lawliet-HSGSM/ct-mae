"""Download adaptation outputs and independently audit labels, checkpoints and sampling."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from ct_mae.adaptation_data import class_weights, paired_batches
from ct_mae.adaptation_evaluate import calibration
from ct_mae.config import write_json
from ct_mae.download import file_md5, official_asset
from ct_mae.error_analysis import (
    balanced_pool,
    classification_summary,
    neighbor_agreement,
    read_predictions,
)
from ct_mae.followup_collect import fetch_file, selected_path
from ct_mae.pipeline import sha256_file

ROOT = 'ct-mae-adaptation'


def audit_predictions(directory, labels, checkpoint):
    values = read_predictions(directory / 'test_predictions.npz', labels)
    summary = classification_summary(labels, values['prediction'])
    test = json.loads((directory / 'test.json').read_text('utf-8'))
    if sha256_file(checkpoint) != test['checkpoint_sha256']:
        raise ValueError('Test evaluated a different checkpoint')
    metrics = {**summary, **calibration(labels, values['probability'])}
    for key in ('accuracy', 'macro_f1', 'nll', 'ece_15'):
        if not np.isclose(metrics[key], test[key], atol=1e-12, rtol=0):
            raise ValueError(f'Saved {key} differs from predictions')
    return summary


def verify_artifacts(output, sources, archive):
    if file_md5(archive) != official_asset(128)['md5']:
        raise ValueError('Report archive is not the original verified dataset')
    if json.loads((output / 'job_status.json').read_text('utf-8'))['status'] != 'completed':
        raise ValueError('Adaptation job is incomplete')
    protocol = json.loads((output / 'protocol.json').read_text('utf-8'))
    expected = json.loads((sources / 'source_manifest.json').read_text('utf-8'))
    if protocol['source_manifest'] != expected or protocol['smoke']:
        raise ValueError('Wrong checkpoint sources or diagnostic outputs')
    indices = np.array(protocol['actual_train_indices'], dtype=np.int64)
    count = protocol.get('labeled_count', 1728)
    epochs = protocol['recipe']['epochs']
    if (count not in (1728, 3456) or len(indices) != count
            or len(np.unique(indices)) != count
            or hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest() != (
                protocol['subset_sha256'])):
        raise ValueError('Training label subset changed')
    parent = json.loads((sources / 'subset.json').read_text('utf-8'))
    if not np.isin(parent['indices'], indices).all():
        raise ValueError('Adaptation omitted original supervised labels')
    if count == 3456:
        if epochs != 25 or protocol['recipe']['warmup_epochs'] != 2.5 or (
                protocol['lambda_warmup_epochs'] != 2.5):
            raise ValueError('10% adaptation did not match the original update schedule')
    with np.load(archive, allow_pickle=False) as saved:
        train_labels = saved['train_labels'].reshape(-1)[indices]
        val_labels = saved['val_labels'].reshape(-1)
        test_labels = saved['test_labels'].reshape(-1)
    proof = json.loads((output / 'wandb_verification.json').read_text('utf-8'))
    if (len(proof) != 10 or any(row['status'] != 'completed' for row in proof)
            or {(r['scope'], r['variant']) for r in proof} != {
                (s, v) for s in ('full', 'partial') for v in 'ABCDE'}):
        raise ValueError('W&B has not verified all ten runs')
    frozen_selection = json.loads((output / 'selection_frozen.json').read_text('utf-8'))
    audits = {}
    for scope in ('full', 'partial'):
        if sha256_file(sources / f'{scope}.pt') != expected['checkpoints'][scope]:
            raise ValueError('Local source checkpoint changed')
        original = torch.load(sources / f'{scope}.pt',
                              map_location='cpu', weights_only=True)['model']
        audits[f'{scope}/source'] = audit_predictions(output / scope / 'source', test_labels,
                                                      sources / f'{scope}.pt')
        best = {'variant': 'source', 'epoch': 0,
                'val_macro_f1': protocol['initial_validation'][scope]['macro_f1']}
        weights = class_weights(protocol['initial_validation'][scope]['per_class_f1'],
                                np.bincount(train_labels, minlength=11))
        np.testing.assert_allclose(weights, protocol['class_weights'][scope], atol=1e-12)
        for variant in 'ABCDE':
            directory = output / scope / variant
            run = json.loads((directory / 'run.json').read_text('utf-8'))
            if (run['status'] != 'completed' or run['completed_epochs'] != epochs
                    or len(run['history']) != epochs or run['config']['seed'] != 42):
                raise ValueError('Wrong adaptation budget')
            scores = [run['initial_val']['macro_f1']]
            scores += [h['val']['macro_f1'] for h in run['history']]
            if int(np.argmax(scores)) != run['best_epoch']:
                raise ValueError('Checkpoint is not earliest best including epoch zero')
            if run['best_score'] != max(scores):
                raise ValueError('Best validation score disagrees with history')
            if run['best_score'] > best['val_macro_f1']:
                best = {'variant': variant, 'epoch': run['best_epoch'],
                        'val_macro_f1': run['best_score']}
            with np.load(directory / 'sampling_trace.npz', allow_pickle=False) as trace:
                for epoch, row in enumerate(run['history']):
                    actual = trace[f'epoch_{epoch + 1}']
                    np.testing.assert_array_equal(np.sort(actual), np.sort(indices))
                    if variant == 'A':
                        plan = np.random.default_rng(42 + epoch).permutation(len(indices))
                    else:
                        batches, _ = paired_batches(train_labels, protocol['recipe']['batch_size'],
                                                     seed=42 + epoch)
                        plan = np.array(batches).reshape(-1)
                    np.testing.assert_array_equal(actual, indices[plan])
                    if hashlib.sha256(actual.astype('<i8').tobytes()).hexdigest() != (
                            row['sampling']['indices_sha256']):
                        raise ValueError('Sampling trace checksum differs from history')
                    if (row['train']['samples'] != count or row['train']['views'] != 2 * count
                            or row['val']['samples'] != 6491):
                        raise ValueError('Unexpected image/view/evaluation count')
                    if row['train']['optimizer_steps'] + row['train']['skipped_steps'] != (
                            count // 64):
                        raise ValueError('Unexpected optimizer update budget')
            selected = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
            last = torch.load(directory / 'last.pt', map_location='cpu', weights_only=True)
            frozen = [k for k in original if k.startswith('encoder.') and scope == 'partial'
                      and not k.startswith(('encoder.blocks.9.', 'encoder.blocks.10.',
                                            'encoder.blocks.11.', 'encoder.norm.'))]
            for key in frozen:
                if not torch.equal(original[key], selected['model'][key]) or not torch.equal(
                        original[key], last['model'][key]):
                    raise ValueError(f'Frozen source tensor changed: {key}')
            if selected['epoch'] != run['best_epoch']:
                raise ValueError('Selected checkpoint has a different epoch')
            summary = audit_predictions(directory, test_labels, directory / 'best.pt')
            audits[f'{scope}/{variant}'] = {'best_epoch': run['best_epoch'],
                                            'frozen_tensors': len(frozen), **summary}
        if best != frozen_selection[scope]:
            raise ValueError('Variant choice disagrees with validation-only selection')
        pool = balanced_pool(val_labels, per_class=100, seed=42)
        for variant in dict.fromkeys(('source', best['variant'])):
            directory = output / scope / variant
            with np.load(directory / 'validation_features.npz', allow_pickle=False) as features:
                np.testing.assert_array_equal(features['index'], pool)
                if features['features'].shape != (1100, 384):
                    raise ValueError('Neighbor diagnostic used different encoder features')
                measured = neighbor_agreement(features['features'], val_labels[pool],
                                               np.arange(len(pool)))
            reported = json.loads((directory / 'neighbors.json').read_text('utf-8'))
            np.testing.assert_allclose(measured['agreement'], reported['agreement'], rtol=0,
                                       atol=1e-12)
    return {'status': 'verified', 'sources': expected, 'audits': audits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--root-name', default=ROOT)
    args = parser.parse_args()
    if not args.verify_only:
        if args.manifest is None:
            parser.error('--manifest is required when downloading')
        files = json.loads(args.manifest.read_text('utf-8'))['files']
        files = [row for row in files
                 if selected_path(row['file_name'], root_name=args.root_name) is not None]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(fetch_file, row, args.output, root_name=args.root_name)
                       for row in files]
            for future in as_completed(futures):
                print('COLLECTED', future.result(), flush=True)
    write_json(args.output / 'collection_verification.json',
               verify_artifacts(args.output, args.sources, args.archive))


if __name__ == '__main__':
    main()
