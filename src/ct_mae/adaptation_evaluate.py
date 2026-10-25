"""Freeze validation decisions before final test inference; diagnose encoder features."""

import json

import numpy as np
import torch

from ct_mae.adaptation_study import VARIANTS, build_model
from ct_mae.config import write_json
from ct_mae.data import OrganDataset, read_split
from ct_mae.engine import evaluate
from ct_mae.error_analysis import balanced_pool, neighbor_agreement
from ct_mae.followup import extract_features
from ct_mae.pipeline import make_loader, sha256_file


def calibration(labels, probability):
    labels, probability = np.asarray(labels).reshape(-1), np.asarray(probability)
    confidence, correct = probability.max(1), probability.argmax(1) == labels
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(float((bins == k).mean()) * abs(float(correct[bins == k].mean())
              - float(confidence[bins == k].mean())) for k in range(15) if (bins == k).any())
    nll = -np.log(np.maximum(probability[np.arange(len(labels)), labels], 1e-30)).mean()
    return {'nll': float(nll), 'ece_15': float(ece)}


def freeze_selection(output, protocol):
    selection = {}
    for scope in ('full', 'partial'):
        best = {'variant': 'source', 'epoch': 0,
                'val_macro_f1': protocol['initial_validation'][scope]['macro_f1']}
        for variant in VARIANTS:
            record = json.loads((output / scope / variant / 'run.json').read_text('utf-8'))
            if record['status'] != 'completed' or record['completed_epochs'] != (
                    protocol.get('recipe', {}).get('epochs', 50)):
                raise ValueError('All production adaptations must finish before test evaluation')
            if record['best_score'] > best['val_macro_f1']:
                best = {'variant': variant, 'epoch': record['best_epoch'],
                        'val_macro_f1': record['best_score']}
        selection[scope] = best
    path = output / 'selection_frozen.json'
    if path.exists() and json.loads(path.read_text('utf-8')) != selection:
        raise ValueError('Validation selection changed after it was frozen')
    write_json(path, selection)
    return selection


def selected_model(sources, output, scope, variant, device):
    model, _ = build_model(sources, scope, 'A' if variant == 'source' else variant, device=device)
    if variant != 'source':
        saved = torch.load(output / scope / variant / 'best.pt',
                           map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model'])
    model.eval()
    return model


def evaluate_study(sources, archive, output, device, *, workers=2):
    protocol = json.loads((output / 'protocol.json').read_text('utf-8'))
    selection = freeze_selection(output, protocol)
    images, labels = read_split(archive, 'test', supervised=True)
    loader = make_loader(OrganDataset(images, labels), 64, workers, seed=42, shuffle=False)
    for scope in ('full', 'partial'):
        for variant in ('source', *VARIANTS):
            directory = output / scope / variant
            directory.mkdir(exist_ok=True)
            checkpoint = sources / f'{scope}.pt' if variant == 'source' else directory / 'best.pt'
            checksum = sha256_file(checkpoint)
            if (directory / 'test.json').exists():
                previous = json.loads((directory / 'test.json').read_text('utf-8'))
                if previous['checkpoint_sha256'] == checksum:
                    continue
                raise ValueError('Previously evaluated checkpoint changed')
            model = selected_model(sources, output, scope, variant, device)
            metrics, predictions = evaluate(model, loader, device=device, pretrain=False,
                                            seed=42, amp=device.type == 'cuda',
                                            return_predictions=True)
            metrics.update(calibration(labels, predictions['probability']))
            metrics['checkpoint_sha256'] = checksum
            np.savez_compressed(directory / 'test_predictions.npz', **predictions)
            write_json(directory / 'test.json', metrics)
            print(f'TEST {scope}/{variant} acc={metrics["accuracy"]:.6f}', flush=True)
            del model
    del images, loader
    val_images, val_labels = read_split(archive, 'val', supervised=True)
    pool = balanced_pool(val_labels, per_class=100, seed=42)
    np.save(output / 'validation_neighbor_indices.npy', pool)
    for scope in ('full', 'partial'):
        for variant in dict.fromkeys(('source', selection[scope]['variant'])):
            directory = output / scope / variant
            model = selected_model(sources, output, scope, variant, device)
            features = extract_features(model, val_images[pool], device, batch_size=64,
                                        workers=workers, seed=42)
            metrics = neighbor_agreement(features, val_labels[pool], np.arange(len(pool)))
            np.savez_compressed(directory / 'validation_features.npz',
                                index=pool, features=features)
            write_json(directory / 'neighbors.json', metrics)
            del model
    write_json(output / 'progress.json', {'status': 'completed', 'selection': selection})
