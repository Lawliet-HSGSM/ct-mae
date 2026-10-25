"""Compare original/augmented train, validation and test without changing training decisions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ct_mae.adaptation_data import TwinViewDataset
from ct_mae.collect import LABELS_VI, png_data_url
from ct_mae.config import write_json
from ct_mae.download import file_md5, official_asset
from ct_mae.error_analysis import classification_summary, read_predictions
from ct_mae.label_study import index_hash, nested_indices

METRICS = {'mean': 'Độ sáng trung bình', 'std': 'Độ phân tán mức xám',
           'dark': 'Tỷ lệ pixel ≤ 0,05', 'edge': 'Chênh lệch pixel lân cận'}


def image_metrics(images):
    """Simple pixel descriptors, not anatomical masks or verified quality scores."""
    result = {key: [] for key in METRICS}
    for start in range(0, len(images), 128):
        raw = images[start:start + 128]
        x = raw.astype(np.float32) / (255 if raw.dtype == np.uint8 else 1)
        if x.ndim != 3 or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError('Expected finite grayscale images in [0,1] or uint8')
        result['mean'].extend(x.mean((1, 2)).tolist())
        result['std'].extend(x.std((1, 2)).tolist())
        result['dark'].extend((x <= .05).mean((1, 2)).tolist())
        edge = (np.abs(np.diff(x, axis=1)).mean((1, 2))
                + np.abs(np.diff(x, axis=2)).mean((1, 2))) / 2
        result['edge'].extend(edge.tolist())
    return {key: np.array(values) for key, values in result.items()}


def accuracy_decomposition(val_labels, val_prediction, test_labels, test_prediction):
    val = classification_summary(val_labels, val_prediction)
    test = classification_summary(test_labels, test_prediction)
    pv = np.bincount(val_labels, minlength=11) / len(val_labels)
    pt = np.bincount(test_labels, minlength=11) / len(test_labels)
    rv = np.array([row['recall'] for row in val['per_class']])
    rt = np.array([row['recall'] for row in test['per_class']])
    return {'validation': val, 'test': test,
            'test_mix_with_validation_recall': float(pt @ rv),
            'class_mix_effect': float((pt - pv) @ rv),
            'within_class_effect': float(pt @ (rt - rv))}


def pick(indices, labels, organ, *, seed, count=16, eligible=None):
    candidates = np.asarray(indices)
    mask = labels[candidates] == organ
    if eligible is not None:
        mask &= eligible[candidates]
    available = candidates[mask]
    return np.random.default_rng(seed).permutation(available)[:count], len(available)


def conditional_errors(train, values, labels, predictions):
    result = {}
    for organ in range(11):
        result[str(organ)] = {}
        for metric in METRICS:
            edges = np.quantile(train[metric][train['label'] == organ], [.25, .5, .75])
            rows = []
            for bucket in range(4):
                row = {'bucket': bucket, 'boundaries': edges.tolist()}
                for split in ('val', 'test'):
                    assigned = np.searchsorted(edges, values[split][metric], side='right')
                    chosen = (labels[split] == organ) & (assigned == bucket)
                    row[split] = {'samples': int(chosen.sum()), 'models': {}}
                    for model in ('full', 'partial'):
                        pred = predictions[model][split]['prediction']
                        row[split]['models'][model] = float((pred[chosen] != organ).mean()) if (
                            chosen.any()) else None
                rows.append(row)
            result[str(organ)][metric] = rows
    return result


def build_review(archive, subset_path, artifacts, output):
    if file_md5(archive) != official_asset(128)['md5']:
        raise ValueError('Dataset checksum differs from official 128px archive')
    manifest = json.loads(subset_path.read_text('utf-8'))
    selected = np.array(manifest['indices'])
    if index_hash(selected) != manifest['indices_sha256'] or len(selected) != 1728:
        raise ValueError('Expected the original 5% supervised training subset')
    with np.load(archive, allow_pickle=False) as saved:
        images = {s: saved[f'{s}_images'] for s in ('train', 'val', 'test')}
        labels = {s: saved[f'{s}_labels'].reshape(-1) for s in images}
    added = np.setdiff1d(nested_indices(labels['train'], selected, 3456, 42), selected)
    protocol = json.loads((artifacts / 'protocol.json').read_text('utf-8'))
    predictions, gaps = {}, {}
    for model in ('full', 'partial'):
        checkpoint = json.loads((artifacts / model / 'source/test.json').read_text('utf-8'))
        if checkpoint['checkpoint_sha256'] != protocol['source_manifest']['checkpoints'][model]:
            raise ValueError('Test predictions do not use the original classifier source')
        predictions[model] = {
            'val': read_predictions(artifacts / model / 'initial_val_predictions.npz',
                                    labels['val']),
            'test': read_predictions(artifacts / model / 'source/test_predictions.npz',
                                     labels['test']),
        }
        gaps[model] = accuracy_decomposition(labels['val'], predictions[model]['val']['prediction'],
                                             labels['test'],
                                             predictions[model]['test']['prediction'])
    descriptors = {s: image_metrics(x) for s, x in images.items()}
    dataset = TwinViewDataset(images['train'], labels['train'], selected)
    augmented = np.stack([dataset[i]['views'][0, 0].numpy() for i in range(len(dataset))])
    descriptors['train_aug'] = image_metrics(augmented)
    descriptors['train5'] = {k: v[selected] for k, v in descriptors['train'].items()}
    stat_labels = {**labels, 'train5': labels['train'][selected],
                   'train_aug': labels['train'][selected]}
    summaries = {}
    for split, metrics in descriptors.items():
        summaries[split] = {str(organ): {
            metric: {'q': np.quantile(x[stat_labels[split] == organ],
                                      [.1, .25, .5, .75, .9]).tolist(),
                     'mean': float(x[stat_labels[split] == organ].mean())}
            for metric, x in metrics.items()} for organ in range(11)}
    del augmented
    review = {'labels': LABELS_VI, 'metrics': METRICS, 'seed': 42,
              'archive_md5': official_asset(128)['md5'], 'source': protocol['source_manifest'],
              'split_counts': {s: len(y) for s, y in labels.items()}, 'summaries': summaries,
              'class_counts': {s: np.bincount(y, minlength=11).tolist()
                               for s, y in stat_labels.items()}, 'gaps': gaps,
              'cases': {}, 'banks': {}, 'pool_counts': {}}
    pools = {'train5': selected, 'train_all': np.arange(len(labels['train'])), 'added5': added}
    augment_dataset = TwinViewDataset(images['train'], labels['train'],
                                      np.arange(len(labels['train'])))

    def record(split, index):
        key = f'{split}:{index}'
        if key in review['cases']:
            return key
        case = {'index': int(index), 'split': split, 'label': int(labels[split][index]),
                'image': png_data_url(images[split][index]),
                'metrics': {k: float(v[index]) for k, v in descriptors[split].items()}}
        if split == 'train':
            views = augment_dataset[int(index)]['views'][:, 0].numpy()
            case['augmented'] = [png_data_url(np.rint(v * 255).astype(np.uint8)) for v in views]
            case['labeled5'] = bool(np.isin(index, selected))
        else:
            case['models'] = {model: {
                'prediction': int(predictions[model][split]['prediction'][index]),
                'confidence': float(predictions[model][split]['probability'][index].max()),
            } for model in predictions}
        review['cases'][key] = case
        return key

    for name, indices in pools.items():
        review['banks'][name], review['pool_counts'][name] = {}, {}
        for organ in range(11):
            picked, count = pick(indices, labels['train'], organ, seed=42 + organ)
            review['banks'][name][str(organ)] = [record('train', i) for i in picked]
            review['pool_counts'][name][str(organ)] = count
    for split in ('val', 'test'):
        review['banks'][split], review['pool_counts'][split] = {}, {}
        for model in predictions:
            review['banks'][split][model], review['pool_counts'][split][model] = {}, {}
            correct = predictions[model][split]['prediction'] == labels[split]
            for organ in range(11):
                bank, sizes = {}, {}
                for mode, eligible in [('random', None), ('correct', correct), ('wrong', ~correct)]:
                    chosen, count = pick(np.arange(len(correct)), labels[split], organ,
                                         seed=42 + organ, eligible=eligible)
                    bank[mode], sizes[mode] = [record(split, i) for i in chosen], count
                review['banks'][split][model][str(organ)] = bank
                review['pool_counts'][split][model][str(organ)] = sizes
    train = {**descriptors['train5'], 'label': stat_labels['train5']}
    review['conditional_errors'] = conditional_errors(train, descriptors, labels, predictions)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'pixel_descriptors.npz', **{
        f'{s}_{key}': value for s, x in descriptors.items() for key, value in x.items()})
    write_json(output / 'split_review.json', review)
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--subset', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    review = json.loads((args.output / 'split_review.json').read_text('utf-8')) if args.reuse else (
        build_review(args.archive, args.subset, args.artifacts, args.output))
    from ct_mae.split_review_plots import write_review
    write_review(review, args.output)


if __name__ == '__main__':
    main()
