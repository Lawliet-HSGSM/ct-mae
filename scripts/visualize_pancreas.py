"""Show true-pancreas test crops alongside an existing misclassified reference."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from ct_mae.collect import LABELS_VI
from ct_mae.download import file_md5, official_asset
from ct_mae.error_analysis import read_predictions

STAGES = {'scratch': 'Scratch', 'linear': 'Linear', 'finetune': 'Full', 'partial': 'Partial'}


def draw_sheet(images, labels, predictions, indices, reference, title, target):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12.6), layout='constrained')
    for axis, index in zip(axes.flat, [reference, *indices], strict=True):
        axis.imshow(images[index], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        axis.set_axis_off()
        predicted = int(predictions['finetune']['prediction'][index])
        confidence = predictions['finetune']['probability'][index, predicted]
        prefix = 'THAM CHIẾU · ' if index == reference else ''
        axis.set_title(f'{prefix}#{index} · Nhãn: {LABELS_VI[labels[index]]}\n'
                       f'Full: {LABELS_VI[predicted]} · {confidence:.1%}', fontsize=9,
                       color='#b91c1c' if predicted != labels[index] else '#134e4a')
    fig.suptitle(title + '\nPixel gốc 128×128 · thang xám cố định 0–255 · '
                 'dự đoán từ lượt 5% nhãn', fontsize=13)
    fig.savefig(target, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=int, default=10267)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / 'reports/followup'
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    archive = root / 'data/organamnist_128.npz'
    digest = file_md5(archive)
    if digest != official_asset(128)['md5']:
        raise ValueError('Official archive checksum mismatch')
    with np.load(archive, allow_pickle=False) as saved:
        images, labels = saved['test_images'], saved['test_labels'].reshape(-1)
    reference = args.reference
    if not 0 <= reference < len(labels):
        raise ValueError('Reference must be an official test index')
    if images.dtype != np.uint8 or images.shape != (len(labels), 128, 128):
        raise ValueError('Expected original grayscale uint8 test images')
    predictions = {}
    for stage in STAGES:
        base = root / ('reports/followup/artifacts' if stage == 'partial'
                       else '.worktrees/dataset/reports/kaggle-v7')
        predictions[stage] = read_predictions(base / stage / 'test_predictions.npz', labels)
    pool = np.flatnonzero(labels == 9)
    pool = pool[pool != reference]
    assert LABELS_VI[9] == 'Tụy'
    features = images.reshape(-1, 32, 4, 32, 4).mean(axis=(2, 4), dtype=np.float32) / 255
    distance = np.square(features[pool] - features[reference]).mean(axis=(1, 2))
    nearest = pool[np.argsort(distance, kind='stable')]
    random = np.random.default_rng(42).choice(pool, 15, replace=False)
    selection = {
        'reference': reference, 'reference_label': int(labels[reference]),
        'organ_label': 9, 'test_organ_count': len(pool), 'random_seed': 42,
        'random_indices': random.tolist(), 'nearest_indices': nearest.tolist(),
        'nearest_mse': np.sort(distance, kind='stable').tolist(),
        'matching': 'MSE of area-averaged 32x32 grayscale pixels /255; '
                    'no anatomy verification, no prediction-outcome filtering',
        'dataset_md5': digest, 'budget': 'Original 5%-label checkpoints, seed 42',
    }
    for name, indices, title in (
        ('random', random, 'Ảnh #10267 và 15 ảnh có nhãn tụy · chọn ngẫu nhiên, seed 42'),
        ('nearest', nearest[:15], 'Ảnh #10267 và 15 ảnh có nhãn tụy gần nhất theo bố cục pixel'),
    ):
        draw_sheet(images, labels, predictions, indices, reference, title,
                   figures / f'pancreas_{name}_{reference}.png')
    (output / 'pancreas_review.json').write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print(json.dumps({'reference': reference, 'pancreas_test_images': len(pool),
                      'random': random.tolist(), 'nearest': nearest[:15].tolist()},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
