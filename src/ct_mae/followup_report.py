"""Build a self-contained Vietnamese follow-up report from measured model outputs."""

import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ct_mae.collect import LABELS_VI, png_data_url
from ct_mae.config import write_json
from ct_mae.download import METADATA, file_md5
from ct_mae.error_analysis import (
    balanced_pool,
    classification_summary,
    fit_projection,
    neighbor_agreement,
    paired_outcomes,
    read_predictions,
    select_errors,
)

STAGES = ('scratch', 'linear', 'finetune', 'partial')
FEATURE_STAGES = ('random', 'mae', 'scratch', 'finetune', 'partial')
NAMES = {'scratch': 'Supervised từ đầu', 'linear': 'SSL · linear probing',
         'finetune': 'SSL · full fine-tuning', 'partial': 'SSL · partial (3 block cuối)',
         'random': 'Encoder ngẫu nhiên', 'mae': 'Encoder sau MAE, chưa học nhãn'}
SHORT = {'scratch': 'Scratch', 'linear': 'Linear', 'finetune': 'Full', 'partial': 'Partial',
         'random': 'Random', 'mae': 'MAE'}
COLORS = ['#1976d2', '#ec407a', '#8e24aa', '#ff8f00', '#00897b', '#7cb342',
          '#795548', '#ef5350', '#5c6bc0', '#00acc1', '#a57900']


def figure_save(fig, output, name):
    path = output / 'figures' / f'{name}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return path


def distribution_plot(counts, output):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), layout='constrained')
    x = np.arange(11)
    for j, (split, values) in enumerate(counts.items()):
        values = np.array(values)
        label = f'{split} (n={values.sum():,})'
        axes[0].bar(x + (j - 1.5) * .2, values, width=.2, label=label)
        axes[1].bar(x + (j - 1.5) * .2, values / values.sum() * 100, width=.2, label=label)
    for axis in axes:
        axis.set_xticks(x, METADATA['labels'], rotation=60, ha='right')
        axis.grid(axis='y', alpha=.2)
    axes[0].set(title='Số ảnh mỗi cơ quan', ylabel='Ảnh')
    axes[1].set(title='Tỷ trọng trong từng tập — không đồng đều', ylabel='% ảnh')
    axes[0].legend(fontsize=8)
    figure_save(fig, output, 'distributions')


def metric_plots(summary, runs, output):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    x = np.arange(11)
    for j, stage in enumerate(STAGES):
        axes[0].bar(x + (j - 1.5) * .2,
                    [row['f1'] * 100 for row in summary[stage]['per_class']],
                    width=.2, label=SHORT[stage])
        history = runs[stage]['history']
        epochs = [row['epoch'] for row in history]
        axes[1].plot(epochs, [row['val']['macro_f1'] * 100 for row in history],
                     label=SHORT[stage])
        best = runs[stage]['best_epoch']
        axes[1].scatter(best, history[best - 1]['val']['macro_f1'] * 100, s=25)
    axes[0].set_xticks(x, METADATA['labels'], rotation=60, ha='right')
    axes[0].set(ylabel='Test F1 (%)', ylim=(0, 102), title='F1 từng cơ quan')
    axes[1].set(xlabel='Epoch', ylabel='Validation macro-F1 (%)',
                title='Chọn checkpoint bằng validation; chấm = epoch được chọn')
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(axis='y', alpha=.2)
    figure_save(fig, output, 'metrics')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    for stage, axis in zip(STAGES, axes.flat, strict=True):
        history = runs[stage]['history']
        for split in ('train', 'val'):
            axis.plot([row['epoch'] for row in history],
                      [row[split]['loss'] for row in history], label=split)
        axis.set(title=NAMES[stage], xlabel='Epoch', ylabel='Cross-entropy')
        axis.grid(alpha=.2)
        axis.legend()
    figure_save(fig, output, 'loss_curves')


def confusion_plot(summary, output):
    fig, axes = plt.subplots(2, 2, figsize=(17, 15), layout='constrained')
    for stage, axis in zip(STAGES, axes.flat, strict=True):
        matrix = np.array(summary[stage]['confusion'])
        rates = matrix / np.maximum(matrix.sum(1, keepdims=True), 1) * 100
        axis.imshow(rates, cmap='Blues', vmin=0, vmax=100)
        for row in range(11):
            for col in range(11):
                if matrix[row, col]:
                    axis.text(col, row, f'{matrix[row, col]}\n{rates[row, col]:.1f}%',
                              ha='center', va='center', fontsize=7,
                              color='white' if rates[row, col] > 55 else '#17324d')
        axis.set_xticks(range(11), METADATA['labels'], rotation=65, ha='right', fontsize=8)
        axis.set_yticks(range(11), METADATA['labels'], fontsize=8)
        axis.set(title=NAMES[stage], xlabel='Model dự đoán →', ylabel='Nhãn thật →')
    fig.suptitle('Confusion matrix: số ảnh và % theo hàng (nhãn thật)', fontsize=17)
    figure_save(fig, output, 'confusion_matrices')


def worst_case_groups(labels, predictions):
    full = predictions['finetune']
    partial = predictions['partial']
    scratch = predictions['scratch']
    groups = {'full_worst': select_errors(labels, full['probability'], 12).tolist(),
              'partial_worst': select_errors(labels, partial['probability'], 12).tolist(),
              'regression': select_errors(labels, full['probability'], 12,
                                          scratch['prediction'] == labels).tolist(),
              'partial_recovers': select_errors(labels, full['probability'], 12,
                                                partial['prediction'] == labels).tolist(),
              'partial_regresses': select_errors(labels, partial['probability'], 12,
                                                 full['prediction'] == labels).tolist(),
              'heart_to_lung': select_errors(labels, full['probability'], 12,
                                             (labels == 3) & (full['prediction'] == 7)).tolist(),
              'spleen_to_lung': select_errors(labels, full['probability'], 12,
                                              (labels == 10) & (full['prediction'] == 7)).tolist(),
              'kidney_to_spleen': select_errors(labels, full['probability'], 12,
                                               (labels == 4)
                                               & (full['prediction'] == 10)).tolist()}
    rng = np.random.default_rng(42)
    errors = []
    controls = []
    for label in range(11):
        wrong = np.flatnonzero((labels == label) & (full['prediction'] != labels))
        errors.extend(rng.choice(wrong, min(3, len(wrong)), replace=False).tolist())
        correct = np.flatnonzero((labels == label) & (full['prediction'] == labels))
        controls.extend(rng.choice(correct, min(2, len(correct)), replace=False).tolist())
    groups['random_errors'] = errors
    groups['correct_controls'] = controls
    groups['focus_controls'] = [index for index in controls if labels[index] in (0, 3, 4, 7, 10)]
    comparison_rng = np.random.default_rng(42)
    for source, target, name in ((3, 7, 'heart_comparison'), (4, 10, 'kidney_comparison')):
        indices = select_errors(labels, full['probability'], 4,
                                (labels == source) & (full['prediction'] == target)).tolist()
        for label in (source, target):
            candidates = np.flatnonzero((labels == label) & (full['prediction'] == label))
            indices.extend(comparison_rng.choice(candidates, min(4, len(candidates)),
                                                  replace=False).tolist())
        groups[name] = indices
    return groups


def image_galleries(images, labels, predictions, groups, output):
    for name, indices in groups.items():
        indices = indices[:12]
        if not indices:
            continue
        columns = 4
        rows = (len(indices) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(14, 4.2 * rows),
                                 squeeze=False, layout='constrained')
        for axis in axes.flat:
            axis.set_axis_off()
        for axis, index in zip(axes.flat, indices, strict=False):
            axis.imshow(images[index], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            text = [f'Test #{index} · Thật: {METADATA["labels"][labels[index]]}']
            for stage in ('scratch', 'finetune', 'partial'):
                values = predictions[stage]
                predicted = values['prediction'][index]
                confidence = values['probability'][index, predicted]
                text.append(f'{SHORT[stage]}: {METADATA["labels"][predicted]} ({confidence:.1%})')
            axis.set_title('\n'.join(text), fontsize=10)
        fig.suptitle(f'{name} · ảnh gốc 128×128, grayscale cố định 0–255', fontsize=14)
        figure_save(fig, output, f'cases_{name}')


def project_all(followup, labels, output):
    (output / 'projections').mkdir(parents=True, exist_ok=True)
    results, coordinates = {}, {}
    pool = balanced_pool(labels)
    np.save(output / 'projections/neighbor_pool_test_indices.npy', pool)
    for stage in FEATURE_STAGES:
        with np.load(followup / f'embeddings/{stage}_train.npz') as saved:
            if not np.array_equal(saved['index'], np.arange(len(saved['features']))):
                raise ValueError('Train feature indices are not in official order')
            train = saved['features']
        with np.load(followup / f'embeddings/{stage}_test.npz') as saved:
            if not np.array_equal(saved['index'], np.arange(len(labels))):
                raise ValueError('Test feature indices are not in official order')
            test = saved['features']
        results[stage] = {'neighbors': neighbor_agreement(test, labels, pool)}
        coordinates[stage] = {}
        for method in ('pca', 'svd'):
            fit = fit_projection(train, test, centered=method == 'pca')
            coordinates[stage][method] = fit['test_coordinates'].round(5).tolist()
            np.savez_compressed(output / f'projections/{stage}_{method}.npz', **fit)
            results[stage][method] = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in fit.items()
                if key not in ('mean', 'components', 'train_coordinates', 'test_coordinates')
            }
        print(f'PROJECTION COMPLETE {stage}', flush=True)
        del train, test
    for method in ('pca', 'svd'):
        fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout='constrained')
        for stage, axis in zip(FEATURE_STAGES, axes.flat, strict=False):
            xy = np.array(coordinates[stage][method])
            for label in range(11):
                selected = labels == label
                axis.scatter(xy[selected, 0], xy[selected, 1], s=2, alpha=.3,
                             color=COLORS[label], rasterized=True,
                             label=METADATA['labels'][label])
            ratio = results[stage][method]['centered_variance_ratio']
            axis.set(title=f'{SHORT[stage]} · 2D giữ {sum(ratio):.1%} phương sai đã trừ mean',
                     xlabel='PC1' if method == 'pca' else 'SVD1 (không trừ mean)',
                     ylabel='PC2' if method == 'pca' else 'SVD2')
            axis.grid(alpha=.15)
        axes.flat[-1].set_axis_off()
        handles, names = axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].legend(handles, names, loc='center', markerscale=4,
                             title='Nhãn chỉ dùng để tô màu test')
        fig.suptitle(f'{method.upper()} · fit trên train không nhãn → chiếu toàn bộ test\n'
                     'Mỗi encoder có hệ trục riêng; không so trực tiếp tọa độ giữa các ô',
                     fontsize=15)
        figure_save(fig, output, f'embedding_{method}')
    return results, coordinates


def collect_analysis(baseline: Path, followup: Path, archive: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    manifest = json.loads((baseline / 'subset.json').read_text('utf-8'))
    if json.loads((followup / 'subset.json').read_text('utf-8')) != manifest:
        raise ValueError('Labeled subsets disagree')
    if 'md5' in manifest and file_md5(archive) != manifest['md5']:
        raise ValueError('Analysis dataset checksum differs from the training data')
    with np.load(archive, allow_pickle=False) as data:
        labels = data['test_labels'].reshape(-1)
        images = data['test_images']
        counts = {split: np.bincount(data[f'{split}_labels'].reshape(-1), minlength=11).tolist()
                  for split in ('train', 'val', 'test')}
    counts['labeled_train'] = manifest['labeled_class_counts']
    predictions, summaries, runs = {}, {}, {}
    for stage in STAGES:
        directory = (followup if stage == 'partial' else baseline) / stage
        values = read_predictions(directory / 'test_predictions.npz', labels)
        summary = classification_summary(labels, values['prediction'])
        recorded = json.loads((directory / 'test.json').read_text('utf-8'))
        if not np.isclose(summary['accuracy'], recorded['accuracy'], atol=1e-10):
            raise ValueError(f'Recorded accuracy does not match saved predictions: {stage}')
        if not np.isclose(summary['macro_f1'], recorded['macro_f1'], atol=1e-10):
            raise ValueError(f'Recorded F1 does not match saved predictions: {stage}')
        predictions[stage], summaries[stage] = values, summary
        runs[stage] = json.loads((directory / 'run.json').read_text('utf-8'))
    distribution_plot(counts, output)
    metric_plots(summaries, runs, output)
    confusion_plot(summaries, output)
    groups = worst_case_groups(labels, predictions)
    image_galleries(images, labels, predictions, groups, output)
    projections, coordinates = project_all(followup, labels, output)
    rows = []
    for index in sorted(set(sum(groups.values(), []))):
        rows.append({'index': index, 'label': int(labels[index]),
                     'image': png_data_url(images[index]),
                     'groups': [name for name, indices in groups.items() if index in indices],
                     'models': {
                         stage: {'prediction': int(values['prediction'][index]),
                                 'confidence': float(values['probability'][index].max()),
                                 'true_probability': float(
                                     values['probability'][index, labels[index]])}
                         for stage, values in predictions.items()}})
    paired = {f'{a}_to_{b}': paired_outcomes(labels, predictions[a]['prediction'],
                                            predictions[b]['prediction'])
              for a, b in (('scratch', 'finetune'), ('finetune', 'partial'))}
    result = {
        'labels': METADATA['labels'], 'labels_vi': LABELS_VI, 'names': NAMES, 'colors': COLORS,
        'test_labels': labels.tolist(), 'test_predictions': {
            key: value['prediction'].tolist() for key, value in predictions.items()},
        'metrics': summaries, 'paired': paired, 'counts': counts, 'groups': groups,
        'cases': rows, 'projections': projections, 'coordinates': coordinates,
        'protocol': json.loads((followup / 'resolved_config.json').read_text('utf-8')),
        'freeze_audit': json.loads((followup / 'freeze_audit.json').read_text('utf-8')),
        'loss_diagnostics': json.loads((followup / 'loss_diagnostics.json').read_text('utf-8')),
        'embedding_provenance': json.loads(
            (followup / 'embedding_provenance.json').read_text('utf-8')),
        'runs': {stage: {
            **{key: value for key, value in run.items() if key != 'history'},
            'peak_vram_bytes': max(row['resource']['peak_vram_bytes'] for row in run['history']),
            'train_loop_seconds': sum(row['train']['seconds'] for row in run['history']),
        } for stage, run in runs.items()},
    }
    write_json(output / 'analysis.json', result)
    return result


def table(headers, rows):
    def cell(value):
        return html.escape(str(value))
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(
        f'<th>{cell(value)}</th>' for value in headers
    ) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{cell(value)}</td>' for value in row) + '</tr>' for row in rows
    ) + '</tbody></table></div>'


def embedded_figure(output, name, caption):
    path = output / 'figures' / f'{name}.png'
    if not path.exists():
        return '<p>Không có ảnh thỏa điều kiện chọn.</p>'
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}">' \
           f'<figcaption>{html.escape(caption)}</figcaption></figure>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--followup', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    data = (json.loads((args.output / 'analysis.json').read_text('utf-8')) if args.reuse else
            collect_analysis(args.baseline, args.followup, args.archive, args.output))
    from ct_mae.followup_writeup import write_reports
    write_reports(data, args.output)


if __name__ == '__main__':
    main()
