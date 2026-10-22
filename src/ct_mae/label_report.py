"""Measured 5%-versus-10% comparison using aligned, selected-checkpoint predictions."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ct_mae.collect import LABELS_VI
from ct_mae.config import write_json
from ct_mae.download import METADATA, file_md5
from ct_mae.error_analysis import (
    classification_summary,
    paired_outcomes,
    read_predictions,
    select_errors,
)
from ct_mae.followup_report import NAMES, SHORT, STAGES, figure_save

METHOD_COLORS = {'scratch': '#526782', 'linear': '#008774',
                 'finetune': '#e57b26', 'partial': '#7953b8'}


def comparison_figures(data, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for stage in STAGES:
        for metric, axis in zip(('accuracy', 'macro_f1'), axes, strict=True):
            axis.plot([5, 10], [data['budgets'][key]['methods'][stage][metric] * 100
                                for key in ('5', '10')], marker='o',
                       color=METHOD_COLORS[stage], label=SHORT[stage])
            axis.set(xticks=[5, 10], xlabel='% nhãn train', ylabel=f'Test {metric} (%)')
            axis.grid(alpha=.2)
            axis.legend()
    fig.suptitle('Hai ngân sách nhãn · 100 epoch / nhánh · seed 42\n'
                 'Đường nối chỉ biểu diễn hai điểm đã đo, chưa có error bar', fontsize=14)
    figure_save(fig, output, 'label_comparison')
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout='constrained')
    for stage, axis in zip(STAGES, axes.flat, strict=True):
        for budget, style in (('5', '--'), ('10', '-')):
            metrics = data['budgets'][budget]['methods'][stage]
            axis.plot(range(11), [row['f1'] * 100 for row in metrics['per_class']],
                       linestyle=style, marker='o', label=f'{budget}% nhãn',
                       color='#8d9aa7' if budget == '5' else METHOD_COLORS[stage])
        axis.set_xticks(range(11), METADATA['labels'], rotation=60, ha='right', fontsize=8)
        axis.set(title=NAMES[stage], ylabel='Test F1 (%)', ylim=(0, 102))
        axis.grid(alpha=.2)
        axis.legend()
    figure_save(fig, output, 'per_class_comparison')
    differences = np.array([[data['budgets']['10']['methods'][s]['per_class'][i]['f1']
                              - data['budgets']['5']['methods'][s]['per_class'][i]['f1']
                              for i in range(11)] for s in STAGES]) * 100
    limit = max(1, float(np.abs(differences).max()))
    fig, axis = plt.subplots(figsize=(14, 4), layout='constrained')
    axis.imshow(differences, cmap='RdYlGn', vmin=-limit, vmax=limit, aspect='auto')
    for row in range(4):
        for col in range(11):
            axis.text(col, row, f'{differences[row, col]:+.2f}', ha='center', va='center')
    axis.set_xticks(range(11), METADATA['labels'], rotation=50, ha='right')
    axis.set_yticks(range(4), [SHORT[s] for s in STAGES])
    axis.set_title('Thay đổi F1: 10% − 5%, đơn vị điểm phần trăm')
    figure_save(fig, output, 'per_class_delta')
    for metric, filename in (('val_macro_f1', 'validation_epochs'),
                             ('train_loss', 'training_loss'), ('val_loss', 'validation_loss')):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout='constrained')
        for stage, axis in zip(STAGES, axes.flat, strict=True):
            for budget, style in (('5', '--'), ('10', '-')):
                curve = data['budgets'][budget]['methods'][stage]['curve']
                factor = 100 if metric == 'val_macro_f1' else 1
                axis.plot([row['epoch'] for row in curve],
                           [row[metric] * factor for row in curve], linestyle=style,
                           label=f'{budget}% nhãn')
            axis.set(title=NAMES[stage], xlabel='Epoch', ylabel=metric)
            axis.grid(alpha=.2)
            axis.legend()
        figure_save(fig, output, filename)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout='constrained')
    for stage, axis in zip(STAGES, axes.flat, strict=True):
        for budget, style in (('5', '--'), ('10', '-')):
            curve = data['budgets'][budget]['methods'][stage]['curve']
            axis.plot([row['updates'] for row in curve],
                       [row['val_macro_f1'] * 100 for row in curve], linestyle=style,
                       label=f'{budget}% nhãn')
        axis.set(title=NAMES[stage], xlabel='Số optimizer updates đã thực hiện',
                 ylabel='Validation macro-F1 (%)')
        axis.grid(alpha=.2)
        axis.legend()
    fig.suptitle('LR giữ nguyên theo epoch; hai đường không có cùng lịch LR theo step')
    figure_save(fig, output, 'validation_updates')


def confusion_figures(data, output):
    for budget in ('5', '10'):
        for stage in STAGES:
            matrix = np.array(data['budgets'][budget]['methods'][stage]['confusion'])
            rates = matrix / np.maximum(matrix.sum(1, keepdims=True), 1) * 100
            fig, axis = plt.subplots(figsize=(10, 9), layout='constrained')
            axis.imshow(rates, cmap='Blues', vmin=0, vmax=100)
            for row in range(11):
                for col in range(11):
                    if matrix[row, col]:
                        axis.text(col, row, f'{matrix[row, col]}\n{rates[row, col]:.1f}%',
                                  ha='center', va='center', fontsize=7,
                                  color='white' if rates[row, col] > 55 else '#18334c')
            axis.set_xticks(range(11), METADATA['labels'], rotation=60, ha='right', fontsize=8)
            axis.set_yticks(range(11), METADATA['labels'], fontsize=8)
            axis.set(title=f'{NAMES[stage]} · {budget}% nhãn',
                     xlabel='Dự đoán →', ylabel='Nhãn thật →')
            figure_save(fig, output, f'confusion_{stage}_{budget}')


def paired_galleries(data, images, labels, predictions, output):
    cases = {}
    for stage in STAGES:
        smaller, larger = predictions['5'][stage], predictions['10'][stage]
        for name, wrong, other in (('recovered', smaller, larger), ('regressed', larger, smaller)):
            indices = select_errors(labels, wrong['probability'], 6, other['prediction'] == labels)
            cases[f'{stage}_{name}'] = indices.tolist()
            if not len(indices):
                continue
            fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
            for axis in axes.flat:
                axis.set_axis_off()
            for axis, index in zip(axes.flat, indices, strict=False):
                axis.imshow(images[index], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                title = f'Test #{index} · Thật: {METADATA["labels"][labels[index]]}'
                for budget in ('5', '10'):
                    values = predictions[budget][stage]
                    predicted = values['prediction'][index]
                    title += (f'\n{budget}%: {METADATA["labels"][predicted]} '
                              f'({values["probability"][index, predicted]:.1%})')
                axis.set_title(title, fontsize=10)
            fig.suptitle(f'{SHORT[stage]} · {name} · ảnh gốc, grayscale cố định 0–255')
            figure_save(fig, output, f'cases_{stage}_{name}')
    return cases


def collect_comparison(baseline5, partial5, labels10, archive, output):
    output.mkdir(parents=True, exist_ok=True)
    parents = {key: json.loads((root / 'subset.json').read_text('utf-8'))
               for key, root in (('5', baseline5), ('10', labels10))}
    if not set(parents['5']['indices']) <= set(parents['10']['indices']):
        raise ValueError('Larger budget does not contain the original subset')
    if parents['10']['parent_indices_sha256'] != parents['5']['indices_sha256']:
        raise ValueError('Label budgets have inconsistent parent provenance')
    if 'md5' in parents['5'] and file_md5(archive) != parents['5']['md5']:
        raise ValueError('Comparison data differs from training data')
    with np.load(archive, allow_pickle=False) as saved:
        labels = saved['test_labels'].reshape(-1)
        images = saved['test_images']
        counts = {split: np.bincount(saved[f'{split}_labels'].reshape(-1), minlength=11).tolist()
                  for split in ('train', 'val', 'test')}
    budgets, predictions, sources = {}, {}, set()
    for budget in ('5', '10'):
        methods, predictions[budget] = {}, {}
        for stage in STAGES:
            root = labels10 if budget == '10' else (partial5 if stage == 'partial' else baseline5)
            run = json.loads((root / stage / 'run.json').read_text('utf-8'))
            test = json.loads((root / stage / 'test.json').read_text('utf-8'))
            epochs = run['config']['stages'][stage]['epochs']
            if (run['status'] != 'completed' or run['smoke'] or len(run['history']) != epochs
                    or run['completed_epochs'] != epochs
                    or run['train_samples'] != parents[budget]['labeled_count']
                    or run['subset_sha256'] != parents[budget]['indices_sha256']):
                raise ValueError(f'Invalid completed {budget}% {stage} run')
            best = int(np.argmax([row['val']['macro_f1'] for row in run['history']])) + 1
            if best != run['best_epoch'] or test['selected_epoch'] != best:
                raise ValueError('Checkpoint is not the earliest best validation macro-F1')
            values = read_predictions(root / stage / 'test_predictions.npz', labels)
            summary = classification_summary(labels, values['prediction'])
            for metric in ('accuracy', 'macro_f1'):
                if not np.isclose(summary[metric], test[metric], atol=1e-12, rtol=0):
                    raise ValueError(f'{metric} disagrees with saved predictions')
            curve, updates = [], 0
            for row in run['history']:
                updates += row['train']['optimizer_steps']
                curve.append({'epoch': row['epoch'], 'updates': updates,
                              'train_loss': row['train']['loss'], 'val_loss': row['val']['loss'],
                              'val_macro_f1': row['val']['macro_f1'], 'lr': row['train']['lr']})
            methods[stage] = {
                **summary, 'curve': curve, 'best_epoch': best, 'updates': updates,
                'best_val_macro_f1': run['best_score'], 'recipe': run['recipe'],
                'parameters': run['parameters'], 'environment': run['environment'],
                'seconds': run['seconds_this_session'], 'wandb_url': run['wandb_url'],
                'checkpoint_sha256': test['checkpoint_sha256'],
                'ssl_checkpoint_sha256': run['ssl_checkpoint_sha256'],
                'peak_vram_bytes': max(r['resource']['peak_vram_bytes'] for r in run['history']),
                'train_seconds': sum(r['train']['seconds'] for r in run['history']),
            }
            predictions[budget][stage] = values
            if stage != 'scratch':
                sources.add(run['ssl_checkpoint_sha256'])
        budgets[budget] = {'count': parents[budget]['labeled_count'],
                           'manifest': parents[budget], 'methods': methods}
    if len(sources) != 1 or None in sources:
        raise ValueError('Transfer branches did not use the same MAE checkpoint')
    for stage in STAGES:
        for field in ('epochs', 'lr', 'warmup_epochs', 'weight_decay', 'betas'):
            if budgets['5']['methods'][stage]['recipe'][field] != (
                budgets['10']['methods'][stage]['recipe'][field]
            ):
                raise ValueError(f'Downstream recipe changed: {stage}/{field}')
    data = {'budgets': budgets, 'counts': counts, 'labels': METADATA['labels'],
            'labels_vi': LABELS_VI, 'source_checkpoint_sha256': sources.pop(),
            'paired': {stage: paired_outcomes(labels, predictions['5'][stage]['prediction'],
                                              predictions['10'][stage]['prediction'])
                       for stage in STAGES}}
    comparison_figures(data, output)
    confusion_figures(data, output)
    data['cases'] = paired_galleries(data, images, labels, predictions, output)
    write_json(output / 'label_comparison.json', data)
    return data


def write_comparison(data, output):
    from ct_mae.label_writeup import write_report
    write_report(data, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline5', type=Path, required=True)
    parser.add_argument('--partial5', type=Path, required=True)
    parser.add_argument('--labels10', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    data = (json.loads((args.output / 'label_comparison.json').read_text('utf-8')) if args.reuse
            else collect_comparison(args.baseline5, args.partial5, args.labels10,
                                    args.archive, args.output))
    write_comparison(data, args.output)


if __name__ == '__main__':
    main()
