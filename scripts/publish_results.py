"""Export audited scalar results, then redraw the public figures without private artifacts."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ct_mae.config import write_json
from ct_mae.download import METADATA, file_md5
from ct_mae.error_analysis import classification_summary, paired_outcomes, read_predictions

METHODS = ('scratch', 'linear', 'finetune', 'partial')
NAMES = ('Supervised scratch', 'SSL linear probe', 'SSL full fine-tuning', 'SSL last 3 blocks')
COLORS = ('#58677b', '#008779', '#dd7824', '#8056ad')


def read_json(path):
    return json.loads(path.read_text('utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_run(directory, labels, *, study, method, count):
    record, recorded = read_json(directory / 'run.json'), read_json(directory / 'test.json')
    if record['status'] != 'completed':
        raise ValueError(f'Incomplete run: {directory}')
    predictions = read_predictions(directory / 'test_predictions.npz', labels)
    measured = classification_summary(labels, predictions['prediction'])
    for key in ('accuracy', 'macro_f1'):
        if not np.isclose(measured[key], recorded[key], rtol=0, atol=1e-12):
            raise ValueError(f'Prediction/metric mismatch: {directory}/{key}')
    if measured['confusion'] != recorded['confusion_matrix']:
        raise ValueError(f'Confusion mismatch: {directory}')
    candidates = [(r['epoch'], r['val']['macro_f1']) for r in record['history']]
    if 'initial_val' in record:
        candidates.insert(0, (0, record['initial_val']['macro_f1']))
    selected, score = max(candidates, key=lambda pair: pair[1])
    if selected != record['best_epoch'] or not np.isclose(score, record['best_score']):
        raise ValueError(f'Checkpoint selection mismatch: {directory}')
    history = [{
        'epoch': r['epoch'],
        'train': {k: v for k, v in r['train'].items()
                  if isinstance(v, (float, int))},
        'val': {k: v for k, v in r['val'].items()
                if isinstance(v, (float, int))},
        'peak_vram_bytes': r['resource']['peak_vram_bytes'],
    } for r in record['history']]
    return {
        'study': study, 'method': method, 'labeled_count': count,
        'completed_epochs': record['completed_epochs'], 'selected_epoch': selected,
        'val_macro_f1': score, 'test': measured,
        'successful_updates': sum(r['train']['optimizer_steps'] for r in history),
        'amp_skipped_updates': sum(r['train'].get('skipped_steps', 0) for r in history),
        'parameters': record['parameters'], 'recipe': record['recipe'],
        'environment': record['environment'], 'history': history,
        'provenance': {
            'run_json_sha256': digest(directory / 'run.json'),
            'test_json_sha256': digest(directory / 'test.json'),
            'predictions_sha256': digest(directory / 'test_predictions.npz'),
            'checkpoint_sha256': recorded['checkpoint_sha256'],
        },
    }, predictions


def export_snapshot(args):
    if file_md5(args.archive) != METADATA['assets']['128']['md5']:
        raise ValueError('Official OrganAMNIST128 archive checksum mismatch')
    with np.load(args.archive, allow_pickle=False) as archive:
        labels = archive['test_labels'].reshape(-1)
        counts = {s: np.bincount(archive[f'{s}_labels'].reshape(-1), minlength=11).tolist()
                  for s in ('train', 'val', 'test')}
    args.output.mkdir(parents=True, exist_ok=True)
    data = {'schema_version': 1, 'seed': 42, 'labels': METADATA['labels'],
            'dataset_md5': METADATA['assets']['128']['md5'], 'class_counts': counts,
            'runs': [], 'adaptation_selection': {}, 'paired': {}}
    for count, path in ((1728, args.baseline), (3456, args.labels10)):
        subset = read_json(path / 'subset.json')
        write_json(args.output / f'subset_{count}.json', subset)
        for method in METHODS:
            directory = args.partial / 'partial' if count == 1728 and method == 'partial' else (
                path / method)
            run, _ = export_run(directory, labels, study='initial', method=method, count=count)
            data['runs'].append(run)
    previous = {}
    for count, root in ((1728, args.adapt5), (3456, args.adapt10)):
        data['adaptation_selection'][str(count)] = read_json(root / 'selection_frozen.json')
        for scope in ('full', 'partial'):
            for variant in 'ABCDE':
                run, predictions = export_run(root / scope / variant, labels,
                    study='continuation', method=f'{scope}_{variant}', count=count)
                data['runs'].append(run)
                if scope == 'full' and variant == 'A':
                    previous[count] = predictions['prediction']
    data['paired']['full_A_5_to_10'] = paired_outcomes(labels, previous[1728], previous[3456])
    pretrain = read_json(args.baseline / 'pretrain/run.json')
    data['pretrain'] = {
        'completed_epochs': pretrain['completed_epochs'],
        'selected_epoch': pretrain['best_epoch'], 'recipe': pretrain['recipe'],
        'environment': pretrain['environment'],
        'history': [{'epoch': r['epoch'], 'train_loss': r['train']['loss'],
                     'val_loss': r['val']['loss']} for r in pretrain['history']],
        'run_json_sha256': digest(args.baseline / 'pretrain/run.json'),
    }
    data['model'] = read_json(args.baseline / 'model_summary.json')
    data['config'] = read_json(args.baseline / 'resolved_config.json')
    # Auth settings are operational details; numerical settings remain unmodified.
    data['config'].pop('wandb')
    write_json(args.output / 'results.json', data)
    args.figures.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.baseline / 'pretrain/reconstruction.png',
                    args.figures / 'reconstruction.png')
    print(f'Exported {len(data["runs"])} completed classification runs with recomputed metrics')


def draw(data, figures, output):
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    initial = {(r['labeled_count'], r['method']): r for r in data['runs']
               if r['study'] == 'initial'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for method, name, color in zip(METHODS, NAMES, COLORS, strict=True):
        for metric, axis in zip(('accuracy', 'macro_f1'), axes, strict=True):
            axis.plot([5, 10], [initial[n, method]['test'][metric] * 100
                              for n in (1728, 3456)], 'o-', color=color, label=name)
            axis.set(xticks=[5, 10], xlabel='Labeled training images (%)',
                     ylabel=f'Test {metric.replace("_", " ")} (%)')
            axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Two measured label budgets | 100 downstream epochs | seed 42\n'
                 '10% uses twice the planned updates; SSL adds pretraining compute', fontsize=11)
    fig.savefig(figures / 'label_budgets.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    history = data['pretrain']['history']
    for key, label in (('train_loss', 'Train (augmented)'), ('val_loss', 'Validation')):
        axes[0].plot([r['epoch'] for r in history], [r[key] for r in history], label=label)
    axes[0].set(title='MAE: masked-pixel MSE', xlabel='Pretraining epoch', ylabel='MSE')
    axes[0].legend(fontsize=8)
    for method, name, color in zip(METHODS, NAMES, COLORS, strict=True):
        history = initial[1728, method]['history']
        epochs = [r['epoch'] for r in history]
        axes[1].plot(epochs, [r['val']['macro_f1'] * 100 for r in history], color=color, label=name)
        axes[2].plot(epochs, [r['train']['loss'] for r in history], color=color, label=name)
        axes[2].plot(epochs, [r['val']['loss'] for r in history], color=color, linestyle='--')
    axes[1].set(title='5% labels: checkpoint selection', xlabel='Downstream epoch',
                ylabel='Validation macro-F1 (%)')
    axes[1].legend(fontsize=7)
    axes[2].set(title='5% labels: solid train / dashed validation', xlabel='Downstream epoch',
                ylabel='Cross-entropy', yscale='log')
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(figures / 'learning_curves.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
    for count, axis in zip((1728, 3456), axes, strict=True):
        matrix = np.asarray(initial[count, 'finetune']['test']['confusion'])
        normalized = matrix / matrix.sum(1, keepdims=True)
        axis.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
        for i, j in np.ndindex(matrix.shape):
            if normalized[i, j] >= .03:
                axis.text(j, i, f'{normalized[i, j]:.0%}', ha='center', va='center', fontsize=7,
                          color='white' if normalized[i, j] > .6 else '#17324b')
        axis.set(xticks=range(11), yticks=range(11), xlabel='Predicted class',
                 ylabel='True class', title=f'SSL full fine-tuning | {count:,} labels')
        axis.set_xticklabels(data['labels'], rotation=65, ha='right', fontsize=8)
        axis.set_yticklabels(data['labels'], fontsize=8)
    fig.suptitle('Row-normalized test confusion | labels mark cells with at least 3%')
    fig.savefig(figures / 'confusion.png', dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 4), layout='constrained')
    positions = np.arange(11)
    for offset, (split, counts) in zip((-.25, 0, .25), data['class_counts'].items(), strict=True):
        proportions = np.asarray(counts) / sum(counts) * 100
        axis.bar(positions + offset, proportions, width=.25, label=split)
    axis.set(xticks=positions, ylabel='Share of each official split (%)',
             title='Official class distribution | class proportions are not equal')
    axis.set_xticklabels(data['labels'], rotation=55, ha='right')
    axis.legend()
    axis.grid(axis='y', alpha=.2)
    fig.savefig(figures / 'class_distribution.png', dpi=160)
    plt.close(fig)

    fields = ('study', 'method', 'labeled_count', 'completed_epochs', 'selected_epoch',
              'successful_updates', 'val_macro_f1', 'test_accuracy', 'test_macro_f1')
    with (output / 'metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in data['runs']:
            row = {key: run[key] for key in fields if key in run}
            row.update(test_accuracy=run['test']['accuracy'], test_macro_f1=run['test']['macro_f1'])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/results'))
    parser.add_argument('--figures', type=Path, default=Path('docs/assets'))
    for name in ('baseline', 'partial', 'labels10', 'adapt5', 'adapt10', 'archive'):
        parser.add_argument(f'--{name}', type=Path)
    args = parser.parse_args()
    artifacts = [getattr(args, name) for name in
                 ('baseline', 'partial', 'labels10', 'adapt5', 'adapt10', 'archive')]
    if any(artifacts):
        if not all(artifacts):
            parser.error('An export needs all six artifact/archive arguments; omit all to redraw')
        export_snapshot(args)
    draw(read_json(args.output / 'results.json'), args.figures, args.output)


if __name__ == '__main__':
    main()
