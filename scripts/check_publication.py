"""Check committed/publication files without data, credentials or model checkpoints."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERN = re.compile(
    r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}'
    r'|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
KEY_PATTERN = re.compile(
    r'''(?i)(?:api[_-]?key|access[_-]?token)\s*["']?\s*[:=]\s*["']([A-Za-z0-9_=-]{32,})["']''')


def check_results():
    data = json.loads((ROOT / 'docs/results/results.json').read_text('utf-8'))
    runs = data['runs']
    assert len(runs) == 28
    with (ROOT / 'docs/results/metrics.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(runs)
    for run, row in zip(runs, rows, strict=True):
        for key in ('study', 'method', 'labeled_count', 'completed_epochs', 'selected_epoch'):
            assert str(run[key]) == row[key], (run['method'], key)
        matrix = np.asarray(run['test']['confusion'])
        assert matrix.shape == (11, 11) and matrix.sum() == 17778
        f1 = 2 * matrix.diagonal() / (matrix.sum(0) + matrix.sum(1))
        assert np.isclose(f1.mean(), run['test']['macro_f1'], rtol=0, atol=1e-12)
        assert np.isclose(matrix.trace() / matrix.sum(), run['test']['accuracy'],
                          rtol=0, atol=1e-12)
        assert float(row['test_accuracy']) == run['test']['accuracy']
        assert float(row['test_macro_f1']) == run['test']['macro_f1']
        assert len(run['history']) == run['completed_epochs']
        if run['study'] == 'continuation':
            assert run['successful_updates'] == 1350 and run['amp_skipped_updates'] == 0
        for value in run['provenance'].values():
            assert re.fullmatch(r'[0-9a-f]{64}', value)
    subsets = []
    for count in (1728, 3456):
        manifest = json.loads((ROOT / f'docs/results/subset_{count}.json').read_text('utf-8'))
        indices = np.asarray(manifest['indices'], dtype='<i8')
        assert len(indices) == len(np.unique(indices)) == count
        assert (indices >= 0).all() and (indices < 34561).all()
        assert hashlib.sha256(indices.tobytes()).hexdigest() == manifest['indices_sha256']
        subsets.append(set(indices.tolist()))
    assert subsets[0] < subsets[1]
    for budget, selection in data['adaptation_selection'].items():
        for scope in ('full', 'partial'):
            selected = selection[scope]
            candidates = [r for r in runs if r['study'] == 'continuation'
                          and r['labeled_count'] == int(budget)
                          and r['method'].startswith(scope + '_')]
            assert selected['val_macro_f1'] >= max(r['val_macro_f1'] for r in candidates)
            if selected['variant'] != 'source':
                chosen = next(r for r in candidates
                              if r['method'] == f'{scope}_{selected["variant"]}')
                assert chosen['selected_epoch'] == selected['epoch']
    print('Results: 28 runs, two nested subsets, confusion metrics and selection verified')


def check_files():
    output = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
    paths = sorted({ROOT / p.decode('utf-8') for p in output.split(b'\0') if p})
    forbidden = ('.pt', '.pth', '.ckpt', '.npz', '.npy', '.pem', '.key')
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        assert not relative.startswith(('data/', 'reports/', 'work/', '.worktrees/', 'wandb/'))
        assert path.suffix not in forbidden, relative
        assert path.stat().st_size < 8 * 1024**2, f'Oversized public file: {relative}'
        if path.suffix == '.png':
            continue
        text = path.read_text('utf-8')
        assert not TOKEN_PATTERN.search(text) and not KEY_PATTERN.search(text), (
            f'Credential pattern detected in {relative}; inspect locally')
        if path.suffix == '.md':
            for raw in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
                target = raw.strip('<>').split(' ', 1)[0]
                if urlsplit(target).scheme or target.startswith('#'):
                    continue
                target = unquote(target.split('#', 1)[0])
                assert (path.parent / target).is_file(), f'Broken link: {relative} -> {target}'
        if relative == 'README.md' or relative.startswith('docs/'):
            assert not re.search(r'[CD]:[\\/]', text), f'Local machine path: {relative}'
    provenance = json.loads((ROOT / 'docs/results/figure_provenance.json').read_text('utf-8'))
    for name, record in provenance.items():
        path = ROOT / 'docs/assets' / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['output_sha256'], name
    print(f'Publication files: {len(paths)} paths checked; local links and figure hashes verified')


def check_history():
    """Inspect reachable Git blobs; report only paths, never matching values."""
    records = subprocess.check_output(['git', 'rev-list', '--objects', 'HEAD'], cwd=ROOT)
    count = 0
    for record in records.decode('utf-8').splitlines():
        object_id, _, name = record.partition(' ')
        if not name or Path(name).suffix == '.png':
            continue
        kind = subprocess.check_output(['git', 'cat-file', '-t', object_id], cwd=ROOT).strip()
        if kind != b'blob':
            continue
        blob = subprocess.check_output(['git', 'cat-file', '-p', object_id], cwd=ROOT)
        text = blob.decode('utf-8', errors='replace')
        assert not TOKEN_PATTERN.search(text) and not KEY_PATTERN.search(text), (
            f'Credential pattern in Git history: {object_id}:{name}; inspect locally')
        count += 1
    print(f'History: {count} reachable text blobs checked for credential patterns')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true')
    args = parser.parse_args()
    check_results()
    check_files()
    if args.history:
        check_history()
