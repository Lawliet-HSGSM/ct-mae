"""Download selected outputs from an MCP-provided Kaggle file manifest; no remote writes."""

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from ct_mae.config import write_json
from ct_mae.pipeline import sha256_file


def selected_path(filename: str, *, root_name='ct-mae-followup') -> Path | None:
    path = PurePosixPath(filename)
    if (path.is_absolute() or '..' in path.parts or '\\' in filename
            or not path.parts or path.parts[0] != root_name
            or any(':' in part for part in path.parts) or 'wandb' in path.parts):
        return None
    relative = Path(*path.parts[1:])
    if relative.suffix not in ('.json', '.npz', '.pt'):
        return None
    if 'smoke' in relative.parts and relative.suffix == '.pt':
        return None
    return relative


def fetch_file(item, output, *, root_name='ct-mae-followup'):
    relative = selected_path(item['file_name'], root_name=root_name)
    if relative is None:
        return None
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.download')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(item['url'], timeout=180) as response:
                with temporary.open('wb') as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
            temporary.replace(destination)
            return relative.as_posix()
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise RuntimeError(f'Could not download {relative.as_posix()}') from None
            time.sleep(2)


def validate(output):
    job = json.loads((output / 'job_status.json').read_text('utf-8'))
    if job['status'] != 'completed':
        raise ValueError('Remote follow-up is incomplete')
    run = json.loads((output / 'partial/run.json').read_text('utf-8'))
    manifest = json.loads((output / 'subset.json').read_text('utf-8'))
    if (run['status'] != 'completed' or run['completed_epochs'] != 100
            or run['train_samples'] != 1728 or run['val_samples'] != 6491
            or run['subset_sha256'] != manifest['indices_sha256']):
        raise ValueError('Partial run used an unexpected budget or subset')
    audit = json.loads((output / 'freeze_audit.json').read_text('utf-8'))
    if not audit['passed'] or audit['checkpoint_sha256'] != sha256_file(output / 'partial/best.pt'):
        raise ValueError('Selected partial checkpoint does not match freeze audit')
    provenance = json.loads((output / 'embedding_provenance.json').read_text('utf-8'))
    for stage, source in provenance['stages'].items():
        for split in ('train', 'test'):
            path = output / f'embeddings/{stage}_{split}.npz'
            if sha256_file(path) != source[split]['sha256']:
                raise ValueError(f'Embedding checksum failed: {stage}/{split}')
    return {'status': 'verified', 'source_sha256': job['source_sha256'],
            'completed_epochs': run['completed_epochs'], 'wandb_url': run['wandb_url']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    listing = json.loads(args.manifest.read_text('utf-8'))
    files = [item for item in listing['files'] if selected_path(item['file_name']) is not None]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_file, item, args.output) for item in files]
        for future in as_completed(futures):
            print('DOWNLOADED', future.result(), flush=True)
    verification = validate(args.output)
    write_json(args.output / 'collection_verification.json', verification)
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
