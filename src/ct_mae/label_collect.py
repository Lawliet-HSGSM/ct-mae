"""Collect only the label-study outputs, then verify the completed experimental protocol."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from ct_mae.config import write_json
from ct_mae.followup import audit_frozen
from ct_mae.followup_collect import fetch_file, selected_path
from ct_mae.label_study import STAGES
from ct_mae.pipeline import sha256_file

ROOT_NAME = 'ct-mae-labels10pct'


def validate_collection(output, baseline, expected):
    job = json.loads((output / 'job_status.json').read_text('utf-8'))
    manifest = json.loads((output / 'subset.json').read_text('utf-8'))
    if job['status'] != 'completed' or manifest != expected:
        raise ValueError('Study incomplete or sampled indices differ from the checked subset')
    source = baseline / 'pretrain/best.pt'
    source_hash = sha256_file(source)
    proof = json.loads((output / 'wandb_verification.json').read_text('utf-8'))
    if {row['stage'] for row in proof if row['status'] == 'completed'} != set(STAGES):
        raise ValueError('W&B did not verify all four stages')
    results = {}
    for stage in STAGES:
        run = json.loads((output / stage / 'run.json').read_text('utf-8'))
        test = json.loads((output / stage / 'test.json').read_text('utf-8'))
        if (run['status'] != 'completed' or run['smoke'] or run['completed_epochs'] != 100
                or len(run['history']) != 100 or run['train_samples'] != 3456
                or run['val_samples'] != 6491
                or run['subset_sha256'] != manifest['indices_sha256']
                or run['config']['seed'] != 42):
            raise ValueError(f'Wrong data, protocol or budget for {stage}')
        if stage != 'scratch' and run['ssl_checkpoint_sha256'] != source_hash:
            raise ValueError('Study changed its original MAE checkpoint')
        if sha256_file(output / stage / 'best.pt') != test['checkpoint_sha256']:
            raise ValueError(f'Selected {stage} checkpoint checksum differs')
        results[stage] = {key: test[key] for key in ('accuracy', 'macro_f1', 'selected_epoch')}
    partial = audit_frozen(output / 'partial/best.pt', source, 3)
    original = torch.load(source, map_location='cpu', weights_only=True)['model']
    linear = torch.load(output / 'linear/best.pt', map_location='cpu', weights_only=True)['model']
    for key, value in linear.items():
        if key.startswith('encoder.') and not torch.equal(value, original[key]):
            raise ValueError('Linear probing changed its original encoder')
    return {'status': 'verified', 'source_sha256': job['source_sha256'],
            'results': results, 'partial_independent_audit': partial,
            'linear_encoder_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-subset', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    listing = json.loads(args.manifest.read_text('utf-8'))
    files = [row for row in listing['files']
             if selected_path(row['file_name'], root_name=ROOT_NAME) is not None]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_file, item, args.output, root_name=ROOT_NAME)
                   for item in files]
        for future in as_completed(futures):
            print('DOWNLOADED', future.result(), flush=True)
    result = validate_collection(args.output, args.baseline,
                                 json.loads(args.expected_subset.read_text('utf-8')))
    write_json(args.output / 'collection_verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
