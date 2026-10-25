"""Package the approved adaptation matrix with verified existing source checkpoints."""

import json
import math
import shutil

from ct_mae.config import write_json
from ct_mae.followup_kaggle import BOOTSTRAP, build_script
from ct_mae.pipeline import sha256_file


def prepare_sources(baseline, partial, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ('subset.json', 'resolved_config.json'):
        shutil.copyfile(baseline / filename, destination / filename)
    shutil.copyfile(baseline / 'resolved_config.json', destination / 'baseline.json')
    manifest = json.loads((baseline / 'subset.json').read_text('utf-8'))
    hashes = {}
    for scope, directory in (('full', baseline / 'finetune'), ('partial', partial)):
        target = destination / scope
        target.mkdir(exist_ok=True)
        for filename in ('best.pt', 'run.json'):
            shutil.copyfile(directory / filename, target / filename)
        shutil.copyfile(target / 'best.pt', destination / f'{scope}.pt')
        hashes[scope] = sha256_file(target / 'best.pt')
    write_json(destination / 'source_manifest.json', {
        'checkpoints': hashes, 'subset_sha256': manifest['indices_sha256'],
    })
    return hashes


COMMANDS = r'''
import shutil
shutil.copyfile(baseline/"resolved_config.json",baseline/"baseline.json")
hashes = {}
for scope in ("full","partial"):
    shutil.copyfile(baseline/scope/"best.pt",baseline/f"{scope}.pt")
    hashes[scope] = sha256_file(baseline/f"{scope}.pt")
manifest = json.loads((baseline/"subset.json").read_text())
write_json(baseline/"source_manifest.json",{
    "checkpoints":hashes,"subset_sha256":manifest["indices_sha256"]})
command = [sys.executable,"-m","ct_mae.adaptation_study","--sources",str(baseline),
           "--data-dir",str(data)]
from ct_mae.adaptation_kaggle import verify_adaptation
write_json(output/"job_status.json",{"status":"smoke"})
subprocess.run([*command,"--output",str(output/"smoke"),"--smoke"],cwd=source,check=True)
verify_adaptation(output/"smoke", api, smoke=True)
print("ADAPTATION SMOKE GATE PASSED",flush=True)
write_json(output/"job_status.json",{"status":"training"})
subprocess.run([*command,"--output",str(output)],cwd=source,check=True)
verify_adaptation(output, api, smoke=False)
write_json(output/"job_status.json",{"status":"completed"})
print("ADAPTATION COMPLETE",flush=True)
'''


def build_adaptation_script(destination, sources, urls, *, labeled_count=1728):
    if labeled_count not in (1728, 3456):
        raise ValueError('Only 5% and nested 10% are supported')
    commands = COMMANDS.replace('str(data)]', f'str(data),"--labeled-count","{labeled_count}"]')
    template = BOOTSTRAP.replace('__RUN_COMMANDS__', commands)
    root = 'ct-mae-adaptation10pct' if labeled_count == 3456 else 'ct-mae-adaptation'
    template = template.replace('ct-mae-followup', root)
    return build_script(destination, sources, urls, template=template)


def verify_adaptation(output, api, *, smoke):
    expected = 'smoke_completed' if smoke else 'completed'
    proof = []
    for scope in ('full', 'partial'):
        for variant in 'ABCDE':
            record = json.loads((output / scope / variant / 'run.json').read_text('utf-8'))
            if record['status'] != expected:
                raise RuntimeError(f'{scope}/{variant} did not finish')
            if sum(row['train']['optimizer_steps'] for row in record['history']) < 1:
                raise RuntimeError('No successful optimizer steps')
            remote = api.run(f'{api.default_entity}/ct-mae/{record["wandb_run_id"]}')
            if remote.summary.get('status') != expected:
                raise RuntimeError('W&B did not confirm completion')
            for key in ('train/ce', 'train/weighted_ce', 'train/supcon', 'val/macro_f1'):
                if not math.isfinite(float(remote.summary.get(key, float('nan')))):
                    raise RuntimeError(f'W&B missing scalar metric: {key}')
            proof.append({'scope': scope, 'variant': variant, 'url': remote.url,
                          'best_epoch': record['best_epoch'], 'status': expected})
    write_json(output / 'wandb_verification.json', proof)
    return proof
