"""Reuse the verified Kaggle bootstrap for four classifiers at the 10% label budget."""

from pathlib import Path

from ct_mae.followup_kaggle import BOOTSTRAP, build_script

COMMANDS = r'''
command = [sys.executable,"-m","ct_mae.label_study","--baseline",str(baseline),
           "--data-dir",str(data),"--count","3456","--blocks","3"]
write_json(output/"job_status.json",{"status":"smoke","source_sha256":__DIGEST__})
subprocess.run([*command,"--output",str(output/"smoke"),"--smoke"],cwd=source,check=True)
print("LABEL STUDY SMOKE GATE PASSED",flush=True)
write_json(output/"job_status.json",{"status":"training","source_sha256":__DIGEST__})
subprocess.run([*command,"--output",str(output)],cwd=source,check=True)
write_json(output/"job_status.json",{"status":"completed","source_sha256":__DIGEST__})
print("LABEL STUDY COMPLETE",flush=True)
'''

TEMPLATE = BOOTSTRAP.replace('ct-mae-followup', 'ct-mae-labels10pct').replace(
    '__RUN_COMMANDS__', COMMANDS,
)


def build_label_script(destination: Path, baseline: Path, checkpoint_url: str) -> dict:
    return build_script(destination, baseline, {'pretrain': checkpoint_url}, template=TEMPLATE)
