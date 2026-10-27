"""Package a private Kaggle follow-up while preserving the completed baseline outputs."""

import base64
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

BOOTSTRAP = r'''import base64, hashlib, importlib.metadata, io, json, os, subprocess, sys
import urllib.request, zipfile
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["CT_MAE_COMMIT"] = __COMMIT__
payload = base64.b64decode(__PAYLOAD__)
assert hashlib.sha256(payload).hexdigest() == __DIGEST__
os.environ["CT_MAE_BUNDLE_SHA256"] = __DIGEST__
source = Path("/kaggle/working/ct-mae-followup-source")
source.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
    bundle.extractall(source)
sys.path.insert(0, str(source))
os.environ["PYTHONPATH"] = str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")
from packaging.version import Version
requirements = [("numpy","numpy>=2.0,<3","2.0"), ("Pillow","Pillow>=11,<13","11"),
                ("matplotlib","matplotlib>=3.9,<4","3.9"),
                ("wandb","wandb>=0.19,<1","0.19"), ("pytest","pytest>=8","8")]
install = []
for name, requirement, minimum in requirements:
    try: version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: install.append(requirement); continue
    if Version(version) < Version(minimum): install.append(requirement)
if install: subprocess.run([sys.executable,"-m","pip","install",*install],check=True)
import torch
if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU unavailable")
from ct_mae.config import write_json
from ct_mae.pipeline import environment_info, sha256_file
from ct_mae.download import fetch_asset, official_asset
from kaggle_secrets import UserSecretsClient
logging_config = json.loads((source/"baseline/resolved_config.json").read_text())["wandb"]
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret(logging_config["secret_label"])
import wandb
api = wandb.Api()
print("W&B authenticated", api.default_entity, flush=True)
output = Path("/kaggle/working/ct-mae-followup")
output.mkdir(exist_ok=True)
write_json(output/"environment.json", environment_info(torch.device("cuda")))
baseline = source / "baseline"
checkpoint_urls = __URLS__
for stage, url in checkpoint_urls.items():
    target = baseline / stage / "best.pt"
    target.parent.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(url,timeout=120) as response, target.open("wb") as stream:
        while block := response.read(1024*1024): stream.write(block)
    expected = json.loads((baseline/stage/"expected.json").read_text())["sha256"]
    if sha256_file(target) != expected: raise RuntimeError(f"Checkpoint checksum failed: {stage}")
    print("BASELINE VERIFIED",stage,expected,flush=True)
data = Path("/kaggle/temp/ct-mae-data")
fetch_asset(official_asset(128),data)
subprocess.run([sys.executable,"-m","pytest","-q",str(source/"tests")],cwd=source,check=True)
__RUN_COMMANDS__
'''

FOLLOWUP_COMMANDS = r'''
command = [sys.executable,"-m","ct_mae.followup","--baseline",str(baseline),
           "--data-dir",str(data),"--output",str(output),"--blocks","3"]
smoke = [sys.executable,"-m","ct_mae.followup","--baseline",str(baseline),
         "--data-dir",str(data),"--output",str(output/"smoke"),"--blocks","3","--mode","smoke"]
write_json(output/"job_status.json",{"status":"smoke","source_sha256":__DIGEST__})
subprocess.run(smoke,cwd=source,check=True)
print("FOLLOWUP SMOKE GATE PASSED",flush=True)
write_json(output/"job_status.json",{"status":"training_and_extracting","source_sha256":__DIGEST__})
subprocess.run([*command,"--mode","all"],cwd=source,check=True)
write_json(output/"job_status.json",{"status":"completed","source_sha256":__DIGEST__})
print("FOLLOWUP COMPLETE",flush=True)
'''

TEMPLATE = BOOTSTRAP.replace('__RUN_COMMANDS__', FOLLOWUP_COMMANDS)


def build_script(destination: Path, baseline: Path, checkpoint_urls: dict[str, str], *,
                 template: str = TEMPLATE) -> dict:
    root = Path(__file__).resolve().parents[2]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted((root / 'src/ct_mae').iterdir()):
            if file.suffix in ('.py', '.json', '.html', '.css', '.js'):
                bundle.writestr(file.relative_to(root / 'src').as_posix(), file.read_bytes())
        for file in sorted((root / 'tests').glob('test_*.py')):
            bundle.writestr(file.relative_to(root).as_posix(), file.read_bytes())
        for file in [root / 'LICENSE', root / 'THIRD_PARTY_NOTICES.md',
                     *sorted((root / 'licenses').glob('*.txt'))]:
            if file.exists():
                bundle.writestr(file.relative_to(root).as_posix(), file.read_bytes())
        for filename in ('resolved_config.json', 'subset.json'):
            bundle.writestr(f'baseline/{filename}', (baseline / filename).read_bytes())
        for stage in checkpoint_urls:
            bundle.writestr(f'baseline/{stage}/run.json',
                            (baseline / stage / 'run.json').read_bytes())
            from ct_mae.pipeline import sha256_file
            bundle.writestr(f'baseline/{stage}/expected.json', json.dumps({
                'sha256': sha256_file(baseline / stage / 'best.pt'),
            }))
    payload = stream.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    commit = subprocess.run(['git','rev-parse','HEAD'],cwd=root,check=True,capture_output=True,
                            text=True).stdout.strip()
    script = template.replace('__PAYLOAD__', repr(base64.b64encode(payload).decode()))
    script = script.replace('__COMMIT__', repr(commit)).replace('__DIGEST__', repr(digest))
    script = script.replace('__URLS__', repr(checkpoint_urls))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(script, 'utf-8')
    return {'source_sha256':digest,'commit':commit,'bytes':len(script.encode())}
