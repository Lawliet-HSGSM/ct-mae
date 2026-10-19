"""Build a self-contained Kaggle script; submission and monitoring use the Kaggle MCP."""

import argparse
import base64
import hashlib
import io
import subprocess
import zipfile
from pathlib import Path

TEMPLATE = r'''import base64, hashlib, io, json, os, subprocess, sys, zipfile
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MPLBACKEND"] = "Agg"
os.environ["CT_MAE_COMMIT"] = __COMMIT__
payload = base64.b64decode(__PAYLOAD__)
assert hashlib.sha256(payload).hexdigest() == __DIGEST__
os.environ["CT_MAE_BUNDLE_SHA256"] = __DIGEST__
source = Path("/kaggle/working/ct-mae-source")
source.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(source)
sys.path.insert(0, str(source))
os.environ["PYTHONPATH"] = str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")
import importlib.metadata
from packaging.version import Version
dependencies = [("numpy", "numpy>=2.0,<3", "2.0"),
                ("matplotlib", "matplotlib>=3.9,<4", "3.9"),
                ("Pillow", "Pillow>=11,<13", "11"),
                ("wandb", "wandb>=0.19,<1", "0.19"),
                ("pytest", "pytest>=8", "8")]
install = []
for name, requirement, minimum in dependencies:
    try: installed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: install.append(requirement); continue
    if Version(installed) < Version(minimum): install.append(requirement)
if install:
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "--disable-pip-version-check", *install],
                   check=True)
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Kaggle did not supply a CUDA-enabled GPU runtime; stop before training")
if Version(torch.__version__.split("+")[0]) < Version("2.6"):
    raise RuntimeError("PyTorch 2.6 or later is required")
from ct_mae.config import config_digest, load_config
from ct_mae.run import diagnostic_config
from ct_mae.pipeline import environment_info
config = load_config()
SMOKE = __SMOKE__
RESUME = __RESUME__
if SMOKE: config = diagnostic_config(config)
output = Path("/kaggle/working/ct-mae")
experiment = output / "smoke" if SMOKE else output
experiment.mkdir(parents=True, exist_ok=True)
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret(config["wandb"]["secret_label"])
import wandb
api = wandb.Api()
entity = config["wandb"]["entity"] or api.default_entity
environment = environment_info(torch.device("cuda"))
environment["bundle_sha256"] = __DIGEST__
(output / "environment.json").write_text(json.dumps(environment, indent=2))
print("ENVIRONMENT", json.dumps(environment), flush=True)
if RESUME:
    for stage in ("scratch", "pretrain", "linear", "finetune"):
        reference = (f"{entity}/{config['wandb']['project']}/"
                     f"ct-mae-{config_digest(config)[:12]}-{stage}-resume:latest")
        try: artifact = api.artifact(reference)
        except wandb.errors.CommError as error:
            if "not found" in str(error).lower() or "does not exist" in str(error).lower():
                print("No saved checkpoint for", stage, flush=True); continue
            raise
        artifact.download(root=str(experiment / stage))
subprocess.run([sys.executable, "-m", "pytest", "-q", str(source / "tests")], check=True,
               cwd=source)
# A CPU-side augmentation must not reset the CUDA RNG used for masking/drop path.
import numpy as np
from ct_mae.data import OrganDataset
before = torch.cuda.get_rng_state()
_ = OrganDataset(np.zeros((1,128,128), dtype=np.uint8), augment=True)[0]
assert torch.equal(before, torch.cuda.get_rng_state())
command = [sys.executable, "-m", "ct_mae.run", "--stage", __STAGE__,
           "--data-dir", "/kaggle/temp/ct-mae-data", "--output", str(output)]
if SMOKE: command.append("--smoke")
if RESUME: command.append("--resume")
subprocess.run(command, check=True, cwd=source)
proof = []
for stage in ("scratch", "pretrain", "linear", "finetune"):
    path = experiment / stage / "run.json"
    if not path.exists(): continue
    record = json.loads(path.read_text())
    remote = api.run(f"{entity}/{config['wandb']['project']}/{record['wandb_run_id']}")
    assert remote.summary.get("preflight/ok") == 1
    assert remote.summary.get("status") == record["status"]
    proof.append({"stage": stage, "url": remote.url, "status": remote.summary.get("status"),
                  "last_validation_loss": record["history"][-1]["val"]["loss"],
                  "peak_vram_bytes": max(row["resource"]["peak_vram_bytes"]
                                         for row in record["history"])})
(experiment / "wandb_verification.json").write_text(json.dumps(proof, indent=2))
print("WANDB VERIFIED", json.dumps(proof), flush=True)
'''


def build_script(destination: Path, *, stage: str = "all", smoke: bool = False,
                 resume: bool = False) -> dict:
    root = Path(__file__).resolve().parents[2]
    buffer = io.BytesIO()
    paths = sorted(path for path in (root / "src" / "ct_mae").iterdir()
                   if path.suffix in (".py", ".json", ".html", ".css", ".js"))
    paths += sorted((root / "tests").glob("test_*.py"))
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            name = path.relative_to(root / "src") if path.is_relative_to(root / "src") else (
                path.relative_to(root)
            )
            info = zipfile.ZipInfo(name.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                            capture_output=True, check=True).stdout.strip()
    script = TEMPLATE.replace("__PAYLOAD__", repr(base64.b64encode(payload).decode()))
    for name, value in (("__COMMIT__", commit), ("__DIGEST__", digest), ("__STAGE__", stage),
                        ("__SMOKE__", smoke), ("__RESUME__", resume)):
        script = script.replace(name, repr(value))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(script, "utf-8")
    return {"script": str(destination), "source_sha256": digest, "commit": commit,
            "stage": stage, "smoke": smoke, "resume": resume, "bytes": len(script.encode())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "scratch", "pretrain", "linear", "finetune"],
                        default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import json
    print(json.dumps(build_script(args.output, stage=args.stage, smoke=args.smoke,
                                   resume=args.resume)))


if __name__ == "__main__":
    main()
