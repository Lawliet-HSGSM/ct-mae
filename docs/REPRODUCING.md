# Reproducing CT-MAE

Run commands from the repository root. Python 3.12 and uv are used for local
checks. The training modules also work in the recorded Kaggle Linux runtime;
Kaggle uses its installed CUDA PyTorch instead of the Windows CPU uv index.

## What the repository includes

Code, tests, exact selected train indices, scalar run histories, metrics and
provenance hashes are committed. Datasets, checkpoints, feature arrays, W&B logs,
temporary signed URLs and large self-contained HTML reports are excluded.
The public snapshot is an export of completed runs, not a checkpoint download.
Private Kaggle notebooks and W&B run links in historical records may require
the owner's account and are not a dependency for running your own experiment.

## Local setup and data exploration

```powershell
uv sync --locked --group dev
uv run python -m ct_mae.download --data-dir data --sizes 128
uv run python -m ct_mae.explore --data-dir data --sizes 128 --output reports/exploration
```

Open `reports/exploration/explorer.html`. Optional resolution comparison:

```powershell
uv run python -m ct_mae.download --data-dir data --sizes 28 64 128
uv run python -m ct_mae.explore --data-dir data --sizes 28 64 128 --output reports/resolutions
```

The official 128 archive MD5 is `eeae80d0a227a8d099027e1b3cfd3b60`.
Source URLs, expected sizes and checksums are in
[organamnist.json](../src/ct_mae/organamnist.json).

## Tests and CPU input preparation

```powershell
uv sync --locked --extra train --group dev
uv run --extra train ruff check .
uv run --extra train pytest -q
uv run --extra train python -m ct_mae.run --stage prepare --device cpu --wandb-mode disabled --data-dir data --output reports/local-check
```

Tests use synthetic data, require no API keys and do not download the dataset.
Preparation uses the downloaded archive to write previews, the selected indices
and parameter counts without training. For an end-to-end GPU diagnostic:

```powershell
python -m ct_mae.run --stage all --smoke --device cuda --wandb-mode disabled --data-dir data --output reports/gpu-check
```

Smoke mode retains the full model with only 2 epochs per stage, 512 train images
and 128 validation images. It does not evaluate the official test split or
reproduce production scores. CPU smoke is supported but can be slow.

## Baseline on a CUDA GPU

The full configuration is [baseline.json](../src/ct_mae/baseline.json). On a
machine with compatible CUDA PyTorch and the package dependencies installed:

```powershell
python -m ct_mae.run --stage prepare --data-dir data --output reports/baseline --wandb-mode disabled
python -m ct_mae.run --stage all --device cuda --data-dir data --output reports/baseline --wandb-mode disabled
```

The first command writes the subset, input preview and resolved config. `all`
runs scratch → pretrain → linear → full, then evaluates selected classifiers.
Keep W&B mode consistent between prepare and training because it is part of the
recorded config fingerprint. Use a new output directory when changing settings.
`--resume` restores a stage's `last.pt` when present. Production checkpoints include
model, optimizer, scheduler state, AMP scaler, epoch and RNG state.

On Windows, uv intentionally installs CPU wheels. For GPU training, create a
separate environment with the appropriate CUDA wheels from the
[PyTorch installer](https://pytorch.org/get-started/locally/); do not overwrite a
working Kaggle CUDA runtime with Windows CPU dependencies.

### Kaggle and W&B

1. Create your own private Kaggle notebook, enable GPU and Internet.
2. Add a W&B API key in Kaggle Secrets and enable it for that notebook. Never put
   the key in source or a committed notebook.
3. Generate the self-contained source bundle:

```powershell
uv run python -m ct_mae.kaggle --output work/kaggle_baseline.py --secret-label WANDB_API_KEY
```

Paste/run that generated script in the notebook, or submit it using your Kaggle
connector. `--secret-label` is the Secret label, not the key value. If omitted,
the historical label in `baseline.json` is used. The script verifies CUDA,
data checksums, W&B connectivity, unit tests and GPU smoke execution before
production. It records actual software versions and source-bundle hashes.
Kaggle outputs are under `/kaggle/working/ct-mae`; copy that directory to
`reports/baseline` after completion. Never commit a generated bundle containing
temporary signed download URLs used by follow-up jobs.

The observed runtime was Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128,
W&B 0.28.1 and Tesla T4. The local uv lock and recorded GPU environment are
different environments; the lock is not a claim of bitwise GPU reproducibility.
Changing only an operational secret label also changes the config fingerprint.

## Partial fine-tuning and 10% labels

Keep these artifact paths when collecting your own runs:

```text
reports/baseline/              resolved_config.json, subset.json, model_summary.json
  pretrain/                   best.pt, last.pt, run.json, reconstruction.png
  scratch|linear|finetune/     best.pt, last.pt, run.json, test.json, test_predictions.npz
reports/followup/artifacts/
  partial/                    best.pt, last.pt, run.json, test.json, test_predictions.npz
reports/labels10pct/artifacts/ resolved_config.json, subset.json, four classifier directories
```

Given a completed baseline, train the last three blocks without repeating MAE:

```powershell
python -m ct_mae.followup --baseline reports/baseline --output reports/followup/artifacts --data-dir data --mode all --blocks 3 --device cuda
python -m ct_mae.label_study --baseline reports/baseline --output reports/labels10pct/artifacts --data-dir data --count 3456 --blocks 3 --device cuda
```

These stages inherit the baseline's logging configuration. `followup --mode all`
also extracts clean features and fixed-checkpoint loss diagnostics. It needs the
original baseline classifier checkpoints. See
[partial/error protocol](experiments/partial-and-errors.md) and
[label-budget protocol](experiments/label-budgets.md).

## Continuation with CE, batch pairs and SupCon

Prepare copies of the original 5%-label classifiers for both label budgets:

```powershell
python -c "from pathlib import Path; from ct_mae.adaptation_kaggle import prepare_sources; prepare_sources(Path('reports/baseline'), Path('reports/followup/artifacts/partial'), Path('work/adaptation-sources'))"
python -m ct_mae.adaptation_study --sources work/adaptation-sources --data-dir data --output reports/adaptation-study/artifacts --labeled-count 1728 --device cuda
python -m ct_mae.adaptation_study --sources work/adaptation-sources --data-dir data --output reports/adaptation-10pct/artifacts --labeled-count 3456 --device cuda
```

All ten full/partial variants complete before final test evaluation. Existing
outputs resume automatically after protocol/fingerprint checks. A 10% run starts
from the same original sources as 5%; do not substitute the 5% continuation winner
or the initial 10%-trained classifier. Protocols:
[5%](experiments/adaptation-5pct.md), [10%](experiments/adaptation-10pct.md).

## Generate offline reports

These commands require completed artifacts, not just the committed scalar snapshot:

```powershell
python -m ct_mae.run --stage report --output reports/baseline
python -m ct_mae.followup_report --baseline reports/baseline --followup reports/followup/artifacts --archive data/organamnist_128.npz --output reports/followup
python -m ct_mae.label_report --baseline5 reports/baseline --partial5 reports/followup/artifacts --labels10 reports/labels10pct/artifacts --archive data/organamnist_128.npz --output reports/label-study
python -m ct_mae.adaptation_report --artifacts reports/adaptation-study/artifacts --archive data/organamnist_128.npz --output reports/adaptation-study
python -m ct_mae.adaptation_report --artifacts reports/adaptation-10pct/artifacts --archive data/organamnist_128.npz --previous reports/adaptation-study/adaptation_comparison.json --output reports/adaptation-10pct
python -m ct_mae.split_review --archive data/organamnist_128.npz --subset reports/baseline/subset.json --artifacts reports/adaptation-study/artifacts --output reports/split-review
```

The original manually written observations are not invented for a new run. Inspect
its actual plots and examples before adding explanations. Report browser checks
live in `scripts/check_*report.py` and `scripts/check_*review.py`; they use installed
Microsoft Edge through Playwright. For example:

```powershell
python scripts/check_adaptation_report.py reports/adaptation-10pct/adaptation_report.html --output reports/adaptation-10pct/browser
```

## Public result export

Redraw the public learning/label-budget/confusion figures from committed numbers,
without downloading data, installing torch or logging into an external account:

```powershell
uv run python scripts/publish_results.py
```

To replace the snapshot with your own completed experiments, explicitly provide
all original artifact roots. This checks the archive MD5, prediction alignment,
Accuracy/F1/confusion counts and earliest-best checkpoint selection:

```powershell
python scripts/publish_results.py --baseline reports/baseline --partial reports/followup/artifacts --labels10 reports/labels10pct/artifacts --adapt5 reports/adaptation-study/artifacts --adapt10 reports/adaptation-10pct/artifacts --archive data/organamnist_128.npz
```

The PCA and paired-example images are retained exports from the follow-up report;
their original filenames and SHA-256 hashes are recorded in
[figure_provenance.json](results/figure_provenance.json). Regenerating those images
requires the feature/prediction artifacts and report commands above. Do not treat
redrawing a scalar snapshot as an independent rerun of inference.
