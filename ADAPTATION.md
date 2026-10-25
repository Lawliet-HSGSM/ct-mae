# Classifier adaptation study

Approved 2026-10-25: warm-start the original full (epoch70) and partial (epoch25)
5%-label classifiers. Preserve their 1,728 training indices and the official
validation/test splits. No MAE retraining, new labels or test-based selection.

For each source, run 50 adaptation epochs, seed42, with five variants:
A shuffled CE; B paired-batch CE; C paired-batch class-weighted CE;
D paired-batch CE+SupCon; E paired-batch class-weighted CE+SupCon.
All variants use two independent mild affine views of each image. Every training
index occurs exactly once per epoch. Batch32 images, accumulation2: 27 planned
updates per epoch, 1,350 per run. Contrastive comparisons do not cross microbatches.

Pair priorities are liver/lung-right, kidney-left/spleen, kidney-left/kidney-right,
selected from the existing full/partial validation confusions. In that order,
pair floor(half the smaller remaining class) without reusing images. Distribute
the resulting paired units evenly among batches and fill with the singletons.

Per source, freeze class weights from initial validation F1: deficit=1-F1;
raw=1+deficit/max(deficit), or ones if all deficits are zero. Normalize by the
original labeled training class proportions. These weights apply only to CE.
Weighted CE is normalized over the entire accumulation group.

SupCon projector: Linear384→384, ReLU, Linear384→128, L2 normalization;
temperature0.1; lambda ramps 0→0.05 over five epochs. Compute similarities and
logsumexp in float32. Classification uses the encoder features before projection.

Fresh AdamW: LR3e-5, betas(0.9,0.999), eps1e-8, WD0.05, no decay for bias/norm;
projector LR multiplier10. Five-epoch warmup, cosine to1e-6 (projector1e-5).
Preserve original encoder trainability/drop-path. AMP FP16, one GPU.
Profile full/E first; if necessary use batch16/accumulation4 for every run.

Initial source is an epoch0 candidate. Select earliest best validation macro-F1,
then compare variants by validation only, tie preference source,A,B,C,D,E.
Freeze all choices before final test evaluation. Preserve source artifacts;
report mixed metrics as tradeoffs and one-seed results as exploratory.

Outputs include model/projector/optimizer/scaler/scheduler/RNG checkpoints,
source/data/subset hashes, class weights, sampled indices and rich W&B logs.
Report Accuracy, macro/per-class F1, precision/recall, confusion changes,
NLL/ECE15, paired outcomes and original-pixel galleries. Diagnose pooled encoder
features using one fixed balanced validation pool (100/class), not projector
training loss alone. HTML/Markdown report lives under reports/adaptation-study.

## Measured outcome — Kaggle version 11

The private notebook completed all ten production runs after all ten GPU smoke
runs passed. Every production run completed 50 epochs and 1,350 successful
optimizer updates, using batch 32/accumulation 2 on one Tesla T4. Training code:
`e5b0f1882b3804ebd4d6f5820c4270dcc85dc19b`. Runtime: Python 3.13.15,
Torch 2.11.0+cu128, CUDA 12.8, W&B 0.28.1. Local/report tools use the Windows CPU venv.

Each row reports the checkpoint chosen by validation, including epoch 0.
The source rows are original classifiers, not additional training runs.

| Source | Variant | Selected epoch | Val macro-F1 | Test Accuracy | Test macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Full | Source | 0 | 97.7741% | 88.7501% | 88.0351% |
| Full | A: CE | 13 | 97.8768% | 89.0876% | 88.6839% |
| Full | B: sampler | 25 | 97.8550% | 89.5208% | 89.1695% |
| Full | C: weighted CE | 0 | 97.7741% | 88.7501% | 88.0351% |
| Full | D: SupCon | 1 | 97.7805% | 89.1608% | 88.5623% |
| Full | E: combined | 0 | 97.7741% | 88.7501% | 88.0351% |
| Partial | Source and A–E | 0 | 98.1075% | 89.1720% | 88.5456% |

Validation chooses **full/A**, and retains the **partial source**. Full/B has
higher measured test metrics than full/A, but test does not change selection.
Full/A recovers 360 errors and introduces 300: +60 correct predictions out of 17,778.
Left-lung F1 increases 3.28 percentage points; right-kidney F1 decreases 3.16.
Full/A does not dominate the original partial source on both Accuracy and macro-F1.

Weighted CE and SupCon have not justified replacing the existing recipe in this
single-seed study. Epoch 0 selections mean training did not improve validation;
they do not mean optimizer updates were absent. Full/D's selected epoch 1 uses
mean lambda 0.00481, before full contrastive weighting. Its epoch 50 train SupCon
is 1.8365, close to the positive-count lower bound 1.8317, while validation F1 falls
to 97.3404%. Strong train-objective optimization does not imply better validation.

For a batch containing n base images of an anchor's class, each of its two views
has P=2n−1 positives. SupCon averages negative log probabilities over positives,
so its anchor loss is bounded below by log(P). The report computes this bound
from actual epoch 50 sampling traces; it is a lower bound, not a guarantee that
every embedding/temperature can attain it.

The final source/checkpoint decision was frozen before twelve test evaluations
(two sources and ten candidate variants). The test had already been inspected
in previous project stages; findings remain exploratory. No scan IDs are
available to establish independence of visually similar crops.

## Artifacts and reproduction

Report: `D:/Development/SchoolProject/ct-mae/reports/adaptation-study/adaptation_report.html`.
The matching Markdown, JSON, original-pixel example PNGs, plots and browser
verification sit beside it. `artifacts/` contains best/last checkpoints, predictions,
run histories, actual index traces, W&B completion proofs and collection audits.

```powershell
Set-Location 'D:\Development\SchoolProject\ct-mae'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
uv run --extra train python -m ct_mae.adaptation_collect --verify-only --sources .worktrees/adaptation/work/adaptation-sources --archive data/organamnist_128.npz --output reports/adaptation-study/artifacts
uv run --extra train python -m ct_mae.adaptation_report --artifacts reports/adaptation-study/artifacts --archive data/organamnist_128.npz --output reports/adaptation-study
uv run --extra train python scripts/check_adaptation_report.py reports/adaptation-study/adaptation_report.html --output reports/adaptation-study/browser
```

For downloading anew, pass a fresh MCP file manifest to `adaptation_collect`
using `--manifest` instead of `--verify-only`. Signed download URLs are temporary,
local-only work files and are not committed. The independent audit recomputes
test metrics/calibration and validates original-source hashes, all sampling
traces, the complete update budget, unchanged frozen tensors, earliest-best
checkpoint/variant choices, and balanced validation-feature indices.

The offline report includes explicit pairs of recovered/regressed examples and
correct controls. Source pixels are lossless 128×128 with grayscale 0–255; confidence
is uncalibrated softmax. Visual observations describe patterns and hypotheses,
not anatomical diagnoses or confirmed causes of model error.

References: [SupCon](https://arxiv.org/abs/2004.11362),
[reference loss implementation](https://github.com/HobbitLong/SupContrast/blob/master/losses.py),
[MAE](https://arxiv.org/abs/2111.06377). Joint CE+SupCon classifier adaptation,
weight formula and frequency-preserving pair sampler are project adjustments.
