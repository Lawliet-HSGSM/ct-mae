# Nested 10% classifier adaptation

User requested a fresh 5% subset or a 10% label budget on 2026-10-26.
Use 3,456 stratified training indices: the original 1,728 plus 1,728 additional
indices, seed 42. This is cumulative 10% supervised data. Replacing training
images while retaining a classifier trained on the original labels would also
increase cumulative label exposure; it would not be a fresh 5% experiment.

Index SHA-256:
`c940edbefbacd3d0beeb996da01f3982b1bd82fb31177a6b2a031bc82192669d`.
Selection uses the existing deterministic nested-stratified sampling function.
MAE is not retrained. Start from original full/partial 5% classifiers, not from
the previous adaptation winners or classifiers trained on 10% labels.

Repeat A–E for full and partial: shuffled CE; paired CE; paired weighted CE;
paired CE+SupCon; paired weighted CE+SupCon. Preserve both old and added examples
in the training pool. All branches share indices and two affine views per image.

Match the previous planned optimizer budget, batch 32/accumulation 2:

| Setting | 5% adaptation | New 10% adaptation |
| --- | ---: | ---: |
| Images | 1,728 | 3,456 |
| Epochs | 50 | 25 |
| Updates/epoch | 27 | 54 |
| Planned updates/run | 1,350 | 1,350 |
| LR warmup epochs | 5 | 2.5 |
| SupCon lambda warmup epochs | 5 | 2.5 |
| Warmup updates | 135 | 135 |

Actual skipped AMP updates, if any, are reported. LR remains 3e-5, projector×10,
lambda maximum 0.05, temperature 0.1. Other optimizer/architecture settings are
unchanged. Source epoch 0 remains a candidate; choose earliest best validation
macro-F1 and choose the variant by validation, before any new test evaluations.

Three confusion pairs remain liver/right-lung, left-kidney/spleen,
left-kidney/right-kidney, using source validation only. The two source validation
matrices jointly contain 85, 27, 28 cross-confusions for these pairs respectively.
Class-weight deficits also use source validation F1; normalization uses the new
training class frequencies. Validation does not supply gradient updates.

Finding hard pairs from training data is also valid, preferably using held-out or
out-of-fold train predictions when in-sample training accuracy is saturated.
Test must not determine pairs, sampling, weights, hyperparameters or checkpoints.
This project's test has already been inspected in earlier stages, so future test
results remain exploratory and are not a fresh confirmatory evaluation.

Primary comparisons are A–E within the new 10% setting. Comparisons against old
5% runs match planned updates and LR/lambda schedules, but validation/checkpoint
evaluation happens 25 times rather than 50, and each image receives 25 rather than 50
passes. These limits must be explicit; do not attribute every difference solely
to label count or describe the existing 100-epoch label study as the same recipe.

Kaggle root: `ct-mae-adaptation10pct`. W&B group:
`ct-mae-adaptation10pct-seed42`. Existing secret authorization is reused.
Run all ten smoke variants and verify finite W&B metrics before production;
abort on checksum, split, smoke or logging failures. Original artifacts are read
only, new outputs live in `reports/adaptation-10pct/`. Estimated GPU time 2–3 hours.

Artifact audit checks label/source hashes, every epoch's index order and counts,
frozen partial tensors, actual update counts, earliest-best selection and test
predictions/calibration. Produce a separate offline HTML/Markdown report with
per-class/error changes and original-pixel examples after completion.

## Measured result

Kaggle notebook version 12 completed all ten production runs and all ten smoke
variants. Every production run made 1,350 successful updates without AMP skips.
Source bundle commit: `a25ce8cf919f8e5667d49f47dad128f2290f49e8`;
bundle SHA-256: `458a5f5807d18d050626fc6544e8dedc0213ba41e6f0f3f64f18c71b5c9a594e`.
The logged interval to completion was 8,530.9 seconds, about 2h22m, on one Tesla T4.

| Source | Variant | Selected epoch | Val macro-F1 | Test Accuracy | Test macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Full | Original 5% source | 0 | 97.7741% | 88.7501% | 88.0351% |
| Full | A: CE | 6 | 98.0878% | 90.8707% | 90.2599% |
| Full | B: sampler | 7 | 97.9357% | 91.3657% | 90.9072% |
| Full | C: weighted CE | 7 | 97.9503% | 91.5457% | 91.0839% |
| Full | D: SupCon | 21 | 97.9485% | 91.6751% | 91.1449% |
| Full | E: combined | 7 | 97.9541% | 91.5570% | 91.0991% |
| Partial | Original source and A–E | 0 | 98.1075% | 89.1720% | 88.5456% |

Validation selects full/A within the full-source candidates and retains the
partial source within partial-source candidates. D's higher test result does
not override this decision. All variants inherit MAE and the original 5%
classifier; A is the CE continuation control for this stage.

Compared against full/A at 5% with the same adaptation update budget, full/A 10%
improves Accuracy 89.0876%→90.8707% and macro-F1 88.6839%→90.2599%.
It recovers 601 previous errors and introduces 284 new ones, a net 317 additional
correct predictions. The report includes paired galleries against both the
original source and the corresponding 5% CE adaptation control.

Full/A improves per-organ F1 for all 11 organs relative to the original source,
but not all confusion directions: right-kidney→left-kidney increases 161→232.
Against B, D improves measured test Accuracy by 0.31 percentage points and
macro-F1 by 0.24 points, while validation macro-F1 differs by only 0.013 points.
The combined E adds only two correct test predictions over C. Treat small
differences as exploratory, not established improvements from one seed.

The retained partial source does not mean the additional runs did no learning.
Final clean training Accuracy is 99.54–99.59% for partial and 100% for full;
partial's validation scores remain below source throughout. The study fixes
optimization budget, rather than establishing every setting's best achievable
result after unlimited training.

Model rankings differ between validation and test. Even across the two source
families, original partial has slightly higher validation macro-F1 than full/A,
while full/A has higher test metrics. Do not retroactively choose a global
winner using test. Validation representativeness and the previously inspected
test are material limits, discussed alongside the split-review diagnostic.

## Artifacts and reproduction

Report: `reports/adaptation-10pct/adaptation_report.html`, with Markdown/JSON,
standalone figures, original-pixel paired examples and browser verification.
`artifacts/collection_verification.json` independently verifies hashes, data,
sampling traces, frozen weights, selection and prediction/calibration metrics.
W&B completion proofs cover all ten production runs.

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
uv run --extra train python -m ct_mae.adaptation_collect --verify-only --sources .worktrees/adaptation10/work/adaptation-sources --archive data/organamnist_128.npz --output reports/adaptation-10pct/artifacts
uv run --extra train python -m ct_mae.adaptation_report --artifacts reports/adaptation-10pct/artifacts --archive data/organamnist_128.npz --previous reports/adaptation-study/adaptation_comparison.json --output reports/adaptation-10pct
uv run --extra train python scripts/check_adaptation_report.py reports/adaptation-10pct/adaptation_report.html --output reports/adaptation-10pct/browser
```

For a fresh download, use the MCP manifest, `--manifest` instead of
`--verify-only`, and `--root-name ct-mae-adaptation10pct`. Do not commit expiring
signed URLs. Browser checks verify all twelve result rows/matrices, native
128px examples, responsive layout and absence of external page dependencies.
