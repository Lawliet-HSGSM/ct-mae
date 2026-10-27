# CT-MAE follow-up: partial adaptation and error analysis

Completed study initiated on 2026-10-23. Preserve the completed seed-42
baseline and official train/validation/test splits. Use its exact MAE checkpoint
and the same 1,728 labeled indices; do not repeat pretraining.

## Controlled training comparison

- New partial classifier: train the last 3 of 12 encoder blocks, final LayerNorm,
  and Linear(384, 11). Freeze patch embedding, CLS and the first 9 blocks.
- Frozen modules stay in evaluation mode; the last blocks retain the original
  stochastic-depth schedule while training. Fixed position embeddings stay fixed.
- Copy the full fine-tuning recipe: 100 epochs, peak LR 3e-4, warmup 5,
  AdamW, physical batch 32 with accumulation 2, seed 42, identical augmentation.
- Select the earliest checkpoint maximizing validation macro-F1, then evaluate
  test. Use a separate W&B run and output directory. Save resume checkpoints.
- Check baseline configuration, source checkpoint SHA-256, and labeled subset.
- Run meaningful freeze/gradient tests and a GPU smoke test before long training.

## Analysis and artifacts

- Reuse original predictions and verify alignment before paired comparisons.
- Class distributions, per-class precision/recall/F1, normalized confusion,
  recovered/regressed examples, confidence-ranked and fixed-random errors.
- Show actual CT pixels with ground truth, each model's prediction and score.
- Extract full-image mean-pooled 384-dimensional features in eval mode for
  random, MAE, supervised, full and partial encoders. Linear probing shares the
  MAE encoder. Fit projections only on clean train features; plot official test.
- PCA uses centering. SVD uses uncentered features. Record retained variance and
  approximation checks; do not interpret separate axes as shared coordinates.
- Use same examples/colors, show all test points and selected error overlays.
- Diagnose the train/validation loss gap at fixed checkpoint and eval mode,
  comparing clean and augmented train inputs with clean validation inputs.
- Deliver readable Vietnamese offline HTML, Markdown, figures and structured
  results. Explain observations separately from possible visual causes.
- Results remain exploratory, one seed, under the current training budget;
  inspection of test examples must not be described as independent tuning.

Oversampling and contrastive training are conditional future extensions, not
part of this controlled follow-up. Visual appearance alone does not establish
that texture caused a classification error.
# Reproducing the report

From the repository root, after collecting the completed Kaggle output under
`reports/followup/artifacts`:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
uv run --extra train python -m ct_mae.followup_report --baseline 'reports/baseline' --followup 'reports/followup/artifacts' --archive 'data/organamnist_128.npz' --output 'reports/followup'
uv run --extra train python scripts/check_followup_report.py 'reports/followup/followup_report.html' --output 'reports/followup/browser-qa'
```

The report includes exact PCA/SVD bases and test indices, a frozen-parameter audit,
per-class confusion counts, paired prediction changes, fixed-checkpoint loss
diagnostics, and both extreme-error and randomly chosen example galleries.
The HTML contains its own data, images, CSS and JavaScript and opens offline.

## Measured outcome (2026-10-23, seed 42)

Kaggle notebook `lawlietwilllucky/ct-mae-baseline`, version 8, completed the
100-epoch partial run, then feature extraction and fixed-checkpoint diagnostics.
Version 7 remains the original baseline. The best partial checkpoint was selected
at epoch 25 using validation macro-F1; the full run still completed 100 epochs.

| Classifier | Test Accuracy | Test macro-F1 | Trainable parameters |
|---|---:|---:|---:|
| Supervised from scratch | 81.36% | 81.17% | 21,323,915 |
| SSL linear probing | 83.81% | 83.13% | 4,235 |
| SSL full fine-tuning | 88.75% | 88.04% | 21,323,915 |
| SSL partial, last 3 blocks | 89.17% | 88.55% | 5,328,395 |

- Full → partial: 689 errors corrected, 614 new errors; net 75 more correct.
- Heart → left lung: 111 → 27. Spleen → left lung: 102 → 49.
- Left kidney → spleen worsened: 295 → 345. Neither model confused left/right lungs.
- MAE-only features show partial grouping, not eleven isolated clusters in PCA.
  On the fixed balanced 2,200-test-image pool, 10-neighbor same-label agreement in
  384D was 58.05% random, 82.28% MAE, 80.84% scratch, 89.52% full, 89.94% partial.
- Partial train peak VRAM: 0.55 GiB, full: 1.73 GiB, both on Tesla T4. Training-loop
  time was 6.4 vs 11.7 minutes; the complete stages took 22.5 vs 25.5 minutes.
- Independent local checkpoint audit confirmed 112 frozen tensors unchanged and
  38 selected encoder tensors updated. The subset and input/checkpoint checksums
  matched the original baseline. W&B remotely confirmed the completed run.

W&B: https://wandb.ai/lawlietwilllucky-nguyen-thang/ct-mae/runs/4ru5zzt8

Report: `reports/followup/followup_report.html`.
The same directory contains Markdown, 19 PNG figures, structured measurements,
projection bases, observation notes, and raw collected artifacts. HTML embeds its
data/images and is self-contained; large features/checkpoints stay separate.

Verification: 52 local tests, Ruff, checksum/parameter audit, actual-data metric
recomputation, and Edge browser checks of filters, hover, image loading, offline
operation and widths 320/768/1440. These remain exploratory one-seed findings;
test images were inspected, not used to select a new configuration or oversampling.
