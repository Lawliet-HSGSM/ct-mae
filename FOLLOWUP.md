# CT-MAE follow-up: partial adaptation and error analysis

The user authorized execution on 2026-10-23. Preserve the completed seed-42
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
