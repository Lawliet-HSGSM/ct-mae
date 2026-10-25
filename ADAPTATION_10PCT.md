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
