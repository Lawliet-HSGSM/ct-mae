# Label budget extension: 5% → 10%

The user authorized an additional label-budget experiment on 2026-10-23.

- Official OrganAMNIST128 splits unchanged; seed42, all recipes unchanged.
- 10% means 3,456 labeled train images. Extend the original 1,728 indices by
  class-proportional largest-remainder allocation and sampling without replacement.
  Save the resulting indices, parent hash, exact class counts and subset hash.
- Reuse the exact original 200-epoch MAE checkpoint; do not repeat pretraining.
- Train scratch, SSL linear, SSL full, SSL partial (last3blocks/finalnorm/head).
  Each trains 100epochs, same LR/batch/augmentation as its original counterpart.
- This fixes epochs, not optimizer updates: 10% has 54updates/epoch vs27 at5%.
  Report that limitation rather than attributing every gain to labels alone.
- Pick earliest best validation macro-F1, then test each selected checkpoint.
  Test is already inspected from previous rounds; this remains exploratory.
- W&B project ct-mae, new group/run suffix labels3456; private Kaggle version9
  planned on the existing secret-enabled notebook. Smoke all four branches first.
- New output directory ct-mae-labels10pct; preserve v7/v8 and old local reports.
- Deliver an offline HTML/Markdown report comparing both budgets, per-class F1,
  confusion directions, selected checkpoints, training curves, compute and update
  counts. Two label-budget points do not establish an entire efficiency curve.

Implementation reuses the training/evaluation engine through an optional model
factory. The label-study factory explicitly validates the original SSL protocol
while allowing the downstream label count to change. Existing baseline transfer
checks remain intact. No modified/fabricated pretraining metadata is used.
