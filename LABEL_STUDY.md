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
  submitted on the existing secret-enabled notebook. Smoke all four branches first.
- New output directory ct-mae-labels10pct; preserve v7/v8 and old local reports.
- Deliver an offline HTML/Markdown report comparing both budgets, per-class F1,
  confusion directions, selected checkpoints, training curves, compute and update
  counts. Two label-budget points do not establish an entire efficiency curve.

Implementation reuses the training/evaluation engine through an optional model
factory. The label-study factory explicitly validates the original SSL protocol
while allowing the downstream label count to change. Existing baseline transfer
checks remain intact. No modified/fabricated pretraining metadata is used.

## Local collection and report

Worktree: `D:\Development\SchoolProject\ct-mae\.worktrees\labels`.
Training commit: `5b93a45`; submitted bundle SHA-256:
`2d98286ddf1f2a9143462d044d6956f8fa6f278f4b721651c17c82102ecb5158`.

Use Kaggle MCP to list completed version `v9` outputs (paginate all files) and
save the signed file manifest under ignored `work/kaggle_v9_manifest.json`.
The locally verified expected subset is `work/expected_subset.json`; its hash is
`c940edbefbacd3d0beeb996da01f3982b1bd82fb31177a6b2a031bc82192669d`.
Do not commit temporary signed URLs.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
& '..\dataset\.venv\Scripts\python.exe' -m ct_mae.label_collect --manifest work/kaggle_v9_manifest.json --expected-subset work/expected_subset.json --baseline ../dataset/reports/kaggle-v7 --output ../../reports/labels10pct/artifacts
& '..\dataset\.venv\Scripts\python.exe' -m ct_mae.label_report --baseline5 ../dataset/reports/kaggle-v7 --partial5 ../../reports/followup/artifacts --labels10 ../../reports/labels10pct/artifacts --archive ../../data/organamnist_128.npz --output ../../reports/label-study
& '..\dataset\.venv\Scripts\python.exe' scripts/check_label_report.py ../../reports/label-study/label_efficiency_report.html --output ../../reports/label-study/browser-qa
```

Inspect actual class/paired-case figures, put evidence-grounded notes in
`reports/label-study/observations.json`, and rerender with `--reuse`.
The report embeds all its figures and uses no external scripts/fonts.
