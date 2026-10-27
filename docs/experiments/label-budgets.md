# Label budget extension: 5% → 10%

Additional label-budget experiment initiated on 2026-10-23.

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

Run commands from the repository root.
Training commit: `5b93a45`; submitted bundle SHA-256:
`2d98286ddf1f2a9143462d044d6956f8fa6f278f4b721651c17c82102ecb5158`.

Use Kaggle MCP to list completed version `v9` outputs (paginate all files) and
save the signed file manifest under ignored `work/kaggle_v9_manifest.json`.
The locally verified expected subset is `work/expected_subset.json`; its hash is
`c940edbefbacd3d0beeb996da01f3982b1bd82fb31177a6b2a031bc82192669d`.
Do not commit temporary signed URLs.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
uv run --extra train python -m ct_mae.label_collect --manifest work/kaggle_v9_manifest.json --expected-subset work/expected_subset.json --baseline reports/baseline --output reports/labels10pct/artifacts
uv run --extra train python -m ct_mae.label_report --baseline5 reports/baseline --partial5 reports/followup/artifacts --labels10 reports/labels10pct/artifacts --archive data/organamnist_128.npz --output reports/label-study
uv run --extra train python scripts/check_label_report.py reports/label-study/label_efficiency_report.html --output reports/label-study/browser-qa
```

Inspect actual class/paired-case figures, put evidence-grounded notes in
`reports/label-study/observations.json`, and rerender with `--reuse`.
The report embeds all its figures and uses no external scripts/fonts.

## Completed execution: Kaggle v9

Completed on 2026-10-23 (Asia/Bangkok), approximately 140 minutes including
setup/smoke/evaluation; four downstream training sessions total 132.1 minutes.
Runtime: Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu128, NumPy 2.1.3.
All four stages completed 100 epochs with 3,456 train / 6,491 validation images.
Select the earliest maximum validation macro-F1; evaluate its checkpoint on all
17,778 test crops. The larger subset exactly matches the locally checked
manifest and contains every original 5% index.

| Method | Test accuracy 5% | Test accuracy 10% | Test macro-F1 10% | Selected epoch 10% |
| --- | ---: | ---: | ---: | ---: |
| Supervised scratch | 81.36% | 84.98% | 84.92% | 84 |
| SSL linear probing | 83.81% | 86.18% | 85.85% | 66 |
| SSL full fine-tuning | 88.75% | 91.66% | 91.17% | 63 |
| SSL partial, last 3 blocks | 89.17% | 91.00% | 90.53% | 72 |

All four methods improve overall accuracy and macro-F1, but not every organ or
confusion direction improves. Full fine-tuning gains F1 on all 11 organs; its
right-kidney → left-kidney errors nevertheless increase 161 → 256. Left-kidney →
left-lung errors fall 44 → 0. The qualitative galleries include both recovered
and newly incorrect cases; similar crops must not be counted as independent
patients without scan IDs. Full overtakes partial at 10%, so this single seed
does not establish a universally better number of unfrozen blocks.

The full-vs-scratch accuracy gap is 7.39 points at 5% and 6.69 points at 10%.
Full at 5% also exceeds scratch at 10% (88.75% vs 84.98%), with the extra SSL
data/pretraining compute acknowledged. Actual optimizer updates at 10% are
5,400 per stage; at 5% they are 2,700, except full with 2,699 after one AMP skip.
This remains a fixed-epoch, single-seed exploratory comparison.

Verified source MAE SHA-256:
`2c70f88c6b743975dc2a93ff56870501b9b848caa11c85238e57a0122c735a5b`.
Independent audit confirms all 112 frozen partial-encoder tensors unchanged,
38 selected encoder tensors updated, and the complete linear encoder unchanged.
All four W&B runs were verified completed:

- [Scratch](https://wandb.ai/lawlietwilllucky-nguyen-thang/ct-mae/runs/04utg0ci)
- [Linear](https://wandb.ai/lawlietwilllucky-nguyen-thang/ct-mae/runs/6uoetaog)
- [Full](https://wandb.ai/lawlietwilllucky-nguyen-thang/ct-mae/runs/cwcubrec)
- [Partial](https://wandb.ai/lawlietwilllucky-nguyen-thang/ct-mae/runs/09bzmi0y)

The offline report is `reports/label-study/label_efficiency_report.html` under the
main checkout, with matching Markdown, recomputed prediction metrics in JSON,
23 PNG figures, and manually inspected observations. Artifacts and generated
reports stay outside Git. Collection checksums, data/subset identity, budget,
test metrics and encoder freeze audits are recorded in the artifact directory.

Validation passed: 58 tests in the Kaggle bundle; 60 local tests after report
completion; Ruff; prediction/manifest/checkpoint verification; and Edge checks
of all eight confusion selections, embedded images, responsive widths and
absence of JavaScript errors or external resource requests.
