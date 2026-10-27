# Experiment records

These records preserve the protocol and measured outcome of each completed stage.
Commands use paths relative to the repository root. Large reports and training
artifacts are local outputs, not files included in a fresh clone.
For a portable end-to-end workflow, start with [REPRODUCING.md](../REPRODUCING.md).

| Record | Question |
| --- | --- |
| [Data exploration](data-exploration.md) | What is stored in the official archives? |
| [Baseline](baseline.md) | Does MAE help classification with 5% training labels? |
| [Partial tuning and errors](partial-and-errors.md) | How do last-3-block tuning and encoder features compare? |
| [Label budgets](label-budgets.md) | What changes at 10%, with the same epoch budget? |
| [Crop review](crop-review.md) | Do similar-looking crops receive different predictions? |
| [5% continuation](adaptation-5pct.md) | Do pairing, weighting or SupCon help the existing classifier? |
| [10% continuation](adaptation-10pct.md) | Do more labels help at the same continuation-update budget? |
| [Split review](split-review.md) | Where does the validation–test accuracy gap occur? |

The [results discussion](../RESULTS.md) separates observations from hypotheses
and records which comparisons are exploratory.
