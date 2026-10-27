# Public result snapshot

This directory is a compact export of the completed experiments, not a set of
model checkpoints. No API account, data archive or torch installation is needed
to read it or redraw the scalar charts.

| File | Contents |
| --- | --- |
| [metrics.csv](metrics.csv) | One row per classifier run: 8 initial classifiers and 20 continuation variants |
| [results.json](results.json) | Config, class counts, selected-checkpoint metrics, scalar histories, resource measurements, environment and provenance hashes |
| [subset_1728.json](subset_1728.json) | Original stratified 5% train indices, class counts and hash |
| [subset_3456.json](subset_3456.json) | Nested 10% train indices, parent identity and hash |
| [figure_provenance.json](figure_provenance.json) | Input/output hashes for the retained reconstruction, PCA and paired-case exports |

Scores in JSON/CSV are fractions in [0, 1]; Markdown tables display percentages.
Epochs are one-based except candidate epoch 0, which denotes the unchanged
classifier source before continuation. Confusion matrices have true labels in
rows and predicted labels in columns. Class IDs follow the `labels` array.

`initial` means the original 100-epoch scratch/linear/full/partial comparison at
each label budget. `continuation` means 50 epochs at 5% or 25 at 10%, starting
from the original 5%-label MAE-derived classifiers. The two studies must not be
combined into one unqualified leaderboard. `partial` refers to the last three
encoder blocks, final LayerNorm and classifier, not just the classifier head.

SHA-256 values identify the source run metadata, test metadata, prediction file
and selected checkpoint. Config/learning histories are deliberately exported
through a field allowlist; they do not contain authentication values, local
machine paths or temporary signed download URLs. MAE pretraining has no organ
classification score and is recorded separately under `pretrain`.

From the repository root:

```powershell
uv run python scripts/publish_results.py
uv run python scripts/check_publication.py
```

The first command redraws learning, class-distribution, confusion and label-budget
figures and rewrites the scalar CSV. It does not redo model inference. Retained
PCA/qualitative figures require the original feature/image/prediction artifacts
to regenerate; follow [REPRODUCING.md](../REPRODUCING.md). The original seed-42
artifacts are retained locally and are not distributed in this repository.
