# Dataset exploration milestone

## Objective and accepted scope

Create a reproducible, read-only inspection of the official 2D OrganAMNIST data.
Compare the same samples at 28, 64 and 128 pixels without selecting model input
preprocessing. The user has requested implementation of this milestone.

## Acceptance

- Download official files and validate their published MD5 checksums.
- Measure sample counts, shapes, uniformity, dtype, pixel statistics, label
  availability, class counts and storage from the actual arrays.
- Check label/index alignment before comparing different resolutions.
- Show all 11 classes and the same image at different resolutions.
- Produce an offline interactive HTML explorer, PNG previews and JSON/CSV data.
- Preserve original values when displaying images: fixed grayscale 0..255,
  no per-image contrast normalization or training augmentation.
- Explain that stored square dimensions do not establish physical size or pixel
  spacing; those per-sample acquisition details are not provided here.

## Structure and commands

Python 3.12, NumPy, Pillow and Matplotlib; a plain HTML/CSS/JavaScript report.
Source lives in `src/ct_mae/`, behavior tests in `tests/`, generated outputs in
`reports/` (ignored), and source assets in the shared root `data/` (ignored).

```powershell
uv sync --python 3.12 --group dev
uv run python -m ct_mae.download --data-dir ..\..\data --sizes 28 64 128
uv run python -m ct_mae.explore --data-dir ..\..\data --sizes 28 64 128 --output reports
uv run pytest
uv run ruff check .
```

Use typed, direct functions and explicit paths, for example
`summarize_split(images, labels) -> dict`. No server or external frontend CDN is
required by the generated report.

## Validation and implementation order

1. Bootstrap repository and worktree; implement checksum-verified downloads.
2. Test and implement numeric inspection and sample selection.
3. Render report and plots; verify the real data and browser interactions.
4. Review and commit the verified worktree.

Tests cover corrupt archives, exact numeric statistics and sample alignment.
Browser validation covers class/split selection, resolution comparison and native
pixel display. Do not commit raw data, environments, reports or credentials.

## Boundaries

No training, resizing of source assets, fitted preprocessing, augmentation,
masking experiments or new classification splits are part of this milestone.
Keep the official splits and report any metadata disagreement explicitly.

## Decisions after inspection

The user selected official 128×128 OrganAMNIST images and an initial 8×8 patch
size, giving 256 patches per image. The smaller 64×64 images looked too small
to the user; patch size remains adjustable in later experiments.

The user subsequently accepted and requested implementation of the complete
baseline protocol in `BASELINE.md`, with its machine-readable recipe at
`src/ct_mae/baseline.json`. This extends the project after the explorer milestone;
the explorer continues to display the original data without training transforms.
