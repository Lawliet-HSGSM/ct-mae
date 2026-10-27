# Same-organ crop and display review

The user requested comparisons for the right-kidney examples while reading the
5%-label follow-up report. This review reuses the original predictions and data;
there is no new training or model selection.

## What was verified

- References: test indices 9916, 12619, 8329, 13003, 11590, 10180, 2495, 10267.
- Original report PNGs match official uint8 128x128 arrays pixel for pixel.
- In Edge at 1440 CSS px, the previous gallery displayed each 128x128 image at
  258.25x258.25 CSS px with automatic image interpolation. Display enlargement
  adds no source information; the PNG itself was neither recropped nor blurred.
- Official 28/64/128 archives passed their MD5 checks and have aligned test labels.
  All 153 PNGs embedded in this review round-trip to their source arrays exactly.
- The test images used by the models were the official 128 version, scaled /255,
  without test augmentation or another crop/resize in our pipeline.

The official [MedMNIST+ generation notes](https://github.com/MedMNIST/MedMNIST/blob/main/on_medmnist_plus.md#organacsmnist)
describe resizing the source crop separately to each target resolution. The 128
version is not an enlargement of the 28 version. Changing target resolution does
not restore the context outside the crop. The available NPZ files do not provide
the original pre-crop image, source crop dimensions, bounding boxes or masks;
they cannot isolate how much blur originates in source CT versus resizing.

## Selection and limits

All comparison candidates have the same true organ label as the references.
For the 1,965 right-kidney test crops, candidate pools are:

| Condition | Available crops |
| --- | ---: |
| All four classifiers correct | 1,038 |
| Scratch wrong, full correct | 424 |
| Full wrong, partial correct | 153 |

Within each pool, select the three smallest grayscale MSE distances after
area-averaging 128x128 pixels to 32x32, at a fixed /255 scale. Exclude the reference
itself and break ties by official test index. This metric finds coarse visual
layout, not verified anatomical similarity. Nearest within an outcome pool may
still look different. The same control can recur for multiple references.

The eight references plus their controls produce 51 distinct crops. This is an
outcome-conditioned visual review, not a random sample for estimating the
prevalence or causal effect of poor crops. Unknown scan IDs also prevent treating
similar-looking crops as independent patients. The resolution panel displays
28/64/128 source images at equal CSS size; model outcomes apply only to 128.

## Observations from the actual figures

- Reference #9916 and full-correct #16507 have a similar coarse arrangement of
  bright and dark structures and both look blurred. Full calls #9916 pancreas
  (99.7%) but #16507 right kidney (65.9%).
- Control #8479 also looks low-detail but all four classifiers predict right
  kidney correctly. This weakens the blanket explanation that a blurred crop
  must be unclassifiable, without proving blur has no effect.
- For #12619, the same large dark region and narrow bright structures remain in
  all three official resolutions. Higher pixel count does not expand the crop.
  Some matched outcome controls have noticeably different layouts, so these
  pairs alone cannot establish what caused the reference error.
- The source-quality/crop hypothesis remains worth checking with a defined
  review criterion and correct/incorrect controls within each organ. The current
  review does not establish label errors, a crop-generation bug or causal reliance
  on texture. No denoising, sharpening, oversampling or test filtering was applied.

## Reproduce locally

From the repository root:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
uv run --extra train python -m ct_mae.crop_review --data-dir data --baseline reports/baseline --followup reports/followup/artifacts --analysis reports/followup/analysis.json --output reports/followup
uv run --extra train python scripts/check_crop_review.py reports/followup/crop_review.html --output reports/followup/crop-browser-qa
```

The new offline page is `reports/followup/crop_review.html` under the main
checkout, with JSON and `figures/crop_review_pairs.png`. The existing follow-up
report links to it when regenerated with `--reuse`. UI controls switch reference,
nearest-control rank, display scale and pixelated/smooth enlargement. They do
not change evaluated model input or predictions.

Validation passed: 62 project tests, Ruff, official archive MD5s, 153 pixel
round-trips, original-report pixel comparisons, all 24 reference/rank choices,
native image dimensions, zoom/smoothing, responsive widths 320/768/1024/1440,
offline browser controls, navigation between both pages, and the original
follow-up gallery/PCA/SVD browser checks. No JS errors or external resource
requests were observed.
