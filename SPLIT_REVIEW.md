# Train, augmented train, validation and test inspection

Requested 2026-10-26 while the fixed 10% adaptation job was running. This is
read-only exploratory analysis; it does not change that job's pairs, weights,
hyperparameters or checkpoint selection.

Report: `reports/split-review/split_review.html`, with JSON, Markdown and PNG
figures beside it. The page contains 1,866 unique original-pixel examples and
two augmentation views for each packed train example. It shows four columns:
train, the same train image after affine, validation, test. Controls cover all
11 organs, full/partial 5% source classifiers, random/correct/wrong evaluation
samples, train5%/all-train/added5% pools, two sample pages and pixel 1:1/2× display.

Only train and its augmented view are a matched image pair. Across splits the
examples merely share the class. Random galleries and random error galleries
use seed 42; they are not a way to estimate population error rates.

Both classifiers are the original sources used for adaptation, not its winners
or models being trained in the new job. Validation predictions come from source
evaluation, test predictions from the corresponding source-checksummed artifact.

## Measured gap

| Classifier | Val Accuracy | Test Accuracy | Class-mix contribution | Within-class recall contribution |
| --- | ---: | ---: | ---: | ---: |
| Full source | 97.6429% | 88.7501% | +0.0914 pp | −8.9841 pp |
| Partial source | 98.1513% | 89.1720% | +0.1087 pp | −9.0879 pp |

Exact descriptive decomposition, using class proportions p and recalls r:

`Acc_test − Acc_val = Σ (p_test − p_val) r_val + Σ p_test (r_test − r_val)`.

The gap is mostly poorer predictions within the same organ, rather than test
containing a larger proportion of difficult classes. This is arithmetic, not
a causal explanation or an intrinsic dataset-quality ranking.

For example, full left-kidney recall drops 97.89%→64.24%. In the highest dark-pixel
group using training quartile cutoffs, test has 539 images with 63.3% error, while
validation contains only 9 examples in that group. However, the nearly-no-dark
test group still has 24.2% error across 847 images. Dark area alone is insufficient
to explain the gap, and these observations do not establish a causal shortcut.

## Pixel statistics and augmentation

Measure every original train/val/test image and every original 5% train image
with one deterministic augmentation view. Descriptors: pixel mean, standard
deviation, fraction ≤0.05 and mean absolute neighboring-pixel difference, all
after scaling by 255. Per-organ plots show median/IQR with 10–90% whiskers.
Conditional error tables use train-only quartile cutoffs and all evaluation
examples in each organ, displaying empty buckets explicitly.

These are image descriptors, not organ masks or validated quality/blur scores.
Strong edges can reflect contrast, boundaries or noise. Dark pixels are not a
lung annotation. Affine fill 0 introduces dark borders and can change the edge
descriptor independently of internal image detail.

Augmentation uses the existing TwinViewDataset: ±5° rotation, translation≤5%,
bilinear, fill 0, no flip/crop. Examples use seed 42/epoch 0. PNG display quantization
differs from the float32 training tensor by at most half a grayscale step;
statistics are computed on float32 before quantization. The preview distribution
is the configured recipe, not a recorded replay of source epoch 70/25 inputs.

For left kidney, mean dark fraction is 3.43% in raw train 5%, 8.42% after affine,
0.073% in validation and3.96% in test. More synthetic dark border pixels do not
guarantee coverage of the test's spatial/context patterns. Test edge descriptors
are also close to train for this organ, so “test is generally blurrier” is not
established by this analysis.

Official split context and its limitations are cited in the report using
[MedMNIST v2](https://www.nature.com/articles/s41597-022-01721-8) and
[MedMNIST+ generation notes](https://github.com/MedMNIST/MedMNIST/blob/main/on_medmnist_plus.md).
NPZ lacks per-image scan IDs, native HU and pre-crop images. No patient-level
independence, crop-cause attribution or new test-based training decisions follow.

## Reproduce

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
uv run --extra train python -m ct_mae.split_review --archive data/organamnist_128.npz --subset .worktrees/adaptation10/work/adaptation-sources/subset.json --artifacts reports/adaptation-study/artifacts --output reports/split-review
uv run --extra train python scripts/check_split_review.py reports/split-review/split_review.html --output reports/split-review/browser
```

`--reuse` redraws figures/page from saved JSON. Actual artifact verification
decoded all 1,866 original PNGs back to official pixels and all 1,052 augmentation
PNGs back to their deterministic rounded tensors; max display error 0.001961 in
the 0–1 scale. Edge browser QA covers 66 organ/model/mode combinations, paired
indices, train pools, both views and responsive widths 320/768/1024/1440.
