# Classifier adaptation study

Approved 2026-10-25: warm-start the original full (epoch70) and partial (epoch25)
5%-label classifiers. Preserve their 1,728 training indices and the official
validation/test splits. No MAE retraining, new labels or test-based selection.

For each source, run 50 adaptation epochs, seed42, with five variants:
A shuffled CE; B paired-batch CE; C paired-batch class-weighted CE;
D paired-batch CE+SupCon; E paired-batch class-weighted CE+SupCon.
All variants use two independent mild affine views of each image. Every training
index occurs exactly once per epoch. Batch32 images, accumulation2: 27 planned
updates per epoch, 1,350 per run. Contrastive comparisons do not cross microbatches.

Pair priorities are liver/lung-right, kidney-left/spleen, kidney-left/kidney-right,
selected from the existing full/partial validation confusions. In that order,
pair floor(half the smaller remaining class) without reusing images. Distribute
the resulting paired units evenly among batches and fill with the singletons.

Per source, freeze class weights from initial validation F1: deficit=1-F1;
raw=1+deficit/max(deficit), or ones if all deficits are zero. Normalize by the
original labeled training class proportions. These weights apply only to CE.
Weighted CE is normalized over the entire accumulation group.

SupCon projector: Linear384→384, ReLU, Linear384→128, L2 normalization;
temperature0.1; lambda ramps 0→0.05 over five epochs. Compute similarities and
logsumexp in float32. Classification uses the encoder features before projection.

Fresh AdamW: LR3e-5, betas(0.9,0.999), eps1e-8, WD0.05, no decay for bias/norm;
projector LR multiplier10. Five-epoch warmup, cosine to1e-6 (projector1e-5).
Preserve original encoder trainability/drop-path. AMP FP16, one GPU.
Profile full/E first; if necessary use batch16/accumulation4 for every run.

Initial source is an epoch0 candidate. Select earliest best validation macro-F1,
then compare variants by validation only, tie preference source,A,B,C,D,E.
Freeze all choices before final test evaluation. Preserve source artifacts;
report mixed metrics as tradeoffs and one-seed results as exploratory.

Outputs include model/projector/optimizer/scaler/scheduler/RNG checkpoints,
source/data/subset hashes, class weights, sampled indices and rich W&B logs.
Report Accuracy, macro/per-class F1, precision/recall, confusion changes,
NLL/ECE15, paired outcomes and original-pixel galleries. Diagnose pooled encoder
features using one fixed balanced validation pool (100/class), not projector
training loss alone. HTML/Markdown report lives under reports/adaptation-study.
