# CT MAE

Self-supervised representation learning on 2D OrganAMNIST CT organ crops.

The user owns model and preprocessing decisions. Implementation follows the design
agreed in discussion. The first milestone is an inspection tool for the original
dataset, before any model, augmentation, normalization or label budget is chosen.

## Active worktree

Dataset exploration is developed in `.worktrees/dataset`, on
`feature/dataset-explorer`. Raw datasets are shared in `data/` at this repository
root and excluded from Git.

See `SPEC.md` for this milestone. Run commands from the active worktree.

## Dataset

OrganAMNIST includes a supervised class label for each image in its supplied
train, validation and test arrays. SSL ignores the labels during pretraining;
downstream classification uses them again.

- Official source: https://medmnist.com/
- Data record: https://zenodo.org/records/10519652
- Resolution generation: https://github.com/MedMNIST/MedMNIST/blob/main/on_medmnist_plus.md
- Dataset: CC BY 4.0. Cite MedMNIST and its source studies when using it.
