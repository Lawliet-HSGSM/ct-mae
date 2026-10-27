# Attribution and license notices

## Project material

Copyright (c) 2026 CT-MAE contributors.
Code and original project documentation/figures are distributed under
Creative Commons Attribution-NonCommercial 4.0 International, except where a
separate attribution below applies. See [LICENSE](LICENSE).

## Masked Autoencoders (MAE)

- Source: <https://github.com/facebookresearch/mae>
- Reference files: `models_mae.py`, `util/pos_embed.py`, `PRETRAIN.md`.
- Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
- License: [CC BY-NC 4.0](https://github.com/facebookresearch/mae/blob/main/LICENSE),
  reproduced in this repository's [LICENSE](LICENSE).

`src/ct_mae/models.py` adapts the architecture, random masking/restoration,
sine–cosine positions and initialization from that implementation. Changes
include a grayscale 128px input, 8px patches, a 384-wide encoder, a smaller
128-wide decoder, PyTorch SDPA attention without timm, mean-pooled classification,
and explicit full/partial/frozen training modes. Training and reporting code
implement the project's own protocol rather than copying the ImageNet recipe.

## Supervised Contrastive Learning / SupContrast

- Paper: <https://arxiv.org/abs/2004.11362>
- Reference code: <https://github.com/HobbitLong/SupContrast/blob/master/losses.py>
- Copyright (c) 2020, Yonglong Tian. All rights reserved.
- License: [BSD 2-Clause](licenses/SupContrast-BSD-2-Clause.txt).

`src/ct_mae/adaptation_loss.py` implements the supervised contrastive objective
using normalized features, same-label positives and self-excluded softmax.
The reference notice is retained for attribution; this project does not claim
to introduce that objective. Contrastive temperature and CE weighting follow
the separately documented continuation experiments.

## OrganAMNIST / MedMNIST and source CT data

- Dataset creators: MedMNIST Team (Yang et al.).
- Dataset: <https://zenodo.org/records/10519652>
- Dataset license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- Generation notes: <https://github.com/MedMNIST/MedMNIST/blob/main/on_medmnist_plus.md>.
- Source benchmark: Bilic et al., [LiTS](https://doi.org/10.1016/j.media.2022.102680).
- Source organ annotations: Xu et al., *Efficient Multiple Organ Localization
  in CT Image Using 3D Region Proposal Network*, IEEE TMI 2019,
  <https://doi.org/10.1109/TMI.2019.2894854>.

The repository does not distribute the NPZ archives. Small example grids in
`docs/assets/reconstruction.png`, `embedding_pca.png` and `paired_examples.png`
are generated from the official dataset and measured model outputs. CT image
content retains **CC BY 4.0**, independently of the project code license.
Reconstruction figures add masking/model predictions; example grids add
labels, predictions and layout. Native input crops are not manually retouched.
The metadata JSON records public dataset facts and points to its official source.

MedMNIST's software package is separately Apache-2.0 licensed. Dataset images
must not be confused with the software package's license. Dependencies such as
PyTorch, NumPy, Pillow, Matplotlib, W&B and Playwright retain their own licenses.
