"""Supervised contrastive adaptation; classification does not use the projector."""

import torch
from torch import nn
from torch.nn import functional as F

from ct_mae.models import OrganClassifier


def supcon_loss(features, labels, temperature=.1):
    z = F.normalize(features.float(), dim=-1)
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positives = (labels[:, None] == labels[None]) & ~diagonal
    count = positives.sum(1)
    if (count == 0).any():
        raise ValueError('Every contrastive anchor needs another positive view or sample')
    logits = (z @ z.T / temperature).masked_fill(diagonal, -torch.inf)
    log_probability = F.log_softmax(logits, dim=1).masked_fill(diagonal, 0)
    return -(log_probability * positives).sum(1).div(count).mean()


def weighted_ce_sum(logits, labels, weights):
    targets = labels.repeat_interleave(2)
    assigned = weights[targets]
    losses = F.cross_entropy(logits.float(), targets, reduction='none')
    return (losses * assigned).sum(), assigned.sum()


class AdaptationClassifier(OrganClassifier):
    def __init__(self, config, *, partial, contrastive, seed=42):
        super().__init__(config, 11, trainable_blocks=3 if partial else None, seed=seed)
        self.projector = None
        if contrastive:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 1907)
                dimension = config['encoder_dim']
                self.projector = nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(),
                                               nn.Linear(dimension, 128))

    def adaptation_forward(self, images):
        features = self.forward_features(images)
        projected = self.projector(features) if self.projector is not None else None
        return self.head(features), projected
