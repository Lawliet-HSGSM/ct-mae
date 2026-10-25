"""Training-only twin views and frequency-preserving confusion-pair batches."""

import hashlib

import numpy as np
import torch

from ct_mae.data import OrganDataset

PAIRS = ((6, 8), (4, 10), (4, 5))


def class_weights(f1, counts):
    deficit, counts = 1 - np.asarray(f1, dtype=float), np.asarray(counts, dtype=float)
    if (deficit.shape != counts.shape or np.any((deficit < 0) | (deficit > 1))
            or not np.isfinite(deficit).all() or np.any(counts < 0) or counts.sum() <= 0):
        raise ValueError('Invalid validation F1 or training class counts')
    raw = np.ones_like(deficit) if deficit.max() == 0 else 1 + deficit / deficit.max()
    return raw / (counts @ raw / counts.sum())


def paired_batches(labels, batch_size, priorities=PAIRS, *, seed):
    labels = np.asarray(labels)
    if batch_size % 2 or not len(labels) or len(labels) % batch_size:
        raise ValueError('Paired batches require an even batch size dividing the dataset')
    rng = np.random.default_rng(seed)
    available = {int(c): rng.permutation(np.flatnonzero(labels == c)).tolist()
                 for c in np.unique(labels)}
    units = []
    for a, b in priorities:
        left, right = available.get(a, []), available.get(b, [])
        count = min(len(left), len(right)) // 2
        units.extend(zip(left[:count], right[:count], strict=True))
        available[a], available[b] = left[count:], right[count:]
    rng.shuffle(units)
    singles = rng.permutation([i for values in available.values() for i in values]).tolist()
    total = len(labels) // batch_size
    quota = np.full(total, len(units) // total)
    quota[rng.permutation(total)[:len(units) % total]] += 1
    batches, paired_offset, single_offset = [], 0, 0
    for count in quota:
        batch = [int(i) for pair in units[paired_offset:paired_offset + count] for i in pair]
        need = batch_size - len(batch)
        batch.extend(int(i) for i in singles[single_offset:single_offset + need])
        rng.shuffle(batch)
        batches.append(batch)
        paired_offset += count
        single_offset += need
    return batches, [(int(a), int(b)) for a, b in units]


class TwinViewDataset(OrganDataset):
    def __init__(self, images, labels, indices, *, seed=42, **augmentation):
        super().__init__(images, labels, indices, augment=True, seed=seed, **augmentation)

    def __getitem__(self, position):
        index = int(self.indices[position])
        image = torch.from_numpy(self.images[index].copy()).unsqueeze(0).float().div_(255)
        views = []
        for view in range(2):
            key = f'adaptation-v1|{self.seed}|{self.epoch}|{index}|{view}'.encode()
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63 - 1)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                views.append(self.transform(image))
        return {'views': torch.stack(views), 'label': int(self.labels[index]), 'index': index}
