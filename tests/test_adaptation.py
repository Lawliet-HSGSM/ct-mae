import numpy as np
import torch
from torch.nn import functional as F

from ct_mae.adaptation_data import TwinViewDataset, class_weights, paired_batches
from ct_mae.adaptation_loss import supcon_loss, weighted_ce_sum


def test_class_weights_prioritize_weak_organs_and_preserve_mean():
    weights = class_weights([1., .8, .6], [10, 20, 30])
    assert weights[0] < weights[1] < weights[2]
    np.testing.assert_allclose(weights @ np.array([10, 20, 30]) / 60, 1)
    assert weights.max() / weights.min() <= 2
    np.testing.assert_array_equal(class_weights([1., 1.], [2, 5]), [1., 1.])


def test_pairing_preserves_every_image_and_class_count_without_cross_batch_pairs():
    labels = np.repeat(np.arange(3), [40, 32, 24])
    first, pairs = paired_batches(labels, 16, [(0, 1), (1, 2)], seed=42)
    second, _ = paired_batches(labels, 16, [(0, 1), (1, 2)], seed=42)
    assert first == second
    assert sorted(sum(first, [])) == list(range(len(labels)))
    assert all(len(batch) == 16 for batch in first)
    lookup = {i: b for b, batch in enumerate(first) for i in batch}
    assert all(lookup[a] == lookup[b] for a, b in pairs)
    changed, _ = paired_batches(labels, 16, [(0, 1), (1, 2)], seed=43)
    assert changed != first


def test_views_are_reproducible_distinct_and_do_not_change_global_rng():
    images = np.arange(2 * 16 * 16, dtype=np.uint16).reshape(2, 16, 16).astype(np.uint8)
    data = TwinViewDataset(images, np.array([0, 1]), np.array([1]), seed=42)
    state = torch.get_rng_state()
    one = data[0]
    assert torch.equal(torch.get_rng_state(), state)
    torch.testing.assert_close(one['views'], data[0]['views'], rtol=0, atol=0)
    assert one['views'].shape == (2, 1, 16, 16)
    assert not torch.equal(one['views'][0], one['views'][1])
    data.set_epoch(1)
    assert not torch.equal(one['views'], data[0]['views'])


def test_supcon_matches_soft_target_cross_entropy_and_gradient():
    torch.manual_seed(42)
    z = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    actual = supcon_loss(z, y, temperature=.1)
    normalized = F.normalize(z.float(), dim=-1)
    logits = normalized @ normalized.T / .1
    logits = logits.masked_fill(torch.eye(6, dtype=torch.bool), -torch.inf)
    targets = (y[:, None] == y[None]).float()
    targets.fill_diagonal_(0)
    targets /= targets.sum(1, keepdim=True)
    # Avoid 0 * -inf in the independent CE reference.
    reference = -(targets * F.log_softmax(logits, dim=1).masked_fill(
        torch.eye(6, dtype=torch.bool), 0)).sum(1).mean()
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(torch.autograd.grad(actual, z, retain_graph=True)[0],
                               torch.autograd.grad(reference, z)[0])


def test_weighted_ce_accumulation_uses_global_weight_denominator():
    torch.manual_seed(1)
    logits = torch.randn(10, 3, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 2])
    weights = torch.tensor([.5, 1., 2.])
    first = weighted_ce_sum(logits[:6], labels[:3], weights)
    second = weighted_ce_sum(logits[6:], labels[3:], weights)
    total = weighted_ce_sum(logits, labels, weights)
    torch.testing.assert_close(first[0] + second[0], total[0])
    torch.testing.assert_close(first[1] + second[1], total[1])
    per_view = F.cross_entropy(logits, labels.repeat_interleave(2), reduction='none')
    expected = (per_view * weights[labels].repeat_interleave(2)).sum()
    torch.testing.assert_close(total[0], expected)
