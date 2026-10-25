import copy

import torch
from torch import nn

from ct_mae.adaptation_engine import train_adaptation_epoch


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 3)

    def adaptation_forward(self, inputs):
        return self.head(inputs), None


def test_adaptation_accumulation_matches_full_weighted_group():
    torch.manual_seed(2)
    large = Tiny()
    micro = copy.deepcopy(large)
    images = torch.randn(5, 2, 3)
    labels = torch.tensor([0, 0, 1, 2, 2])
    weights = torch.tensor([.5, 1., 2.])
    data = {'views': images, 'label': labels}
    recipe = dict(lr=.03, min_lr=1e-6, epochs=2, warmup_epochs=0, accumulation=1)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    train_adaptation_epoch(large, [data], torch.optim.SGD(large.parameters(), lr=.03),
                           scaler, recipe, 0, weights, device=torch.device('cpu'), amp=False)
    recipe['accumulation'] = 2
    batches = [{key: value[:3] for key, value in data.items()},
               {key: value[3:] for key, value in data.items()}]
    metrics = train_adaptation_epoch(
        micro, batches, torch.optim.SGD(micro.parameters(), lr=.03), scaler,
        recipe, 0, weights, device=torch.device('cpu'), amp=False,
    )
    for first, second in zip(large.parameters(), micro.parameters(), strict=True):
        torch.testing.assert_close(first, second, atol=1e-7, rtol=1e-6)
    assert metrics['optimizer_steps'] == 1
    assert metrics['samples'] == 5 and metrics['views'] == 10
