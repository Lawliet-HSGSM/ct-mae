import torch

from ct_mae.config import load_config
from ct_mae.models import MaskedAutoencoder, OrganClassifier, patchify, random_masking, unpatchify

torch.set_num_threads(2)


def small_config():
    config = load_config()["model"]
    config.update(image_size=16, patch_size=4, encoder_dim=32, encoder_depth=2,
                  encoder_heads=4, decoder_dim=16, decoder_depth=1, decoder_heads=4)
    return config


def test_grayscale_patch_order_and_roundtrip_preserve_every_pixel():
    image = torch.arange(16).reshape(1, 1, 4, 4).float()
    patches = patchify(image, 2)
    torch.testing.assert_close(patches[0, 0], torch.tensor([0., 1., 4., 5.]))
    torch.testing.assert_close(unpatchify(patches, 2, 1), image)


def test_random_mask_keeps_exact_budget_and_restore_indices_recover_spatial_positions():
    patches = torch.arange(256).reshape(1, 256, 1).float()
    visible, mask, restore = random_masking(patches, 0.75)
    assert visible.shape == (1, 64, 1)
    assert mask.sum() == 192
    sequence = torch.cat([visible, torch.full((1, 192, 1), -1.)], dim=1)
    restored = torch.gather(sequence, 1, restore.unsqueeze(-1))
    torch.testing.assert_close(restored[mask == 0], patches[mask == 0])
    assert (restored[mask == 1] == -1).all()


def test_mae_shapes_gradients_and_loss_ignore_visible_patch_errors():
    model = MaskedAutoencoder(small_config())
    images = torch.rand(2, 1, 16, 16)
    loss, predicted, mask = model(images)
    assert predicted.shape == (2, 16, 16)
    assert mask.sum(dim=1).tolist() == [12, 12]
    altered = predicted.detach().clone()
    altered[mask == 0] += 100
    torch.testing.assert_close(model.reconstruction_loss(images, altered, mask), loss.detach())
    loss.backward()
    assert model.encoder.patch_embed.weight.grad.abs().sum() > 0
    assert model.decoder_pred.weight.grad.abs().sum() > 0


def test_linear_probe_stays_frozen_and_eval_after_parent_train():
    model = OrganClassifier(small_config(), num_classes=11, frozen=True)
    model.train()
    assert not model.encoder.training
    before = {key: value.clone() for key, value in model.encoder.state_dict().items()}
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.01)
    logits = model(torch.rand(3, 1, 16, 16))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2]))
    loss.backward()
    optimizer.step()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    for key, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_full_classifier_updates_encoder_and_returns_eleven_logits():
    model = OrganClassifier(small_config(), num_classes=11)
    before = model.encoder.patch_embed.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    logits = model(torch.rand(3, 1, 16, 16))
    assert logits.shape == (3, 11)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
    optimizer.step()
    assert not torch.equal(before, model.encoder.patch_embed.weight)
