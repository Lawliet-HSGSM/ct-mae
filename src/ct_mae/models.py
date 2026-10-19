"""A grayscale asymmetric MAE and its shared ViT classification encoder.

Architecture/initialization: https://github.com/facebookresearch/mae/blob/main/models_mae.py
SDPA: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, channels, height, width = images.shape
    if height != width or height % patch_size:
        raise ValueError("Images must be square and divisible by patch size")
    grid = height // patch_size
    return (images.reshape(batch, channels, grid, patch_size, grid, patch_size)
            .permute(0, 2, 4, 3, 5, 1).reshape(batch, grid * grid, patch_size**2 * channels))


def unpatchify(patches: torch.Tensor, patch_size: int, channels: int) -> torch.Tensor:
    batch, length, _ = patches.shape
    grid = math.isqrt(length)
    if grid * grid != length or patches.shape[-1] != patch_size**2 * channels:
        raise ValueError("Invalid patch grid or channel count")
    images = patches.reshape(batch, grid, grid, patch_size, patch_size, channels)
    return images.permute(0, 5, 1, 3, 2, 4).reshape(
        batch, channels, grid * patch_size, grid * patch_size,
    )


def random_masking(x: torch.Tensor, ratio: float, noise: torch.Tensor | None = None):
    batch, length, dimension = x.shape
    keep = int(length * (1 - ratio))
    if not 0 < keep < length:
        raise ValueError("Masking requires at least one visible and one missing patch")
    noise = torch.rand(batch, length, device=x.device) if noise is None else noise.to(x.device)
    if noise.shape != (batch, length):
        raise ValueError("One independent noise value is required per image patch")
    shuffle = noise.argsort(dim=1)
    restore = shuffle.argsort(dim=1)
    visible = x.gather(1, shuffle[:, :keep, None].expand(-1, -1, dimension))
    mask = torch.ones(batch, length, device=x.device)
    mask[:, :keep] = 0
    return visible, mask.gather(1, restore), restore


def sincos_positions(grid: int, dimension: int) -> torch.Tensor:
    if dimension % 4:
        raise ValueError("2D sine/cosine positions require a dimension divisible by four")
    y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    frequency = 1 / 10000 ** (torch.arange(dimension // 4, dtype=torch.float64) / (dimension // 4))
    axes = [axis.flatten()[:, None] * frequency[None] for axis in (x, y)]
    table = torch.cat([function(axis) for axis in axes for function in (torch.sin, torch.cos)], -1)
    return torch.cat([torch.zeros(1, dimension), table.float()], 0).unsqueeze(0)


class DropPath(nn.Module):
    def __init__(self, probability: float):
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.probability:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Attention(nn.Module):
    def __init__(self, dimension: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dimension, dimension * 3, bias=True)
        self.projection = nn.Linear(dimension, dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dimension = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, dimension // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        return self.projection(output.transpose(1, 2).reshape(batch, length, dimension))


class Block(nn.Module):
    def __init__(self, dimension: int, heads: int, ratio: float, eps: float, drop_path: float = 0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension, eps=eps)
        self.attention = Attention(dimension, heads)
        self.norm2 = nn.LayerNorm(dimension, eps=eps)
        self.mlp = nn.Sequential(nn.Linear(dimension, int(dimension * ratio)), nn.GELU(),
                                 nn.Linear(int(dimension * ratio), dimension))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


def initialize_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ViTEncoder(nn.Module):
    def __init__(self, config: dict, drop_path: float = 0):
        super().__init__()
        self.image_size, self.patch_size = config["image_size"], config["patch_size"]
        self.in_channels, dimension = config["in_channels"], config["encoder_dim"]
        self.patch_embed = nn.Conv2d(self.in_channels, dimension, self.patch_size, self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.positions = nn.Parameter(
            sincos_positions(self.image_size // self.patch_size, dimension), requires_grad=False,
        )
        self.blocks = nn.ModuleList([
            Block(dimension, config["encoder_heads"], config["mlp_ratio"], config["layer_norm_eps"],
                  rate) for rate in torch.linspace(0, drop_path, config["encoder_depth"]).tolist()
        ])
        self.norm = nn.LayerNorm(dimension, eps=config["layer_norm_eps"])
        self.apply(initialize_module)
        nn.init.xavier_uniform_(self.patch_embed.weight.view(dimension, -1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, images: torch.Tensor, mask_ratio: float = 0,
                noise: torch.Tensor | None = None):
        if images.shape[1:] != (self.in_channels, self.image_size, self.image_size):
            raise ValueError("Encoder received an unexpected image shape")
        x = self.patch_embed(images).flatten(2).transpose(1, 2) + self.positions[:, 1:]
        if mask_ratio:
            x, mask, restore = random_masking(x, mask_ratio, noise)
        else:
            mask = x.new_zeros(x.shape[:2])
            restore = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
        cls = (self.cls_token + self.positions[:, :1]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x), mask, restore


class MaskedAutoencoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.encoder = ViTEncoder(config)
        dimension = config["decoder_dim"]
        self.decoder_embed = nn.Linear(config["encoder_dim"], dimension)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.decoder_positions = nn.Parameter(
            sincos_positions(config["image_size"] // config["patch_size"], dimension),
            requires_grad=False,
        )
        self.decoder_blocks = nn.ModuleList([
            Block(dimension, config["decoder_heads"], config["mlp_ratio"], config["layer_norm_eps"])
            for _ in range(config["decoder_depth"])
        ])
        self.decoder_norm = nn.LayerNorm(dimension, eps=config["layer_norm_eps"])
        self.decoder_pred = nn.Linear(dimension, config["patch_size"]**2 * config["in_channels"])
        for name, module in self.named_children():
            if name != "encoder":
                module.apply(initialize_module)
        nn.init.normal_(self.mask_token, std=0.02)

    def reconstruction_loss(self, images: torch.Tensor, predicted: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        target = patchify(images, self.config["patch_size"])
        if self.config["norm_pix_loss"]:
            target = (target - target.mean(-1, keepdim=True)) / (
                target.var(-1, keepdim=True) + 1e-6
            ).sqrt()
        loss = (predicted - target).square().mean(-1)
        return (loss * mask).sum() / mask.sum()

    def forward(self, images: torch.Tensor, noise: torch.Tensor | None = None):
        encoded, mask, restore = self.encoder(images, self.config["mask_ratio"], noise)
        projected = self.decoder_embed(encoded)
        missing = self.mask_token.expand(
            images.shape[0], restore.shape[1] + 1 - encoded.shape[1], -1,
        )
        sequence = torch.cat([projected[:, 1:], missing], dim=1)
        sequence = sequence.gather(1, restore[:, :, None].expand(-1, -1, projected.shape[-1]))
        x = torch.cat([projected[:, :1], sequence], dim=1) + self.decoder_positions
        for block in self.decoder_blocks:
            x = block(x)
        predicted = self.decoder_pred(self.decoder_norm(x))[:, 1:]
        return self.reconstruction_loss(images, predicted, mask), predicted, mask


class OrganClassifier(nn.Module):
    def __init__(self, config: dict, num_classes: int, *, frozen: bool = False, seed: int = 42):
        super().__init__()
        self.encoder = ViTEncoder(config, drop_path=config["classifier_drop_path"])
        self.head = nn.Linear(config["encoder_dim"], num_classes)
        # Identical initial heads for all three downstream experiments.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)
        self.frozen = frozen
        if frozen:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():
                tokens, _, _ = self.encoder(images)
        else:
            tokens, _, _ = self.encoder(images)
        return self.head(tokens[:, 1:].mean(dim=1))
