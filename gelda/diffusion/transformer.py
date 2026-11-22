"""Hourglass diffusion transformer used by GeLDA.

Adapted from ``image_transformer_v2`` of k-diffusion
(https://github.com/crowsonkb/k-diffusion, MIT License, Copyright (c) 2022
Katherine Crowson). Only the global-attention variant used in the paper is kept.

A latent vector is zero-padded and reshaped into a single-channel S x S
"image", split into 2 x 2 patches, and denoised by a two-level hourglass
transformer. The noise level, the emotion class and the language reference
latent are each embedded, summed and passed through a mapping network whose
output conditions every block through adaptive RMS normalisation.
"""

import math
from dataclasses import dataclass
from functools import reduce
from typing import Sequence

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


# ----------------------------------------------------------------- param tags
# Used to build optimizer parameter groups (weight decay / mapping-network LR).

def tag_param(param, tag):
    if not hasattr(param, "_tags"):
        param._tags = set([tag])
    else:
        param._tags.add(tag)
    return param


def tag_module(module, tag):
    for param in module.parameters():
        tag_param(param, tag)
    return module


def apply_wd(module):
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            tag_param(param, "wd")
    return module


def filter_params(function, module):
    for param in module.parameters():
        if function(getattr(param, "_tags", set())):
            yield param


def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


# -------------------------------------------------------------------- kernels

def linear_geglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)


def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


def apply_rotary_emb(x, theta):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d:d * 2], x[..., d * 2:]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1.to(out_dtype), y2.to(out_dtype), x3), dim=-1)


def make_axial_pos(h, w, dtype=None, device=None):
    """Patch-centre coordinates in [-1, 1]^2, shape [h * w, 2]."""
    def centers(n):
        edges = torch.linspace(-1.0, 1.0, n + 1, dtype=dtype, device=device)
        return (edges[:-1] + edges[1:]) / 2

    grid = torch.stack(torch.meshgrid(centers(h), centers(w), indexing="ij"), dim=-1)
    return grid.view(h * w, 2)


def downscale_pos(pos):
    pos = rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=2, nw=2)
    return torch.mean(pos, dim=-2)


# --------------------------------------------------------------------- layers

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer("weight", torch.randn([out_features // 2, in_features]) * std)

    def forward(self, x):
        f = 2 * math.pi * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class LinearGEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        return linear_geglu(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = apply_wd(zero_init(nn.Linear(cond_features, features, bias=False)))
        tag_module(self.linear, "mapping")

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond)[:, None, None, :] + 1, self.eps)


class AxialRoPE(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        log_min, log_max = math.log(math.pi), math.log(10.0 * math.pi)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 4 + 1)[:-1].exp()
        self.register_buffer("freqs", freqs.view(dim // 4, n_heads).T.contiguous())

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs.to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs.to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(nn.Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(nn.Linear(d_model, d_model, bias=False)))

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        pos = rearrange(pos, "... h w e -> ... (h w) e").to(qkv.dtype)
        theta = self.pos_emb(pos).movedim(-2, -3)
        q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh (h w) e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
        q, k = apply_rotary_emb(q, theta), apply_rotary_emb(k, theta)
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh (h w) e -> n h w (nh e)", h=skip.shape[-3], w=skip.shape[-2])
        x = self.dropout(x)
        return self.out_proj(x) + skip


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, cond_features, dropout=0.0):
        super().__init__()
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(nn.Linear(d_ff, d_model, bias=False)))

    def forward(self, x, cond):
        skip = x
        x = self.dropout(self.up_proj(self.norm(x, cond)))
        return self.down_proj(x) + skip


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, d_head, cond_features, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        return self.ff(self.self_attn(x, pos, cond), cond)


class Level(nn.ModuleList):
    def forward(self, x, *args):
        for layer in self:
            x = layer(x, *args)
        return x


class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(nn.Linear(d_ff, d_model, bias=False)))

    def forward(self, x):
        skip = x
        x = self.dropout(self.up_proj(self.norm(x)))
        return self.down_proj(x) + skip


class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(d_model, d_ff, dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h, self.w = patch_size
        self.proj = apply_wd(nn.Linear(in_features * self.h * self.w, out_features, bias=False))

    def forward(self, x):
        x = rearrange(x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=self.h, nw=self.w)
        return self.proj(x)


class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h, self.w = patch_size
        self.proj = apply_wd(nn.Linear(in_features, out_features * self.h * self.w, bias=False))

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)


class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h, self.w = patch_size
        self.proj = apply_wd(nn.Linear(in_features, out_features * self.h * self.w, bias=False))
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip):
        x = self.proj(x)
        x = rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)
        return torch.lerp(skip, x, self.fac.to(x.dtype))


# ---------------------------------------------------------------------- model

@dataclass
class LevelSpec:
    depth: int
    width: int
    d_ff: int
    d_head: int
    dropout: float


@dataclass
class MappingSpec:
    depth: int
    width: int
    d_ff: int
    dropout: float


class LatentDiffusionTransformer(nn.Module):
    """Conditional denoiser D(x, sigma | emotion, language reference)."""

    def __init__(self, levels: Sequence[LevelSpec], mapping: MappingSpec, num_classes: int,
                 ref_dim: int, in_channels: int = 1, patch_size=(2, 2)):
        super().__init__()
        self.num_classes = num_classes  # includes the null (unconditional) class
        self.patch_in = TokenMerge(in_channels, levels[0].width, patch_size)

        self.time_emb = FourierFeatures(1, mapping.width)
        self.time_in_proj = nn.Linear(mapping.width, mapping.width, bias=False)
        self.class_emb = nn.Embedding(num_classes, mapping.width)
        self.ref_in_proj = nn.Linear(ref_dim, mapping.width, bias=False)
        self.mapping = tag_module(MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, mapping.dropout),
                                  "mapping")

        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        for i, spec in enumerate(levels):
            def make():
                return TransformerLayer(spec.width, spec.d_ff, spec.d_head, mapping.width, dropout=spec.dropout)

            if i < len(levels) - 1:
                self.down_levels.append(Level([make() for _ in range(spec.depth)]))
                self.up_levels.append(Level([make() for _ in range(spec.depth)]))
            else:
                self.mid_level = Level([make() for _ in range(spec.depth)])

        self.merges = nn.ModuleList([TokenMerge(a.width, b.width) for a, b in zip(levels[:-1], levels[1:])])
        self.splits = nn.ModuleList([TokenSplit(b.width, a.width) for a, b in zip(levels[:-1], levels[1:])])

        self.out_norm = RMSNorm(levels[0].width)
        self.patch_out = TokenSplitWithoutSkip(levels[0].width, in_channels, patch_size)
        nn.init.zeros_(self.patch_out.proj.weight)

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" not in tags, self)
        mapping_wd = filter_params(lambda tags: "wd" in tags and "mapping" in tags, self)
        mapping_no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" in tags, self)
        return [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {"params": list(mapping_no_wd), "lr": base_lr * mapping_lr_scale, "weight_decay": 0.0},
        ]

    def forward(self, x, sigma, class_cond, ref_cond):
        """x: [B, 1, S, S]; sigma: [B]; class_cond: [B] (long); ref_cond: [B, ref_dim]."""
        x = self.patch_in(x.movedim(-3, -1))
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(x.shape[-3], x.shape[-2], 2)

        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        cond = self.mapping(time_emb + self.class_emb(class_cond) + self.ref_in_proj(ref_cond))

        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, cond)
            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos)

        x = self.mid_level(x, pos, cond)

        for up_level, split, skip, pos in reversed(list(zip(self.up_levels, self.splits, skips, poses))):
            x = split(x, skip)
            x = up_level(x, pos, cond)

        x = self.patch_out(self.out_norm(x))
        return x.movedim(-1, -3)
