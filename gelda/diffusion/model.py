"""Karras et al. (2022) preconditioning, noise-level sampling, EMA and configs.

Adapted from k-diffusion (https://github.com/crowsonkb/k-diffusion, MIT License).
"""

import copy
import json
import math

import torch
from torch import nn

from ..constants import EMOTIONS
from .transformer import LatentDiffusionTransformer, LevelSpec, MappingSpec

# Architecture and optimisation settings shared by all GeLDA diffusion models.
DEFAULT_CONFIG = {
    "model": {
        "input_size": None,        # side S of the S x S matrix the latent is reshaped into
        "latent_dim": None,        # dimension of the latent vectors
        "ref_dim": None,           # dimension of the language-reference latent
        "num_classes": len(EMOTIONS),
        "patch_size": [2, 2],
        "depths": [2, 4],
        "widths": [256, 512],
        "d_head": 64,
        "d_ff_mult": 3,
        "dropout_rate": 0.05,
        "mapping_depth": 2,
        "mapping_width": 256,
        "mapping_d_ff": 768,
        "mapping_dropout_rate": 0.0,
        "sigma_data": None,
        "sigma_min": None,
        "sigma_max": None,
        "loss_weighting": "soft-min-snr",
    },
    "training": {
        "batch_size": 64,
        "steps": 400000,
        "lr": 5e-4,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 1e-4,
        "mapping_lr_scale": 1 / 3,
        "grad_clip": 1.0,
        "cond_dropout_rate": 0.1,
        "ema_power": 0.6667,
        "ema_max": 0.9999,
        "reference": "same_language",
    },
    "sampling": {
        "steps": 20,
        "cfg_scale": 1.2,
        "eta": 0.0,
        "rho": 7.0,
    },
}


def merge_config(config: dict) -> dict:
    out = copy.deepcopy(DEFAULT_CONFIG)
    for section, values in config.items():
        out.setdefault(section, {}).update(values)
    return out


def load_config(path: str) -> dict:
    with open(path) as f:
        return merge_config(json.load(f))


def make_model(config: dict) -> LatentDiffusionTransformer:
    m = config["model"]
    levels = [LevelSpec(depth, width, width * m["d_ff_mult"], m["d_head"], m["dropout_rate"])
              for depth, width in zip(m["depths"], m["widths"])]
    mapping = MappingSpec(m["mapping_depth"], m["mapping_width"], m["mapping_d_ff"], m["mapping_dropout_rate"])
    # One extra class embedding acts as the "no emotion" token for classifier-free guidance.
    return LatentDiffusionTransformer(levels, mapping, num_classes=m["num_classes"] + 1,
                                      ref_dim=m["ref_dim"], patch_size=tuple(m["patch_size"]))


def append_dims(x, target_dims):
    return x[(...,) + (None,) * (target_dims - x.ndim)]


class Denoiser(nn.Module):
    """Karras et al. preconditioner with soft-min-SNR loss weighting."""

    def __init__(self, inner_model, sigma_data=1.0, weighting="soft-min-snr"):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data
        if weighting == "soft-min-snr":
            self.weighting = lambda s: (s * sigma_data) ** 2 / (s ** 2 + sigma_data ** 2) ** 2
        elif weighting == "karras":
            self.weighting = torch.ones_like
        else:
            raise ValueError(f"Unknown weighting {weighting}")

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, x, noise, sigma, **cond):
        c_skip, c_out, c_in = [append_dims(s, x.ndim) for s in self.get_scalings(sigma)]
        noised = x + noise * append_dims(sigma, x.ndim)
        out = self.inner_model(noised * c_in, sigma, **cond)
        target = (x - c_skip * noised) / c_out
        return ((out - target) ** 2).flatten(1).mean(1) * self.weighting(sigma)

    def forward(self, x, sigma, **cond):
        c_skip, c_out, c_in = [append_dims(s, x.ndim) for s in self.get_scalings(sigma)]
        return self.inner_model(x * c_in, sigma, **cond) * c_out + x * c_skip


def make_denoiser(config: dict, inner_model=None) -> Denoiser:
    m = config["model"]
    return Denoiser(inner_model if inner_model is not None else make_model(config),
                    sigma_data=m["sigma_data"], weighting=m["loss_weighting"])


def sample_sigmas_cosine_interpolated(n, config, device="cpu", stratified=True):
    """Training noise levels from the interpolated cosine schedule (simple diffusion).

    With ``stratified=True`` the uniform variates are stratified over the batch.
    """
    m = config["model"]
    sigma_data = m["sigma_data"]
    image_d = noise_d_high = m["input_size"]
    noise_d_low = 32
    min_value = min(m["sigma_min"], 1e-3)
    max_value = max(m["sigma_max"], 1e3)

    def logsnr_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_shifted(t, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = torch.rand(n, device=device)
    if stratified:
        u = (torch.arange(n, device=device) + u) / n
    logsnr = torch.lerp(logsnr_shifted(u, noise_d_low, logsnr_min, logsnr_max),
                        logsnr_shifted(u, noise_d_high, logsnr_min, logsnr_max), u)
    return torch.exp(-logsnr / 2) * sigma_data


class EMAWarmup:
    """EMA decay 1 - (1 + step) ** -power, capped at ``max_value``."""

    def __init__(self, power=0.6667, max_value=0.9999):
        self.power, self.max_value, self.step_count = power, max_value, 0

    def get_value(self):
        value = 1 - (1 + self.step_count) ** -self.power
        return min(self.max_value, max(0.0, value))

    def step(self):
        self.step_count += 1

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, state):
        self.__dict__.update(state)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    params, avg_params = dict(model.named_parameters()), dict(averaged_model.named_parameters())
    for name, param in params.items():
        avg_params[name].lerp_(param, 1 - decay)
    avg_buffers = dict(averaged_model.named_buffers())
    for name, buf in model.named_buffers():
        avg_buffers[name].copy_(buf)


def save_inference_checkpoint(path: str, inner_model: nn.Module, config: dict) -> None:
    """EMA weights + config in a single safetensors file."""
    from safetensors.torch import save_file

    state = {k: v.detach().cpu().contiguous() for k, v in inner_model.state_dict().items()}
    save_file(state, path, metadata={"config": json.dumps(config)})


def load_inference_checkpoint(path: str, device="cpu"):
    """Returns (denoiser, config) from a safetensors checkpoint."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(path, framework="pt") as f:
        config = merge_config(json.loads(f.metadata()["config"]))
    denoiser = make_denoiser(config)
    denoiser.inner_model.load_state_dict(load_file(path, device=str(device)))
    return denoiser.to(device).eval(), config
