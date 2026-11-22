"""Sampling: Karras noise schedule, DPM-Solver++(2M) and classifier-free guidance.

Adapted from k-diffusion (https://github.com/crowsonkb/k-diffusion, MIT License).
"""

import torch


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros([1])]).to(device)


@torch.no_grad()
def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, eta=0.0, s_noise=1.0,
                        solver_type="heun", generator=None):
    """DPM-Solver++(2M) SDE (k-diffusion). With ``eta=0`` (the paper's setting)
    no noise is injected and the solver is deterministic given the initial noise."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_denoised, h_last, h = None, None, None
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h
            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised
            if old_denoised is not None:
                r = h_last / h
                if solver_type == "heun":
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                else:  # midpoint
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)
            if eta:
                noise = torch.randn(x.shape, generator=generator, device="cpu").to(x)
                x = x + noise * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise
        old_denoised = denoised
        h_last = h
    return x


@torch.no_grad()
def generate(denoiser, config, references, language, emotion, num_samples, batch_size=64,
             seed=0, device="cuda", cfg_scale=None, steps=None, eta=None):
    """Draw ``num_samples`` latents of ``emotion`` for ``language``.

    ``references`` is a :class:`~gelda.diffusion.data.ReferencePool`; every sample
    is conditioned on its own randomly drawn reference latent of ``language``.
    Returns a float32 numpy array [num_samples, latent_dim].
    """
    import random

    import numpy as np

    from ..constants import EMOTIONS

    m, s = config["model"], config["sampling"]
    cfg_scale = s["cfg_scale"] if cfg_scale is None else cfg_scale
    steps = s["steps"] if steps is None else steps
    eta = s["eta"] if eta is None else eta
    size = m["input_size"]
    generator = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    model_fn = cfg_denoiser(denoiser, null_class=m["num_classes"], scale=cfg_scale)
    sigmas = get_sigmas_karras(steps, m["sigma_min"], m["sigma_max"], rho=s["rho"], device=device)
    class_idx = EMOTIONS.index(emotion)

    out = []
    for start in range(0, num_samples, batch_size):
        b = min(batch_size, num_samples - start)
        x = torch.randn([b, 1, size, size], generator=generator).to(device) * m["sigma_max"]
        ref = torch.from_numpy(np.stack([references.sample(language, rng) for _ in range(b)])).to(device)
        cls = torch.full([b], class_idx, dtype=torch.long, device=device)
        x0 = sample_dpmpp_2m_sde(model_fn, x, sigmas, {"class_cond": cls, "ref_cond": ref},
                                 eta=eta, generator=generator)
        out.append(x0.reshape(b, -1)[:, :m["latent_dim"]].float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


def cfg_denoiser(denoiser, null_class: int, scale: float):
    """Classifier-free guidance over the joint (emotion, reference) condition.

    The unconditional branch uses the null emotion class and a zero reference,
    exactly as the conditions are dropped during training.
    """
    if scale == 1.0:
        return denoiser

    def fn(x, sigma, class_cond, ref_cond):
        x_in = torch.cat([x, x])
        sigma_in = torch.cat([sigma, sigma])
        class_in = torch.cat([torch.full_like(class_cond, null_class), class_cond])
        ref_in = torch.cat([torch.zeros_like(ref_cond), ref_cond])
        uncond, cond = denoiser(x_in, sigma_in, class_cond=class_in, ref_cond=ref_in).chunk(2)
        return uncond + (cond - uncond) * scale

    return fn
