#!/usr/bin/env python
"""Stage 2: train the conditional latent diffusion model of GeLDA.

    python scripts/train_diffusion.py --config configs/diffusion/whisper-large-v3_z0.json \
        --manifest-dir manifests/zero_shot_french --fold 1 \
        --latents features/whisper-large-v3 --out runs/zero_shot/whisper-large-v3/french/fold1/diffusion_z0

For Z^(1)/Z^(2), pass ``--latents runs/.../stage1/latents/z1`` (see extract_latents.py)
and the matching config. The model is trained on the train split of the
manifest; each sample is conditioned on its emotion and on the latent of a
randomly drawn utterance of the same language (the language reference).

Writes ``state.pt`` (resumable) every ``--save-every`` steps and
``diffusion.safetensors`` (EMA weights + config) at the end.
"""

import argparse
import copy
import functools
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gelda.diffusion.data import DiffusionDataset
from gelda.diffusion.model import (EMAWarmup, ema_update, load_config, make_denoiser,
                                   sample_sigmas_cosine_interpolated, save_inference_checkpoint)
from gelda.emobox import load_manifest
from gelda.ser import LatentStore


def seed_worker(worker_id, base_seed=0):
    # Reference latents are drawn with Python's `random` inside the workers.
    random.seed(base_seed * 1000 + worker_id)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--latents", required=True, help="latent directory (<dataset>/<key>.npy)")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=None, help="override training steps (default 400k)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--save-every", type=int, default=50000)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    config = load_config(args.config)
    t = config["training"]
    steps = args.steps or t["steps"]
    batch_size = args.batch_size or t["batch_size"]
    num_classes = config["model"]["num_classes"]  # index num_classes is the null class
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = load_manifest(args.manifest_dir, args.fold, "train")
    dataset = DiffusionDataset(manifest, LatentStore(args.latents), config["model"]["input_size"],
                               reference=t["reference"])
    print(f"{len(dataset)} training latents, reference languages: {dataset.references.languages()}")
    device = torch.device(args.device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                        pin_memory=device.type == "cuda",
                        worker_init_fn=functools.partial(seed_worker, base_seed=args.seed))

    model = make_denoiser(config).to(device)
    model_ema = copy.deepcopy(model)
    print(f"parameters: {sum(p.numel() for p in model.inner_model.parameters()):,}")
    opt = torch.optim.AdamW(model.inner_model.param_groups(t["lr"], t["mapping_lr_scale"]), lr=t["lr"],
                            betas=tuple(t["betas"]), eps=t["eps"], weight_decay=t["weight_decay"])
    ema_sched = EMAWarmup(power=t["ema_power"], max_value=t["ema_max"])
    step = 0

    state_path = os.path.join(args.out, "state.pt")
    if os.path.isfile(state_path):
        state = torch.load(state_path, map_location="cpu")
        model.inner_model.load_state_dict(state["model"])
        model_ema.inner_model.load_state_dict(state["model_ema"])
        opt.load_state_dict(state["opt"])
        ema_sched.load_state_dict(state["ema_sched"])
        step = state["step"]
        print(f"resumed from step {step}")

    def save_state():
        torch.save({"model": model.inner_model.state_dict(), "model_ema": model_ema.inner_model.state_dict(),
                    "opt": opt.state_dict(), "ema_sched": ema_sched.state_dict(), "step": step,
                    "config": config}, state_path)

    p_drop = t["cond_dropout_rate"]
    progress = tqdm(total=steps, initial=step, smoothing=0.1)
    while step < steps:
        for x, class_cond, ref in loader:
            x, class_cond, ref = x.to(device), class_cond.to(device), ref.to(device)
            # Classifier-free guidance: drop each condition independently.
            class_cond = class_cond.masked_fill(torch.rand(class_cond.shape, device=device) < p_drop, num_classes)
            ref = ref.masked_fill((torch.rand(ref.shape[0], device=device) < p_drop)[:, None], 0.0)

            noise = torch.randn_like(x)
            sigma = sample_sigmas_cosine_interpolated(x.shape[0], config, device=device)
            loss = model.loss(x, noise, sigma, class_cond=class_cond, ref_cond=ref).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            ema_update(model, model_ema, ema_sched.get_value())
            ema_sched.step()

            step += 1
            progress.update(1)
            if step % 25 == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}")
            if step % args.save_every == 0:
                save_state()
            if step >= steps:
                break
    progress.close()
    save_state()
    save_inference_checkpoint(os.path.join(args.out, "diffusion.safetensors"), model_ema.inner_model, config)
    print(f"saved {os.path.join(args.out, 'diffusion.safetensors')}")


if __name__ == "__main__":
    main()
