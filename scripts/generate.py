#!/usr/bin/env python
"""Stage 2 (sampling): synthesise target-language latents with a GeLDA diffusion model.

    python scripts/generate.py --checkpoint .../diffusion_z0.safetensors \
        --manifest-dir manifests/zero_shot_french --fold 1 --latents features/whisper-large-v3 \
        --language french --num-samples 1000 --out runs/.../generated_z0

Each sample is conditioned on an emotion and on the latent of a randomly drawn
training utterance of the target language (for zero-shot SER these are all
NEUTRAL). ``--latents`` must hold latents of the same space as the model
(Z^(0) features, or the z1/z2 directories written by extract_latents.py).

By default one file per emotion of the target language's label inventory
(the classes present in its test split) is written: ``<out>/<Emotion>.npy``
with shape [num_samples, latent_dim].
"""

import argparse
import json
import os

import numpy as np
import torch

from gelda.constants import EMOTIONS
from gelda.diffusion import ReferencePool, generate, load_inference_checkpoint
from gelda.emobox import load_manifest
from gelda.ser import LatentStore, filter_language


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--latents", required=True)
    p.add_argument("--language", required=True)
    p.add_argument("--num-samples", type=int, default=1000, help="samples per emotion")
    p.add_argument("--emotions", nargs="*", default=None, choices=EMOTIONS)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--cfg-scale", type=float, default=None, help="default: 1.2")
    p.add_argument("--steps", type=int, default=None, help="default: 20")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    denoiser, config = load_inference_checkpoint(args.checkpoint, device=args.device)
    train = filter_language(load_manifest(args.manifest_dir, args.fold, "train"), args.language)
    references = ReferencePool(train, LatentStore(args.latents), config["training"]["reference"])
    emotions = args.emotions or sorted(
        {v["emo"] for v in filter_language(load_manifest(args.manifest_dir, args.fold, "test"), args.language).values()},
        key=EMOTIONS.index)
    n_refs = len(references.index.get(args.language, []))
    print(f"{args.language}: {n_refs} reference utterances; emotions: {', '.join(emotions)}")

    os.makedirs(args.out, exist_ok=True)
    for i, emotion in enumerate(emotions):
        z = generate(denoiser, config, references, args.language, emotion, args.num_samples,
                     batch_size=args.batch_size, seed=args.seed * 100 + i, device=args.device,
                     cfg_scale=args.cfg_scale, steps=args.steps)
        np.save(os.path.join(args.out, f"{emotion}.npy"), z)
        print(f"  {emotion}: {z.shape}")
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"checkpoint": os.path.abspath(args.checkpoint), "language": args.language,
                   "fold": args.fold, "num_samples": args.num_samples, "seed": args.seed,
                   "cfg_scale": args.cfg_scale or config["sampling"]["cfg_scale"],
                   "steps": args.steps or config["sampling"]["steps"], "emotions": list(emotions)}, f, indent=2)


if __name__ == "__main__":
    main()
