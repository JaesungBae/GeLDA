#!/usr/bin/env python
"""Map Z^(0) features into Z^(1) and Z^(2) with a Stage-1 task adapter.

Only needed to run GeLDA in Z^(1) or Z^(2).

    python scripts/extract_latents.py --adapter runs/.../stage1/adapter.safetensors \
        --manifest-dir manifests/zero_shot_french --fold 1 \
        --features features/whisper-large-v3 --out runs/.../stage1/latents

Writes ``<out>/z1/<dataset>/<key>.npy`` and ``<out>/z2/<dataset>/<key>.npy``.
"""

import argparse
import os

import numpy as np
import torch

from gelda.adapter import load_adapter
from gelda.emobox import SPLITS, load_manifest
from gelda.ser import LatentStore


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True)
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    adapter = load_adapter(args.adapter).to(args.device).eval()
    store = LatentStore(args.features)
    manifest = {}
    for split in SPLITS:
        manifest.update(load_manifest(args.manifest_dir, args.fold, split))

    keys = list(manifest)
    for start in range(0, len(keys), 4096):
        chunk = keys[start:start + 4096]
        z0 = torch.from_numpy(np.stack([store.load(manifest[k]["dataset"], k) for k in chunk]))
        z1, z2 = adapter.embed(z0.to(args.device))
        for name, z in (("z1", z1), ("z2", z2)):
            z = z.cpu().numpy()
            for i, key in enumerate(chunk):
                path = os.path.join(args.out, name, manifest[key]["dataset"], key + ".npy")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.save(path, z[i])
    print(f"wrote z1/z2 for {len(keys)} utterances to {args.out}")


if __name__ == "__main__":
    main()
