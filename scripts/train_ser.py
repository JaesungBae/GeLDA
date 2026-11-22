#!/usr/bin/env python
"""Stage 1: train the language-agnostic SER task adapter on Z^(0) features.

Zero-shot (trains on D^(\\kappa) U D_NEU^(\\kappa)):
    python scripts/train_ser.py --manifest-dir manifests/zero_shot_french --fold 1 \
        --features features/whisper-large-v3 --out runs/zero_shot/whisper-large-v3/french/fold1/stage1

Few-shot / Oracle (trains on the full multilingual set D):
    python scripts/train_ser.py --manifest-dir manifests/full --fold 1 \
        --features features/whisper-large-v3 --out runs/few_shot/whisper-large-v3/fold1/stage1

Training is the SpeechBrain recipe of the paper (``gelda/recipe.py``, ``configs/ser/adapter.yaml``).
Writes ``adapter.safetensors`` and ``test_metrics.json`` (overall and per language); the SpeechBrain
log and checkpoints are kept in the same folder.
"""

import argparse
import os

from gelda.adapter import save_adapter
from gelda.emobox import load_manifest
from gelda.recipe import DEFAULT_SAMPLER_SEED, default_hparams, train
from gelda.ser import LatentStore, dump_json, evaluate, format_metrics, load_latent_dict, load_latents


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--features", required=True, help="Z^(0) feature directory (<dataset>/<key>.npy)")
    p.add_argument("--out", required=True)
    p.add_argument("--hparams", default=default_hparams(), help="SpeechBrain hyperparameter file")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234, help="seed of the adapter initialisation")
    p.add_argument("--sampler-seed", type=int, default=DEFAULT_SAMPLER_SEED,
                   help="seed of the training sampler (default: SpeechBrain's default, as in the paper)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    store = LatentStore(args.features)
    manifests = {split: load_manifest(args.manifest_dir, args.fold, split) for split in ("train", "valid", "test")}
    print({split: len(m) for split, m in manifests.items()})

    model = train(args.hparams, args.out, manifests["train"], load_latent_dict(manifests["train"], store),
                  manifests["valid"], load_latent_dict(manifests["valid"], store),
                  layer=0, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
                  sampler_seed=args.sampler_seed, device=args.device)

    os.makedirs(args.out, exist_ok=True)
    save_adapter(model, os.path.join(args.out, "adapter.safetensors"))
    metrics = evaluate(model, *load_latents(manifests["test"], store), device=args.device)
    dump_json(metrics, os.path.join(args.out, "test_metrics.json"))
    print("test", format_metrics(metrics["overall"]))
    for lang, m in metrics["per_language"].items():
        print(f"  {lang:10s} {format_metrics(m)}")


if __name__ == "__main__":
    main()
