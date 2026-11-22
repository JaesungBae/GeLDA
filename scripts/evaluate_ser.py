#!/usr/bin/env python
"""Evaluate a task adapter on a test split (e.g. a released checkpoint).

    python scripts/evaluate_ser.py --adapter checkpoints/few_shot/whisper-large-v3/fold1/ser_adapter.safetensors \
        --manifest-dir manifests/full --fold 1 --features features/whisper-large-v3 --language french
"""

import argparse

from gelda.adapter import load_adapter
from gelda.emobox import load_manifest
from gelda.ser import LatentStore, dump_json, evaluate, filter_language, format_metrics, load_latents


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True)
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--features", required=True, help="latent directory of Z^(layer)")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--split", default="test")
    p.add_argument("--language", default=None, help="restrict to one language")
    p.add_argument("--out", default=None, help="optional JSON output path")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    manifest = load_manifest(args.manifest_dir, args.fold, args.split)
    if args.language:
        manifest = filter_language(manifest, args.language)
    metrics = evaluate(load_adapter(args.adapter), *load_latents(manifest, LatentStore(args.features)),
                       layer=args.layer, device=args.device)
    print("overall", format_metrics(metrics["overall"]))
    for lang, m in metrics["per_language"].items():
        print(f"  {lang:10s} {format_metrics(m)}")
    if args.out:
        dump_json(metrics, args.out)


if __name__ == "__main__":
    main()
