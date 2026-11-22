#!/usr/bin/env python
"""Stage 3: specialise the Stage-1 adapter to the target language with GeLDA latents.

Zero-shot (target-language NEUTRAL data + 200 generated latents per emotion):
    python scripts/finetune_ser.py --setting zero_shot --language french --fold 1 --layer 0 \
        --manifest-dir manifests/zero_shot_french --latents features/whisper-large-v3 \
        --adapter .../stage1/adapter.safetensors --generated .../generated_z0 \
        --repetition 0 --out runs/.../stage3_z0_rep0

Few-shot (all labelled target-language data + 500 generated latents per observed emotion):
    python scripts/finetune_ser.py --setting few_shot --language french --fold 1 --layer 0 \
        --manifest-dir manifests/full ...

Training is the SpeechBrain recipe of the paper (``gelda/recipe.py``): it continues from the
Stage-1 adapter for 100 more epochs (101-200) with a new optimizer and learning-rate schedule;
layers producing Z^(<=layer) are frozen. Repetition r uses generated samples [r * n, (r + 1) * n)
of every emotion, so repetitions use disjoint synthetic sets. ``--num-aug 0`` gives the
un-augmented fine-tuning baseline.
"""

import argparse
import os

import numpy as np

from gelda.adapter import load_adapter, save_adapter
from gelda.constants import EMOTIONS
from gelda.emobox import load_manifest
from gelda.recipe import DEFAULT_SAMPLER_SEED, default_hparams, train
from gelda.ser import LatentStore, dump_json, evaluate, filter_language, format_metrics, load_latent_dict, load_latents

DEFAULT_NUM_AUG = {"zero_shot": 200, "few_shot": 500}
REPETITION_SEEDS = (1234, 2345, 3456, 4567, 5678)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setting", required=True, choices=sorted(DEFAULT_NUM_AUG))
    p.add_argument("--language", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--layer", type=int, default=0, choices=(0, 1, 2))
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--latents", required=True, help="latents of Z^(layer) for the real utterances")
    p.add_argument("--adapter", required=True, help="Stage-1 adapter")
    p.add_argument("--generated", default=None, help="directory with <Emotion>.npy from generate.py")
    p.add_argument("--num-aug", type=int, default=None, help="generated latents per emotion")
    p.add_argument("--repetition", type=int, default=0)
    p.add_argument("--hparams", default=default_hparams(), help="SpeechBrain hyperparameter file")
    p.add_argument("--seed", type=int, default=None, help="default: 1234, 2345, ... by repetition")
    p.add_argument("--sampler-seed", type=int, default=DEFAULT_SAMPLER_SEED,
                   help="seed of the training sampler (default: SpeechBrain's default, as in the paper)")
    p.add_argument("--epochs", type=int, default=100, help="fine-tuning epochs on top of the Stage-1 epochs")
    p.add_argument("--stage1-epochs", type=int, default=100, help="epochs the Stage-1 adapter was trained for")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    num_aug = DEFAULT_NUM_AUG[args.setting] if args.num_aug is None else args.num_aug
    seed = REPETITION_SEEDS[args.repetition] if args.seed is None else args.seed

    store = LatentStore(args.latents)
    split = {s: filter_language(load_manifest(args.manifest_dir, args.fold, s), args.language)
             for s in ("train", "valid", "test")}
    if not split["train"]:
        raise SystemExit(f"No {args.language} utterances in the train split of {args.manifest_dir}")

    # Training items: the real target-language utterances followed by the generated latents.
    items = {key: {"emo": item["emo"]} for key, item in split["train"].items()}
    latents = load_latent_dict(split["train"], store)
    if num_aug > 0:
        if args.generated is None:
            raise SystemExit("--generated is required when --num-aug > 0")
        # Zero-shot: every class of the target label inventory (classes of its test split).
        # Few-shot: every class with at least one labelled target-language training utterance.
        source = split["test"] if args.setting == "zero_shot" else split["train"]
        classes = sorted({v["emo"] for v in source.values()}, key=EMOTIONS.index)
        lo, hi = args.repetition * num_aug, (args.repetition + 1) * num_aug
        for emotion in classes:
            pool = np.load(os.path.join(args.generated, f"{emotion}.npy"))
            if pool.shape[0] < hi:
                raise SystemExit(f"{emotion}.npy has {pool.shape[0]} samples; repetition "
                                 f"{args.repetition} needs {hi}. Generate more samples.")
            for i in range(lo, hi):
                key = f"{emotion}_sample_{i:04d}"
                items[key] = {"emo": emotion}
                latents[key] = pool[i].astype(np.float32)
        print(f"augmented {', '.join(classes)} with samples [{lo}, {hi}) each")
    print(f"train: {len(items)} ({len(split['train'])} real), valid: {len(split['valid'])}, "
          f"test: {len(split['test'])}")

    model = train(args.hparams, args.out, items, latents, split["valid"], load_latent_dict(split["valid"], store),
                  layer=args.layer, epochs=args.epochs, lr=args.lr, seed=seed, sampler_seed=args.sampler_seed,
                  init_adapter=load_adapter(args.adapter), start_epoch=args.stage1_epochs, device=args.device)

    os.makedirs(args.out, exist_ok=True)
    save_adapter(model, os.path.join(args.out, "adapter.safetensors"))
    x_test, y_test, _ = load_latents(split["test"], store)
    metrics = evaluate(model, x_test, y_test, layer=args.layer, device=args.device)["overall"]
    metrics.update({"setting": args.setting, "language": args.language, "fold": args.fold,
                    "layer": args.layer, "num_aug": num_aug, "repetition": args.repetition, "seed": seed,
                    "sampler_seed": args.sampler_seed})
    dump_json(metrics, os.path.join(args.out, "test_metrics.json"))
    print(f"{args.language} test {format_metrics(metrics)}")
    print("  " + "  ".join(f"{k} {100 * v:.1f}" for k, v in metrics["recall"].items()))


if __name__ == "__main__":
    main()
