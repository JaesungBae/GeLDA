#!/usr/bin/env python
"""Recognise the emotion of audio files with a released (or your own) task adapter.

    python scripts/predict.py \
        --adapter checkpoints/zero_shot/whisper-large-v3/french/fold1/ser_adapter_gelda_z0.safetensors \
        utterance.wav [more.wav ...]

Each file is encoded by the frozen SFM (chosen from the adapter, or with ``--sfm``), mean-pooled
into z^(0) and classified by the adapter. Prints the probability of every emotion.
"""

import argparse

from gelda.constants import SFM_DIMS
from gelda.inference import EmotionRecognizer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", nargs="+", help="audio files (at most 30 s each)")
    p.add_argument("--adapter", required=True, help="task adapter (.safetensors)")
    p.add_argument("--sfm", default=None, choices=sorted(SFM_DIMS), help="default: inferred from the adapter")
    p.add_argument("--device", default=None, help="default: cuda if available")
    p.add_argument("--cache-dir", default=None, help="where the SFM weights are downloaded")
    args = p.parse_args()

    recognizer = EmotionRecognizer(args.adapter, sfm=args.sfm, device=args.device, cache_dir=args.cache_dir)
    for path in args.audio:
        scores = recognizer(path)
        best = max(scores, key=scores.get)
        print(f"{path}: {best}")
        print("  " + "  ".join(f"{label} {score:.3f}" for label, score in scores.items()))


if __name__ == "__main__":
    main()
