#!/usr/bin/env python
"""Build the train/valid/test manifests of the paper from an EmoBox checkout.

    python scripts/prepare_manifests.py --emobox-root /path/to/EmoBox --out manifests

Writes
    manifests/full/fold_{1,2,3}/{train,valid,test}.json
    manifests/zero_shot_<lang>/...      (target keeps only NEUTRAL in train/valid)
    manifests/source_only_<lang>/...    (target removed from train/valid)
"""

import argparse
import os

from gelda.emobox import build_full, build_source_only, build_zero_shot, write_folds

# Zero-shot target languages of the paper (languages that have NEUTRAL data).
ZERO_SHOT_TARGETS = ["amharic", "bangla", "french", "german", "italian", "persian",
                     "polish", "russian", "spanish", "urdu"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emobox-root", required=True, help="EmoBox repository (contains data/<corpus>/...)")
    p.add_argument("--out", default="manifests")
    p.add_argument("--targets", nargs="*", default=ZERO_SHOT_TARGETS,
                   help="zero-shot target languages (default: the 10 languages of the paper)")
    p.add_argument("--no-source-only", action="store_true", help="skip the source-only baseline manifests")
    args = p.parse_args()

    print("building: full")
    write_folds(build_full(args.emobox_root), os.path.join(args.out, "full"))
    for lang in args.targets:
        print(f"building: zero_shot_{lang}")
        zero_shot = build_zero_shot(args.emobox_root, lang)
        write_folds(zero_shot, os.path.join(args.out, f"zero_shot_{lang}"))
        if not args.no_source_only:
            write_folds(build_source_only(zero_shot, lang), os.path.join(args.out, f"source_only_{lang}"))
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
