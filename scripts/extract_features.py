#!/usr/bin/env python
"""Extract utterance-level SFM features (the latent space Z^(0)).

    python scripts/extract_features.py --sfm whisper-large-v3 \
        --manifest-dir manifests/full --audio-root /path/to/EmoBox \
        --out features/whisper-large-v3

Every utterance of every fold/split in ``--manifest-dir`` is processed once;
features are written to ``<out>/<dataset>/<key>.npy``. ``wav`` paths in the
manifests are resolved relative to ``--audio-root`` (the EmoBox root, whose
``downloads/`` folder holds the corpora). Use ``--shard``/``--num-shards`` to
split the work across GPUs.
"""

import argparse
import glob
import json

from gelda.constants import SFM_DIMS
from gelda.features import extract


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sfm", required=True, choices=sorted(SFM_DIMS))
    p.add_argument("--manifest-dir", default="manifests/full")
    p.add_argument("--audio-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-dir", default=None, help="Hugging Face cache directory")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    manifest = {}
    for path in sorted(glob.glob(f"{args.manifest_dir}/fold_*/*.json")):
        with open(path) as f:
            manifest.update(json.load(f))
    if not manifest:
        raise SystemExit(f"No manifests found in {args.manifest_dir}")
    print(f"{len(manifest)} utterances")

    failed = extract(manifest, args.sfm, args.audio_root, args.out, device=args.device,
                     cache_dir=args.cache_dir, skip_existing=not args.overwrite,
                     shard=args.shard, num_shards=args.num_shards)
    if failed:
        print(f"{len(failed)} utterances failed:")
        for key, err in failed[:20]:
            print(f"  {key}: {err}")


if __name__ == "__main__":
    main()
