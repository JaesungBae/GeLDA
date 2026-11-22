"""Build the multilingual SER manifests used in the paper from EmoBox metadata.

GeLDA does not redistribute any audio or labels. Everything is derived from
the metadata of the EmoBox toolkit (https://github.com/emo-box/EmoBox), and
the audio must be obtained from the original corpora as described there.

Three kinds of manifests are produced, each with three folds:

* ``full``                 : all 14 languages (Oracle model and few-shot SER).
* ``zero_shot_<lang>``     : like ``full`` but, for the target language, only
                             NEUTRAL utterances are kept in train/valid
                             (D^(\\kappa) U D_NEU^(\\kappa); GeLDA zero-shot SER).
* ``source_only_<lang>``   : ``zero_shot_<lang>`` with every target-language
                             utterance removed from train/valid (D^(\\kappa);
                             the "Pretrained" zero-shot baseline).

The test split always contains every language and every emotion.

Corpora that have no official validation split are split 80/20 into
train/valid with a seeded shuffle. The dataset order below and the seed
reproduce the splits used in the paper.
"""

import json
import os
import random
from typing import Callable, Dict, Optional

import numpy as np

from .constants import EMOTIONS, NEUTRAL

# EmoBox corpora used in the paper, in the order in which the 80/20 splits were
# drawn. CASIA, EmoV-DB, M3ED, MSP-Podcast and PAVOQUE are not used.
DATASETS = (
    "emovo", "mesd", "mer2023", "tess", "iemocap", "esd", "meld", "turevdb",
    "ravdess", "jlcorpus", "urdu", "polish", "savee", "aesdd", "asvpesd",
    "emodb", "subesco", "mead", "enterface", "emns", "cremad",
    "emozionalmente", "cafe", "ased", "resd", "shemo", "oreau",
)

CORPUS_LANGUAGE = {
    "aesdd": "greek", "ased": "amharic", "asvpesd": "mix", "cafe": "french",
    "cremad": "english", "emns": "english", "emodb": "german",
    "emovo": "italian", "emozionalmente": "italian", "enterface": "english",
    "esd": "mix", "iemocap": "english", "jlcorpus": "english",
    "mead": "english", "meld": "english", "mer2023": "mandarin",
    "mesd": "spanish", "oreau": "french", "polish": "polish",
    "ravdess": "english", "resd": "russian", "savee": "english",
    "shemo": "persian", "subesco": "bangla", "tess": "english",
    "turevdb": "turkish", "urdu": "urdu",
}

FOLDS = (1, 2, 3)
SPLITS = ("train", "valid", "test")
SPLIT_SEED = 1234
VALID_RATIO = 0.2
MAX_DURATION = 30.0  # seconds; longer clips do not fit the Whisper encoder window

Manifest = Dict[str, dict]


def _language(dataset: str, key: str, item: dict) -> Optional[str]:
    """Language of an utterance (two corpora mix several languages)."""
    lang = CORPUS_LANGUAGE[dataset]
    if lang != "mix":
        return lang
    if dataset == "asvpesd":
        stem = key[len("asvpesd-"):] if key.startswith("asvpesd-") else key
        parts = stem.split("-")
        if len(parts) < 9:
            return None
        return {"01": "mandarin", "02": "english", "04": "french"}.get(parts[8])
    if dataset == "esd":
        speaker = int(item["wav"].split("/")[2])
        return "mandarin" if speaker <= 10 else "english"
    raise ValueError(f"Unknown multilingual corpus: {dataset}")


def _load_label_map(corpus_dir: str) -> Dict[str, str]:
    for name in ("label_map.json", "label_dict.json"):  # EMOVO ships label_dict.json
        path = os.path.join(corpus_dir, name)
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(f"No label_map.json in {corpus_dir}")


def _map_label(label_map: Dict[str, str], raw: str) -> str:
    if raw in label_map:
        return label_map[raw]
    lowered = {k.lower(): v for k, v in label_map.items()}  # EmoDB: "neutral" vs "Neutral"
    return lowered[raw.lower()]


def _load_split(path: str, dataset: str, label_map, keep) -> Manifest:
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for key, item in raw.items():
        item = dict(item)
        item["emo"] = _map_label(label_map, item["emo"])
        if float(item["length"]) > MAX_DURATION or item["emo"] not in EMOTIONS:
            continue
        lang = _language(dataset, key, item)
        if lang is None:
            continue
        item["lang"] = lang
        if keep is not None and not keep(item):
            continue
        out[key] = item
    return out


def build_folds(emobox_root: str, keep: Optional[Callable[[dict], bool]] = None,
                seed: int = SPLIT_SEED) -> Dict[int, Dict[str, Manifest]]:
    """Build train/valid/test manifests for all folds.

    ``keep`` filters train/valid utterances *before* the 80/20 split (the test
    split is never filtered). The shuffle RNG is shared across folds, so all
    folds are always built together.
    """
    data_dir = os.path.join(emobox_root, "data")
    rng = random.Random(seed)
    folds = {}
    for fold in FOLDS:
        merged = {split: {} for split in SPLITS}
        for dataset in DATASETS:
            corpus_dir = os.path.join(data_dir, dataset)
            if not os.path.isdir(corpus_dir):
                raise FileNotFoundError(f"EmoBox metadata not found: {corpus_dir}")
            # Corpora with fewer than three folds reuse their first fold.
            f = fold if os.path.isdir(os.path.join(corpus_dir, f"fold_{fold}")) else 1
            label_map = _load_label_map(corpus_dir)

            def path(split):
                return os.path.join(corpus_dir, f"fold_{f}", f"{dataset}_{split}_fold_{f}.json")

            train = _load_split(path("train"), dataset, label_map, keep)
            if os.path.isfile(path("valid")):
                valid = _load_split(path("valid"), dataset, label_map, keep)
            else:
                keys = list(train)
                n_train = int(len(keys) * (1 - VALID_RATIO))
                order = np.arange(len(keys))
                rng.shuffle(order)
                valid = {keys[i]: train[keys[i]] for i in order[n_train:]}
                train = {keys[i]: train[keys[i]] for i in order[:n_train]}
            test = _load_split(path("test"), dataset, label_map, None)
            merged["train"].update(train)
            merged["valid"].update(valid)
            merged["test"].update(test)
        folds[fold] = merged
    return folds


def build_full(emobox_root: str):
    return build_folds(emobox_root)


def build_zero_shot(emobox_root: str, target: str):
    """Keep only NEUTRAL target-language utterances in train/valid."""
    return build_folds(emobox_root, keep=lambda v: v["lang"] != target or v["emo"] == NEUTRAL)


def build_source_only(zero_shot_folds, target: str):
    """Remove every target-language utterance from train/valid."""
    out = {}
    for fold, splits in zero_shot_folds.items():
        out[fold] = {
            split: (items if split == "test"
                    else {k: v for k, v in items.items() if v["lang"] != target})
            for split, items in splits.items()
        }
    return out


def write_folds(folds, out_dir: str) -> None:
    for fold, splits in folds.items():
        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        for split, items in splits.items():
            with open(os.path.join(fold_dir, f"{split}.json"), "w") as f:
                json.dump(items, f, indent=2)


def load_manifest(manifest_dir: str, fold: int, split: str) -> Manifest:
    with open(os.path.join(manifest_dir, f"fold_{fold}", f"{split}.json")) as f:
        return json.load(f)
