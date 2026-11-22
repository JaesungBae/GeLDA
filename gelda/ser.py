"""Latent loading and evaluation for the SER task adapter.

Training (Stages 1 and 3) is the SpeechBrain recipe in :mod:`gelda.recipe`.
"""

import json
import os
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, recall_score)

from .constants import SER_LABELS


# --------------------------------------------------------------------------- data

class LatentStore:
    """Latent vectors stored as ``<root>/<dataset>/<key>.npy``."""

    def __init__(self, root: str):
        self.root = root

    def path(self, dataset: str, key: str) -> str:
        return os.path.join(self.root, dataset, key + ".npy")

    def load(self, dataset: str, key: str) -> np.ndarray:
        path = self.path(dataset, key)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing latent {path}. Extract features/latents for every utterance in the manifest first.")
        return np.load(path).astype(np.float32)


def load_latents(manifest: Dict[str, dict], store: LatentStore):
    """Stack the latents and SER label indices of a manifest."""
    keys = list(manifest)
    x = np.stack([store.load(manifest[k]["dataset"], k) for k in keys])
    y = np.array([SER_LABELS.index(manifest[k]["emo"]) for k in keys], dtype=np.int64)
    langs = [manifest[k]["lang"] for k in keys]
    return x, y, langs


def load_latent_dict(manifest: Dict[str, dict], store: LatentStore) -> Dict[str, np.ndarray]:
    """Latent of every utterance of a manifest, keyed by utterance id."""
    return {key: store.load(item["dataset"], key) for key, item in manifest.items()}


def filter_language(manifest: Dict[str, dict], lang: str) -> Dict[str, dict]:
    return {k: v for k, v in manifest.items() if v["lang"] == lang}


# -------------------------------------------------------------------- evaluation

@torch.no_grad()
def predict(model, x, layer: int = 0, device="cuda", batch_size: int = 1024) -> np.ndarray:
    model.to(device).eval()
    x = torch.as_tensor(x)
    preds = [model(x[i:i + batch_size].to(device), layer).argmax(-1).cpu() for i in range(0, len(x), batch_size)]
    return torch.cat(preds).numpy()


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """UA (unweighted average recall), WA (accuracy), macro-F1 and per-class recall."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = sorted(set(y_true.tolist()))
    recalls = recall_score(y_true, y_pred, labels=present, average=None, zero_division=0)
    return {
        "UA": float(balanced_accuracy_score(y_true, y_pred)),
        "WA": float(accuracy_score(y_true, y_pred)),
        "macro_F1": float(f1_score(y_true, y_pred, labels=present, average="macro")),
        "recall": {SER_LABELS[c]: float(r) for c, r in zip(present, recalls)},
        "confusion": confusion_matrix(y_true, y_pred, labels=present).tolist(),
        "labels": [SER_LABELS[c] for c in present],
        "num_samples": int(len(y_true)),
    }


def evaluate(model, x, y, langs: Optional[Iterable[str]] = None, layer: int = 0, device="cuda") -> dict:
    """Overall metrics and, if ``langs`` is given, metrics per language."""
    pred = predict(model, x, layer, device)
    out = {"overall": compute_metrics(y, pred)}
    if langs is not None:
        langs = np.asarray(list(langs))
        out["per_language"] = {
            lang: compute_metrics(np.asarray(y)[langs == lang], pred[langs == lang])
            for lang in sorted(set(langs.tolist()))
        }
    return out


def format_metrics(m: dict) -> str:
    return f"UA {100 * m['UA']:.2f}  WA {100 * m['WA']:.2f}  F1 {100 * m['macro_F1']:.2f}"


def dump_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
