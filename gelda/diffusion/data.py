"""Training data for the GeLDA diffusion model and its language-reference conditioning."""

import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import EMOTIONS, NEUTRAL

# How the language-reference latent z_ref is drawn for a training sample of language k:
#   same_language          z_ref of a random utterance of language k (any emotion).
#                          This is the setting used for every released checkpoint.
#                          For a zero-shot target, train/valid only contain NEUTRAL
#                          utterances, so its references are always neutral.
#   same_language_neutral  z_ref of a random NEUTRAL utterance of language k.
REFERENCE_MODES = ("same_language", "same_language_neutral")


def to_square(z: np.ndarray, size: int) -> np.ndarray:
    """Zero-pad a latent vector to size**2 and reshape it to [1, size, size]."""
    padded = np.zeros(size * size, dtype=np.float32)
    padded[:z.shape[-1]] = z
    return padded.reshape(1, size, size)


def from_square(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Inverse of :func:`to_square` for a batch [B, 1, S, S] -> [B, dim]."""
    return x.reshape(x.shape[0], -1)[:, :dim]


class ReferencePool:
    """Latents of each language that may serve as the language reference."""

    def __init__(self, manifest, store, mode="same_language"):
        if mode not in REFERENCE_MODES:
            raise ValueError(f"reference mode must be one of {REFERENCE_MODES}")
        index = defaultdict(list)
        for key, item in manifest.items():
            if mode == "same_language_neutral" and item["emo"] != NEUTRAL:
                continue
            index[item["lang"]].append((item["dataset"], key))
        self.index = dict(index)
        self.store = store
        self._cache = {}

    def languages(self):
        return sorted(self.index)

    def _load(self, dataset, key):
        if (dataset, key) not in self._cache:
            self._cache[(dataset, key)] = self.store.load(dataset, key)
        return self._cache[(dataset, key)]

    def sample(self, lang: str, rng: random.Random = random) -> np.ndarray:
        if lang not in self.index:
            raise KeyError(f"No reference utterances for language '{lang}'")
        return self._load(*rng.choice(self.index[lang]))


class DiffusionDataset(Dataset):
    """Items: (x [1, S, S], emotion index into EMOTIONS, z_ref [ref_dim])."""

    def __init__(self, manifest, store, input_size, reference="same_language"):
        self.items = [(item["dataset"], key, item["emo"], item["lang"]) for key, item in manifest.items()]
        self.store = store
        self.input_size = input_size
        self.references = ReferencePool(manifest, store, reference)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        dataset, key, emo, lang = self.items[idx]
        x = to_square(self.store.load(dataset, key), self.input_size)
        ref = self.references.sample(lang)
        return (torch.from_numpy(x), torch.tensor(EMOTIONS.index(emo), dtype=torch.long),
                torch.from_numpy(np.asarray(ref, dtype=np.float32)))
