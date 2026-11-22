"""Speech emotion recognition for an audio file: frozen SFM -> z^(0) -> task adapter -> emotion scores."""

import torch

from .adapter import load_adapter
from .constants import SER_LABELS, SFM_DIMS
from .features import FeatureExtractor


class EmotionRecognizer:
    """Predicts emotion probabilities for an utterance.

        recognizer = EmotionRecognizer("checkpoints/zero_shot/whisper-large-v3/french/fold1/"
                                       "ser_adapter_gelda_z0.safetensors")
        scores = recognizer("utterance.wav")   # {"Angry": 0.02, ..., "Sad": 0.91, "Surprise": 0.01}

    ``adapter`` is any task adapter of this repository (Stage 1, Stage 3 or a baseline). The SFM
    is chosen from the adapter's input size unless ``sfm`` is given. Utterances should be at most
    30 s long, as in the paper.
    """

    def __init__(self, adapter: str, sfm: str = None, device: str = None, cache_dir: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter = load_adapter(adapter).to(self.device).eval()
        if sfm is None:
            matches = [name for name, dim in SFM_DIMS.items() if dim == self.adapter.in_dim]
            if len(matches) != 1:
                raise ValueError(f"Cannot infer the SFM from input size {self.adapter.in_dim}; pass sfm=...")
            sfm = matches[0]
        self.sfm = sfm
        self.extractor = FeatureExtractor(sfm, device=self.device, cache_dir=cache_dir)

    @torch.no_grad()
    def __call__(self, wav_path: str, channel: int = 1) -> dict:
        """Emotion probabilities of one audio file, keyed by emotion name."""
        z0 = torch.from_numpy(self.extractor(wav_path, channel=channel))[None].to(self.device)
        probs = torch.softmax(self.adapter(z0), dim=-1)[0]
        return {label: float(p) for label, p in zip(SER_LABELS, probs)}

    def predict(self, wav_path: str, channel: int = 1) -> str:
        """The most likely emotion of one audio file."""
        scores = self(wav_path, channel=channel)
        return max(scores, key=scores.get)
