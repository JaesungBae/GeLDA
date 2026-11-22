"""Utterance-level SFM features, i.e. the latent space Z^(0).

For every utterance we take the last hidden layer of a frozen SFM, apply a
parameter-free layer normalisation over the whole (time x feature) matrix, and
average over time. The result is stored as ``<out_dir>/<dataset>/<key>.npy``.
"""

import os
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

SAMPLE_RATE = 16000
HF_SOURCES = {
    "whisper-large-v3": "openai/whisper-large-v3",
    "wavlm-large": "microsoft/wavlm-large",
    "emotion2vec-base": "emotion2vec/emotion2vec_base",
}
VIDEO_EXTENSIONS = (".avi", ".mp4", ".mkv", ".mov")


def _to_wav(path: str, tmp_dir: str) -> str:
    """Return a path torchaudio can read; video containers are converted with ffmpeg."""
    import torchaudio

    if not path.lower().endswith(VIDEO_EXTENSIONS):
        try:
            torchaudio.load(path)
            return path
        except Exception:
            pass
    out = os.path.join(tmp_dir, os.path.splitext(os.path.basename(path))[0] + ".wav")
    subprocess.run(["ffmpeg", "-y", "-i", path, "-vn", "-acodec", "pcm_s16le", out],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def _load(path, channel, start_time=None, end_time=None):
    import torchaudio

    if start_time is not None and end_time is not None:
        sr = torchaudio.info(path).sample_rate
        offset = int(start_time * sr)
        wav, sr = torchaudio.load(path, frame_offset=offset, num_frames=int(end_time * sr) - offset)
    else:
        wav, sr = torchaudio.load(path)
    return wav, sr


def _pool(hidden: torch.Tensor) -> np.ndarray:
    """hidden: [1, T, D] -> layer-norm over (T, D) -> mean over T -> [D]."""
    hidden = F.layer_norm(hidden, hidden.shape[1:])
    return hidden[0].mean(0).float().cpu().numpy()


class FeatureExtractor:
    def __init__(self, sfm: str, device: str = "cuda", cache_dir: str = None):
        self.sfm = sfm
        self.device = device
        source = HF_SOURCES[sfm]
        if sfm.startswith("whisper"):
            from transformers import WhisperFeatureExtractor, WhisperForAudioClassification

            self.processor = WhisperFeatureExtractor.from_pretrained(source, cache_dir=cache_dir)
            self.model = WhisperForAudioClassification.from_pretrained(source, cache_dir=cache_dir)
        elif sfm.startswith("wavlm"):
            from transformers import AutoFeatureExtractor, WavLMModel

            self.processor = AutoFeatureExtractor.from_pretrained(source, cache_dir=cache_dir)
            self.model = WavLMModel.from_pretrained(source, cache_dir=cache_dir)
        elif sfm.startswith("emotion2vec"):
            from funasr import AutoModel

            self.model = AutoModel(model=source, hub="hf", disable_update=True)
            return
        else:
            raise ValueError(f"Unsupported SFM: {sfm}")
        self.model.eval().to(device)

    @torch.no_grad()
    def __call__(self, wav_path, channel=1, start_time=None, end_time=None) -> np.ndarray:
        if self.sfm.startswith("emotion2vec"):
            feats = self.model.generate(wav_path, granularity="frame", extract_embedding=False)[0]["feats"]
            return _pool(torch.as_tensor(feats, device=self.device)[None])

        import torchaudio

        wav, sr = _load(wav_path, channel, start_time, end_time)
        if self.sfm.startswith("whisper"):
            wav = wav[channel - 1]
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            n_frames = wav.numel() // 320  # encoder frame rate: 50 Hz
            inputs = self.processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
            hidden = self.model(inputs.to(self.device), output_hidden_states=True).hidden_states[-1]
            return _pool(hidden[:, :n_frames])

        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav[channel - 1]
        inputs = self.processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self.device)
        return _pool(self.model(**inputs).last_hidden_state)


def extract(manifest: dict, sfm: str, audio_root: str, out_dir: str, device="cuda",
            cache_dir=None, skip_existing=True, shard=0, num_shards=1):
    """Extract Z^(0) features for every utterance in ``manifest``."""
    from tqdm import tqdm

    extractor = FeatureExtractor(sfm, device=device, cache_dir=cache_dir)
    keys = sorted(manifest)[shard::num_shards]
    failed = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for key in tqdm(keys, desc=f"{sfm} [{shard + 1}/{num_shards}]"):
            item = manifest[key]
            out_path = os.path.join(out_dir, item["dataset"], key + ".npy")
            if skip_existing and os.path.isfile(out_path):
                continue
            try:
                wav_path = _to_wav(os.path.join(audio_root, item["wav"]), tmp_dir)
                feat = extractor(wav_path, int(item.get("channel", 1)),
                                 item.get("start_time"), item.get("end_time"))
            except Exception as err:  # keep going; report at the end
                failed.append((key, repr(err)))
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, feat.astype(np.float32))
    return failed
