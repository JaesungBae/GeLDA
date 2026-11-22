"""Shared constants: emotion classes, languages and speech foundation models (SFMs)."""

# Emotion classes. The diffusion model's class-embedding table uses this order
# (index 7 is the "null" class used for classifier-free guidance).
EMOTIONS = ("Disgust", "Happy", "Sad", "Fear", "Neutral", "Surprise", "Angry")
NEUTRAL = "Neutral"

# The SER task adapter predicts classes in alphabetical order.
SER_LABELS = tuple(sorted(EMOTIONS))

LANGUAGES = (
    "amharic", "bangla", "english", "french", "german", "greek", "italian",
    "mandarin", "persian", "polish", "russian", "spanish", "turkish", "urdu",
)

# Mean-pooled output dimension of each SFM, i.e. the size of a z^(0) vector.
SFM_DIMS = {
    "whisper-large-v3": 1280,
    "wavlm-large": 1024,
    "emotion2vec-base": 768,
}

# Output sizes of the first two adapter layers, i.e. the sizes of z^(1) and z^(2).
ADAPTER_HIDDEN = (512, 256)

LAYERS = (0, 1, 2)


def latent_dim(sfm: str, layer: int) -> int:
    """Dimension of the latent space Z^(layer) for a given SFM."""
    if layer == 0:
        return SFM_DIMS[sfm]
    return ADAPTER_HIDDEN[layer - 1]
