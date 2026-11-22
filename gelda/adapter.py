"""The SER task adapter H = H^(3) o H^(2) o H^(1) on top of a frozen SFM.

    z1 = LN(W1 z0 + b1)          (Z^(1), 512-d)
    z2 = LN(W2 relu(z1) + b2)    (Z^(2), 256-d)
    y  = W3 relu(z2) + b3        (logits over SER_LABELS)

LN is a layer normalisation without learnable affine parameters.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .constants import ADAPTER_HIDDEN, SER_LABELS


class TaskAdapter(nn.Module):
    def __init__(self, in_dim: int, hidden=ADAPTER_HIDDEN, num_classes: int = len(SER_LABELS)):
        super().__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(in_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], num_classes)

    @staticmethod
    def _norm(x):
        return F.layer_norm(x, x.shape[-1:])

    def forward(self, z: torch.Tensor, layer: int = 0) -> torch.Tensor:
        """Logits for a latent ``z`` that lives in Z^(layer)."""
        if layer == 0:
            z = self._norm(self.fc1(z))
        if layer <= 1:
            z = self._norm(self.fc2(F.relu(z)))
        return self.fc3(F.relu(z))

    @torch.no_grad()
    def embed(self, z0: torch.Tensor):
        """Map z^(0) to (z^(1), z^(2))."""
        z1 = self._norm(self.fc1(z0))
        z2 = self._norm(self.fc2(F.relu(z1)))
        return z1, z2

    def freeze_up_to(self, layer: int) -> None:
        """Freeze every layer that produces Z^(<=layer) (Stage 3 fine-tuning)."""
        for module in (self.fc1, self.fc2)[:layer]:
            for p in module.parameters():
                p.requires_grad_(False)


def load_adapter(path: str, map_location="cpu") -> TaskAdapter:
    from safetensors.torch import load_file

    state = load_file(path, device=str(map_location))
    model = TaskAdapter(state["fc1.weight"].shape[1], num_classes=state["fc3.weight"].shape[0])
    model.load_state_dict(state)
    return model


def save_adapter(model: TaskAdapter, path: str) -> None:
    from safetensors.torch import save_file

    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, path)
