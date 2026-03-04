"""Noise distortions using Kornia.

Note: Kornia doesn't have a direct GaussianNoise equivalent,
so we use the same torch.randn approach as native.
"""

from typing import Any

import torch
from torch import Tensor

from picode.distortions.base import Distortion


class GaussianNoise(Distortion):
    """Add Gaussian noise to images.

    API-compatible with native.GaussianNoise.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        std: Standard deviation of noise (default: 0.02).
    """

    name = "gaussian_noise"

    def __init__(self, intensity: float = 0.5, std: float = 0.02):
        super().__init__(intensity)
        self.std = std

    def forward(self, x: Tensor) -> Tensor:
        """Apply Gaussian noise."""
        if self.intensity == 0.0:
            return x

        effective_std = self.std * self.intensity
        noise = torch.randn_like(x) * effective_std
        return torch.clamp(x + noise, 0.0, 1.0)

    def sample_parameters(self) -> dict[str, Any]:
        """Sample random noise parameters."""
        return {"std": torch.rand(1).item() * self.std}
