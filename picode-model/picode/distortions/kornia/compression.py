"""Compression distortions using Kornia."""

from typing import Any

import kornia.enhance
import torch
from torch import Tensor

from picode.distortions.base import Distortion


class JPEGCompression(Distortion):
    """Differentiable JPEG compression using kornia.enhance.jpeg_codec_differentiable.

    API-compatible with native.JPEGCompression.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        quality: JPEG quality factor (1-100).
    """

    name = "jpeg_compression"

    def __init__(self, intensity: float = 0.5, quality: int = 50):
        super().__init__(intensity)
        self.quality = max(1, min(100, quality))

    def forward(self, x: Tensor) -> Tensor:
        """Apply differentiable JPEG compression."""
        if self.intensity == 0.0:
            return x

        # Create quality tensor matching batch size
        batch_size = x.shape[0]
        quality_tensor = torch.full((batch_size,), float(self.quality), device=x.device)

        # Apply JPEG compression
        compressed = kornia.enhance.jpeg_codec_differentiable(x, quality_tensor)

        # Blend based on intensity
        output = x + self.intensity * (compressed - x)

        return torch.clamp(output, 0.0, 1.0)

    def sample_parameters(self) -> dict[str, Any]:
        """Sample random compression parameters."""
        quality = int(10 + torch.rand(1).item() * 80)
        return {"quality": quality}


__all__ = ["JPEGCompression"]
