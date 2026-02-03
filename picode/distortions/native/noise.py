"""Noise distortions.

Matches StegaStamp implementation in models.py:144-146:
    noise = tf.random_normal(shape=tf.shape(encoded_image), mean=0.0, stddev=rnd_noise)
    encoded_image = encoded_image + noise
    encoded_image = tf.clip_by_value(encoded_image, 0, 1)
"""

import torch
from torch import Tensor

from picode.distortions.base import Distortion


class GaussianNoise(Distortion):
    """Add Gaussian noise to images.

    Matches StegaStamp: simple additive Gaussian noise with clipping.
    Default std=0.02 matches StegaStamp's --rnd_noise default.

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        std: Standard deviation of noise (StegaStamp default: 0.02).
    """

    name = "gaussian_noise"

    def __init__(self, intensity: float = 0.5, std: float = 0.02):
        super().__init__(intensity)
        self.std = std

    def forward(self, x: Tensor) -> Tensor:
        """Apply Gaussian noise.

        Matches StegaStamp models.py:144-146 exactly:
        1. Generate noise with mean=0, stddev=rnd_noise
        2. Add noise to image
        3. Clip to [0, 1]

        Args:
            x: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Noisy tensor clamped to [0, 1].
        """
        if self.intensity == 0.0:
            return x

        # StegaStamp: rnd_noise = random() * ramp * args.rnd_noise
        # We use intensity to scale the std directly
        effective_std = self.std * self.intensity

        noise = torch.randn_like(x) * effective_std
        return torch.clamp(x + noise, 0.0, 1.0)

    def sample_parameters(self) -> dict:
        """Sample random noise parameters."""
        # StegaStamp samples: rnd_noise = random_uniform([]) * ramp * args.rnd_noise
        return {
            "std": torch.rand(1).item() * self.std,
        }
