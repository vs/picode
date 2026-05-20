"""Adaptive Instance Normalization (AdaIN) and message mapping network.

AdaIN modulates feature statistics (mean/variance) using a message-derived
latent vector, replacing spatial message concatenation to eliminate grid artifacts.
Inspired by StyleGAN's style injection mechanism.
"""

import torch.nn as nn
from torch import Tensor


class MappingNetwork(nn.Module):
    """Maps message bits to a latent vector for AdaIN injection.

    Architecture: Linear → ReLU → Linear → ReLU
    Produces a shared latent vector that each AdaIN layer projects from.

    Args:
        num_bits: Number of input message bits (e.g., 127).
        mapping_dim: Dimension of the output latent vector (e.g., 256).
    """

    def __init__(self, num_bits: int = 127, mapping_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_bits, mapping_dim),
            nn.ReLU(),
            nn.Linear(mapping_dim, mapping_dim),
            nn.ReLU(),
        )

    def forward(self, message: Tensor) -> Tensor:
        """Map message bits to latent vector.

        Args:
            message: (B, num_bits) binary tensor.

        Returns:
            (B, mapping_dim) latent vector.
        """
        return self.net(message)


class AdaIN(nn.Module):
    """Adaptive Instance Normalization with learned projection from latent vector.

    Normalizes features per-instance, then modulates with message-derived
    scale (gamma) and shift (beta). Uses 1+gamma formulation so that
    gamma=0, beta=0 produces identity (no modulation).

    Args:
        mapping_dim: Dimension of input latent vector w.
        num_features: Number of feature channels to modulate.
    """

    def __init__(self, mapping_dim: int, num_features: int) -> None:
        super().__init__()
        self.projection = nn.Linear(mapping_dim, 2 * num_features)
        # Zero-init bias so AdaIN starts as identity when w=0
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Apply AdaIN modulation.

        Args:
            x: (B, C, H, W) feature map.
            w: (B, mapping_dim) latent vector.

        Returns:
            (B, C, H, W) modulated feature map.
        """
        gamma_beta = self.projection(w)  # (B, 2*C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # (B, C) each

        # Instance normalization
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-8
        x_norm = (x - mean) / std

        # Modulate with 1+gamma for identity initialization
        return x_norm * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
