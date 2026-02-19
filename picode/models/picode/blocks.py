"""Shared building blocks for Picode model."""

import torch.nn as nn
from torch import Tensor


class ResBlock(nn.Module):
    """Residual block with GroupNorm and LeakyReLU.

    Allows gradients to bypass the block via skip connection.

    Args:
        channels: Number of input/output channels.
        groups: Number of groups for GroupNorm (default: 8).
    """

    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        # Ensure groups divides channels evenly
        groups = min(groups, channels)
        while channels % groups != 0:
            groups -= 1

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)
