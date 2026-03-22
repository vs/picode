"""Mobile-optimized decoder for picode_v2.

Uses MobileNetV2-style architecture with:
- InvertedResidual blocks
- BatchNorm (fuses with conv at export)
- ReLU6 (hardware accelerated)
- No STN (robustness via training augmentation)
"""

import torch
import torch.nn as nn
from torch import Tensor

from picode.models.base import Decoder as BaseDecoder
from picode.models.picode_v2.blocks import InvertedResidual


class MobileDecoder(BaseDecoder):
    """Mobile-optimized decoder using MobileNetV2 architecture.

    Target: < 500K parameters, < 100ms inference on iPhone 12+.

    Args:
        num_bits: Number of bits in the message (default: 100).
    """

    def __init__(self, num_bits: int = 100) -> None:
        super().__init__()
        self.num_bits = num_bits

        # Stem: standard conv to expand channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        # Inverted residual blocks
        # (in_ch, out_ch, stride, expand_ratio)
        self.blocks = nn.Sequential(
            # 200x200 -> 100x100
            InvertedResidual(16, 24, stride=2, expand_ratio=6),
            InvertedResidual(24, 24, stride=1, expand_ratio=6),
            # 100x100 -> 50x50
            InvertedResidual(24, 32, stride=2, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            # 50x50 -> 25x25
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            # 25x25 -> 13x13
            InvertedResidual(64, 96, stride=2, expand_ratio=6),
        )

        # Head: global pool and classify
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, num_bits),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Extract message logits from image.

        Args:
            image: (B, 3, 400, 400) in [0, 1]

        Returns:
            Message logits (B, num_bits)
        """
        # Normalize to [-0.5, 0.5]
        x = image - 0.5

        x = self.stem(x)
        x = self.blocks(x)
        logits: Tensor = self.head(x)

        return logits

    def decode(self, image: Tensor) -> Tensor:
        """Extract binary message from image.

        Args:
            image: (B, 3, 400, 400) in [0, 1]

        Returns:
            Binary message (B, num_bits)
        """
        logits = self.forward(image)
        probs = torch.sigmoid(logits)
        return (probs > 0.5).float()
