"""Picode decoder with improved gradient flow.

Key improvements over StegaStamp:
- ResNet-style skip connections after each downsample
- GroupNorm for gradient stability
- LeakyReLU to avoid dying neurons
- Fixed STN initialization (small random instead of zeros)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Decoder as BaseDecoder
from picode.models.picode.blocks import ResBlock


class Decoder(BaseDecoder):
    """CNN decoder with STN and improved gradient flow.

    Architecture improvements:
    - ResBlock after each downsample for skip connections
    - GroupNorm throughout for stable gradients
    - LeakyReLU instead of ReLU
    - STN weights initialized to small random values

    Args:
        num_bits: Number of bits in the message (default: 100).
    """

    def __init__(self, num_bits: int = 100) -> None:
        super().__init__()
        self.num_bits = num_bits

        # Spatial Transformer Network (STN)
        self.stn_params = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 50 * 50, 128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # STN output layer - small random init instead of zeros
        self.stn_fc = nn.Linear(128, 6)

        # Main decoder with ResBlocks
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res1 = ResBlock(32)

        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res2 = ResBlock(64)

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res3 = ResBlock(128)

        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Global pooling and output
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_bits)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with improved defaults."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # STN output: small random weights, identity bias
        nn.init.normal_(self.stn_fc.weight, mean=0, std=0.001)
        self.stn_fc.bias.data = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def forward(self, image: Tensor) -> Tensor:
        """Extract message logits from image.

        Args:
            image: (B, 3, 400, 400) in [0, 1]

        Returns:
            Message logits (B, num_bits)
        """
        # Normalize input
        x = image - 0.5

        # STN: predict affine transform
        stn_features = self.stn_params(x)
        theta = self.stn_fc(stn_features)
        theta = theta.view(-1, 2, 3)

        # Apply spatial transform
        grid = F.affine_grid(theta, list(x.size()), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, mode="bilinear", padding_mode="zeros")

        # Decoder with ResBlocks
        x = self.stem(x)  # (B, 32, 200, 200)
        x = self.res1(x)  # Skip connection
        x = self.down1(x)  # (B, 64, 100, 100)
        x = self.res2(x)  # Skip connection
        x = self.down2(x)  # (B, 128, 50, 50)
        x = self.res3(x)  # Skip connection
        x = self.down3(x)  # (B, 256, 25, 25)

        # Output
        x = self.pool(x)  # (B, 256, 1, 1)
        x = x.flatten(1)  # (B, 256)
        logits: Tensor = self.fc(x)  # (B, num_bits)

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
