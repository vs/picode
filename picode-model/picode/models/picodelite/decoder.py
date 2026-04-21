"""PicodeLite decoder network.

Lightweight decoder optimized for mobile inference:
- No Spatial Transformer Network (STN)
- Global average pooling (resolution-independent)
- No BatchNorm
- Input normalization (subtract 0.5)
- Raw logits output (no sigmoid)
- He normal weight initialization
- ~640K parameters (< 1M for mobile)
"""

import torch
import torch.nn as nn
from torch import Tensor

from picode.models.base import Decoder as BaseDecoder


class Decoder(BaseDecoder):
    """CNN decoder with global pooling that extracts message bits from an encoded image.

    Unlike StegaStamp, this decoder has no STN for geometric correction.
    Uses global average pooling for resolution-independent inference.

    Args:
        num_bits: Number of bits in the message (default: 63 for BCH(63,36)).
    """

    def __init__(self, num_bits: int = 63) -> None:
        super().__init__()
        self.num_bits = num_bits

        # Main decoder CNN - 8 conv layers, no BatchNorm
        # Channel progression: 3 -> 32 -> 32 -> 64 -> 64 -> 128 -> 128 -> 128 -> 128
        # Reduced from original 256 channels to stay under 1M params (~640K target)
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)      # 160x160
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)     # 160x160
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)     # 80x80
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)     # 80x80
        self.conv5 = nn.Conv2d(64, 128, 3, stride=2, padding=1)    # 40x40
        self.conv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)   # 40x40
        self.conv7 = nn.Conv2d(128, 128, 3, stride=2, padding=1)   # 20x20
        self.conv8 = nn.Conv2d(128, 128, 3, stride=1, padding=1)   # 20x20

        # Global average pooling for resolution-independence
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # FC head: 128 -> 128 -> num_bits
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_bits)

        self.relu = nn.ReLU()

        # Initialize weights (He normal)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal (He normal)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image: Tensor) -> Tensor:
        """Extract message logits from image.

        Args:
            image: (B, 3, H, W) in [0, 1]

        Returns:
            Message logits (B, num_bits) - unbounded, apply sigmoid for probabilities
        """
        # Normalize input (match original StegaStamp convention)
        x = image - 0.5

        # Conv blocks: stride-2 + stride-1 pairs
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))

        # Global average pooling: (B, 128, H', W') -> (B, 128, 1, 1)
        x = self.global_pool(x)

        # Flatten: (B, 128, 1, 1) -> (B, 128)
        x = x.view(x.size(0), -1)

        # FC head
        x = self.relu(self.fc1(x))
        logits: Tensor = self.fc2(x)

        return logits

    def decode(self, image: Tensor) -> Tensor:
        """Extract binary message from an image.

        Args:
            image: Input image tensor (B, C, H, W) in [0, 1].

        Returns:
            Binary message tensor (B, num_bits).
        """
        logits = self.forward(image)
        probs = torch.sigmoid(logits)
        return (probs > 0.5).float()
