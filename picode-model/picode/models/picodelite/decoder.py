"""PicodeLite decoder network.

StegaStamp-style CNN decoder without STN:
- No Spatial Transformer Network (STN)
- Flatten + large FC (matches StegaStamp's proven approach)
- No BatchNorm
- Input normalization (subtract 0.5)
- Raw logits output (no sigmoid)
- He normal weight initialization
"""

import torch
import torch.nn as nn
from torch import Tensor

from picode.models.base import Decoder as BaseDecoder


class Decoder(BaseDecoder):
    """CNN decoder that extracts message bits from an encoded image.

    Matches the StegaStamp decoder architecture (without STN):
    7 conv layers with stride-2 downsampling, then Flatten + FC.
    This preserves spatial features that carry the hidden message.

    Args:
        num_bits: Number of bits in the message (default: 63 for BCH(63,36)).
        input_size: Expected input spatial dimension (default: 512).
    """

    def __init__(self, num_bits: int = 63, input_size: int = 512) -> None:
        super().__init__()
        self.num_bits = num_bits

        # Main decoder CNN - matches StegaStamp structure (no BatchNorm)
        # 5 stride-2 convs + 2 stride-1 convs = 7 total
        # For 512x512: 512 -> 256 -> 256 -> 128 -> 128 -> 64 -> 32 -> 16
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),    # /2
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),             # same
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # /2
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),             # same
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),   # /2
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), # /2
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size: 5 stride-2 convs reduce spatial by 2^5 = 32
        spatial = input_size // 32
        flatten_size = 128 * spatial * spatial

        # FC head (matches StegaStamp: large FC -> ReLU -> output)
        self.fc = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_bits),
        )

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
        x = image - 0.5
        x = self.decoder(x)
        logits: Tensor = self.fc(x)
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
