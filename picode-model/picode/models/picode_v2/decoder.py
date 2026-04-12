"""StegaStamp-style decoder for picode_v2.

Uses standard convolutions (not depthwise separable) for reliable training.
Key insight: steganography requires aggregating message bits distributed
globally across the image - standard convolutions handle this better than
depthwise separable convolutions.

Architecture based on StegaStamp (Tancik et al., CVPR 2020) without STN:
- Standard 3x3 convolutions
- No BatchNorm (training stability)
- ReLU activations
- Flatten + FC head (proven to train)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Decoder as BaseDecoder


class Decoder(BaseDecoder):
    """StegaStamp-style decoder without STN.

    Proven architecture that trains reliably for steganography.
    Geometric correction handled by detection pipeline.

    Args:
        num_bits: Number of bits in the message (default: 100).
    """

    def __init__(self, num_bits: int = 100) -> None:
        super().__init__()
        self.num_bits = num_bits

        # Conv backbone - standard 3x3 convolutions with stride-2 downsampling
        # Input: (B, 3, 400, 400) -> Output: (B, 128, 13, 13)
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)    # -> 200x200
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)   # -> 200x200
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)   # -> 100x100
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)   # -> 100x100
        self.conv5 = nn.Conv2d(64, 64, 3, stride=2, padding=1)   # -> 50x50
        self.conv6 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # -> 25x25
        self.conv7 = nn.Conv2d(128, 128, 3, stride=2, padding=1) # -> 13x13

        # FC head - flatten approach (matches StegaStamp)
        # 128 * 13 * 13 = 21632
        self.fc1 = nn.Linear(128 * 13 * 13, 512)
        self.fc2 = nn.Linear(512, num_bits)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image: Tensor) -> Tensor:
        """Extract message logits from image.

        Args:
            image: (B, 3, 400, 400) in [0, 1]

        Returns:
            Message logits (B, num_bits)
        """
        # Normalize to [-0.5, 0.5]
        x = image - 0.5

        # Conv backbone (no BatchNorm, just ReLU)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        # FC head
        x = x.flatten(1)  # (B, 21632)
        x = F.relu(self.fc1(x))
        logits: Tensor = self.fc2(x)

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


# Keep MobileDecoder as alias for backwards compatibility during transition
MobileDecoder = Decoder
