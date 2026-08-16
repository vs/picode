"""PicodeLite encoder network.

Simplified encoder aligned with StegaStamp's proven patterns:
- Single Linear → ReLU → nearest-neighbor upsample for message preparation
- 5-level U-Net (matching StegaStamp depth)
- Direct residual output (no extra ReLU gate)
- No BatchNorm (matches StegaStamp)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Encoder as BaseEncoder


class Encoder(BaseEncoder):
    """PicodeLite encoder - aligned with StegaStamp's proven architecture.

    Architecture:
    - Message preparation: Linear → ReLU → reshape(3, 32, 32) → nearest upsample
    - U-Net: 5 levels with skip connections (requires input divisible by 16)

    Attributes:
        num_bits: Number of message bits to encode (default: 63 for BCH(63,36)).
    """

    def __init__(self, num_bits: int = 63) -> None:
        """Initialize encoder.

        Args:
            num_bits: Number of message bits to encode.
        """
        super().__init__()
        self.num_bits = num_bits

        # Message preparation: Linear → ReLU → reshape → nearest upsample
        # Single ReLU (~50% paths survive, matching StegaStamp)
        self.secret_dense = nn.Linear(num_bits, 3 * 32 * 32)

        # U-Net encoder (5 levels)
        # Input: 6 channels (3 image + 3 message)
        self.conv1 = nn.Conv2d(6, 32, 3, padding=1)              # H
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)   # H/2
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)   # H/4
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # H/8
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1) # H/16 (bottleneck)

        # U-Net decoder with skip connections
        self.up5 = nn.Conv2d(256, 128, 2, padding=0)
        self.conv5d = nn.Conv2d(256, 128, 3, padding=1)

        self.up4 = nn.Conv2d(128, 64, 2, padding=0)
        self.conv4d = nn.Conv2d(128, 64, 3, padding=1)

        self.up3 = nn.Conv2d(64, 32, 2, padding=0)
        self.conv3d = nn.Conv2d(64, 32, 3, padding=1)

        self.up2 = nn.Conv2d(32, 32, 2, padding=0)
        # After up2: concat with c1 (32) + inputs (6) = 70 channels
        self.conv2d = nn.Conv2d(70, 32, 3, padding=1)

        # Output: direct residual (no extra conv/ReLU gate, matching StegaStamp)
        self.residual = nn.Conv2d(32, 3, 1)

        # Initialize weights (no BatchNorm to init)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def prepare_message(self, message: Tensor, target_size: tuple[int, int]) -> Tensor:
        """Expand message bits to spatial feature map using nearest-neighbor upsample.

        Args:
            message: (B, num_bits) binary tensor (already normalized to [-0.5, 0.5])
            target_size: (H, W) target spatial dimensions to match input image

        Returns:
            (B, 3, H, W) spatial tensor matching target_size
        """
        x = F.relu(self.secret_dense(message))  # (B, 3*32*32)
        x = x.view(-1, 3, 32, 32)  # (B, 3, 32, 32)
        x = F.interpolate(x, size=target_size, mode="nearest")  # (B, 3, H, W)
        return x

    def forward(self, image: Tensor, message: Tensor) -> Tensor:
        """Encode message into image.

        Args:
            image: (B, 3, H, W) in [0, 1] - must be divisible by 16 for U-Net skip connections
            message: (B, num_bits) binary tensor

        Returns:
            Encoded image (B, 3, H, W) same size as input - NOT clamped to allow gradient flow

        Raises:
            ValueError: If image dimensions are not divisible by 16.
        """
        h, w = image.shape[2], image.shape[3]
        if h % 16 != 0 or w % 16 != 0:
            raise ValueError(
                f"PicodeLite encoder requires image dimensions divisible by 16 "
                f"(for U-Net skip connections). Got {h}\u00d7{w}. "
                f"Recommended: 512\u00d7512 (set model.encoder_size: 512 in config)."
            )

        # Normalize inputs (match original TF implementation)
        image_norm = image - 0.5
        message_norm = message - 0.5

        # Derive target size from input image
        target_size = (h, w)

        # Prepare message and concatenate with image
        secret_enlarged = self.prepare_message(message_norm, target_size)  # (B, 3, H, W)
        inputs = torch.cat([secret_enlarged, image_norm], dim=1)  # (B, 6, H, W)

        # Encoder path (save activations for skip connections)
        c1 = F.relu(self.conv1(inputs))  # (B, 32, H, H)
        c2 = F.relu(self.conv2(c1))      # (B, 32, H/2, H/2)
        c3 = F.relu(self.conv3(c2))      # (B, 64, H/4, H/4)
        c4 = F.relu(self.conv4(c3))      # (B, 128, H/8, H/8)
        c5 = F.relu(self.conv5(c4))      # (B, 256, H/16, H/16) - bottleneck

        # Decoder path with skip connections

        # up5: upsample c5 -> conv -> concat with c4
        x = F.interpolate(c5, scale_factor=2, mode="nearest")  # (B, 256, H/8, H/8)
        x = F.relu(self.up5(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c4, x], dim=1)  # (B, 256, H/8, H/8)
        x = F.relu(self.conv5d(x))

        # up4: upsample -> conv -> concat with c3
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up4(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c3, x], dim=1)
        x = F.relu(self.conv4d(x))

        # up3: upsample -> conv -> concat with c2
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up3(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c2, x], dim=1)
        x = F.relu(self.conv3d(x))

        # up2: upsample -> conv -> concat with c1 and inputs
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up2(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c1, x, inputs], dim=1)  # 32 + 32 + 6 = 70
        x = F.relu(self.conv2d(x))

        # Direct residual output (no extra ReLU gate, matching StegaStamp)
        residual = self.residual(x)

        # Add residual to original (no clamping during training)
        encoded = image + residual
        result: Tensor = encoded
        return result
