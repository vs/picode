"""Picode encoder with improved gradient flow.

Key improvements over StegaStamp:
- GroupNorm after each convolution
- LeakyReLU instead of ReLU
- Same U-Net structure with skip connections (already good)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Encoder as BaseEncoder


class Encoder(BaseEncoder):
    """U-Net encoder with GroupNorm and LeakyReLU.

    Embeds a bit message into an image using a U-Net architecture.
    Improvements over StegaStamp: GroupNorm + LeakyReLU for stable gradients.

    Args:
        num_bits: Number of bits in the message (default: 100).
    """

    def __init__(self, num_bits: int = 100) -> None:
        super().__init__()
        self.num_bits = num_bits

        # Message preparation: num_bits -> 7500 -> (3, 50, 50) -> upsample to (3, 400, 400)
        self.secret_dense = nn.Linear(num_bits, 7500)

        # Encoder (downsampling path) with GroupNorm
        self.conv1 = nn.Conv2d(6, 32, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, 32)

        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, 32)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(8, 64)

        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.norm4 = nn.GroupNorm(8, 128)

        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.norm5 = nn.GroupNorm(8, 256)

        # Decoder (upsampling path) with GroupNorm
        self.up6 = nn.Conv2d(256, 128, 2, padding=0)
        self.norm6a = nn.GroupNorm(8, 128)
        self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.norm6b = nn.GroupNorm(8, 128)

        self.up7 = nn.Conv2d(128, 64, 2, padding=0)
        self.norm7a = nn.GroupNorm(8, 64)
        self.conv7 = nn.Conv2d(128, 64, 3, padding=1)
        self.norm7b = nn.GroupNorm(8, 64)

        self.up8 = nn.Conv2d(64, 32, 2, padding=0)
        self.norm8a = nn.GroupNorm(8, 32)
        self.conv8 = nn.Conv2d(64, 32, 3, padding=1)
        self.norm8b = nn.GroupNorm(8, 32)

        self.up9 = nn.Conv2d(32, 32, 2, padding=0)
        self.norm9a = nn.GroupNorm(8, 32)
        self.conv9 = nn.Conv2d(70, 32, 3, padding=1)  # 32 + 32 + 6 = 70
        self.norm9b = nn.GroupNorm(8, 32)

        # Output layers
        self.conv10 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm10 = nn.GroupNorm(8, 32)
        self.residual = nn.Conv2d(32, 3, 1)

        # Activation
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal for LeakyReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def prepare_message(self, message: Tensor) -> Tensor:
        """Expand message bits to spatial feature map.

        Args:
            message: (B, num_bits) normalized to [-0.5, 0.5]

        Returns:
            (B, 3, 400, 400) spatial tensor
        """
        x = self.act(self.secret_dense(message))  # (B, 7500)
        x = x.view(-1, 3, 50, 50)  # (B, 3, 50, 50)
        x = F.interpolate(x, scale_factor=8, mode="nearest")  # (B, 3, 400, 400)
        return x

    def forward(self, image: Tensor, message: Tensor) -> Tensor:
        """Encode message into image.

        Args:
            image: (B, 3, 400, 400) in [0, 1]
            message: (B, num_bits) binary tensor

        Returns:
            Encoded image (B, 3, 400, 400) in [0, 1]
        """
        # Normalize inputs
        image_norm = image - 0.5
        message_norm = message - 0.5

        # Prepare message and concatenate with image
        secret_enlarged = self.prepare_message(message_norm)
        inputs = torch.cat([secret_enlarged, image_norm], dim=1)  # (B, 6, 400, 400)

        # Encoder path (save activations for skip connections)
        c1 = self.act(self.norm1(self.conv1(inputs)))  # (B, 32, 400, 400)
        c2 = self.act(self.norm2(self.conv2(c1)))      # (B, 32, 200, 200)
        c3 = self.act(self.norm3(self.conv3(c2)))      # (B, 64, 100, 100)
        c4 = self.act(self.norm4(self.conv4(c3)))      # (B, 128, 50, 50)
        c5 = self.act(self.norm5(self.conv5(c4)))      # (B, 256, 25, 25)

        # Decoder path with skip connections
        x = F.interpolate(c5, scale_factor=2, mode="nearest")
        x = self.act(self.norm6a(self.up6(F.pad(x, (0, 1, 0, 1)))))
        x = torch.cat([c4, x], dim=1)
        x = self.act(self.norm6b(self.conv6(x)))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.act(self.norm7a(self.up7(F.pad(x, (0, 1, 0, 1)))))
        x = torch.cat([c3, x], dim=1)
        x = self.act(self.norm7b(self.conv7(x)))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.act(self.norm8a(self.up8(F.pad(x, (0, 1, 0, 1)))))
        x = torch.cat([c2, x], dim=1)
        x = self.act(self.norm8b(self.conv8(x)))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.act(self.norm9a(self.up9(F.pad(x, (0, 1, 0, 1)))))
        x = torch.cat([c1, x, inputs], dim=1)  # (B, 70, 400, 400)
        x = self.act(self.norm9b(self.conv9(x)))

        # Output layers
        x = self.act(self.norm10(self.conv10(x)))
        residual = self.residual(x)  # No activation on final layer

        # Add residual and clamp
        encoded = image + residual
        return torch.clamp(encoded, 0, 1)
