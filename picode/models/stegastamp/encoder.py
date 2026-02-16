"""StegaStamp encoder network.

Matches original TensorFlow implementation exactly:
- No BatchNorm
- Input normalization (subtract 0.5)
- Raw residual output
- He normal weight initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Encoder as BaseEncoder


class Encoder(BaseEncoder):
    """U-Net encoder that embeds a bit message into an image.

    Architecture matches original StegaStamp TensorFlow implementation.

    Args:
        num_bits: Number of bits in the message (default: 100).
    """

    def __init__(self, num_bits: int = 100) -> None:
        super().__init__()
        self.num_bits = num_bits

        # Message preparation: num_bits -> 7500 -> (50, 50, 3) -> upsample to (400, 400, 3)
        self.secret_dense = nn.Linear(num_bits, 7500)

        # Encoder (downsampling path) - no BatchNorm
        self.conv1 = nn.Conv2d(6, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        # Decoder (upsampling path) - no BatchNorm
        self.up6 = nn.Conv2d(256, 128, 2, padding=0)
        self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.up7 = nn.Conv2d(128, 64, 2, padding=0)
        self.conv7 = nn.Conv2d(128, 64, 3, padding=1)
        self.up8 = nn.Conv2d(64, 32, 2, padding=0)
        self.conv8 = nn.Conv2d(64, 32, 3, padding=1)
        self.up9 = nn.Conv2d(32, 32, 2, padding=0)
        self.conv9 = nn.Conv2d(70, 32, 3, padding=1)  # 32 + 32 + 6 = 70

        # Output layers
        self.conv10 = nn.Conv2d(32, 32, 3, padding=1)
        self.residual = nn.Conv2d(32, 3, 1)

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

    def prepare_message(self, message: Tensor) -> Tensor:
        """Expand message bits to spatial feature map.

        Args:
            message: (B, num_bits) binary tensor (already normalized to [-0.5, 0.5])

        Returns:
            (B, 3, 400, 400) spatial tensor
        """
        x = F.relu(self.secret_dense(message))  # (B, 7500)
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
        # Normalize inputs (match original TF implementation)
        image_norm = image - 0.5
        message_norm = message - 0.5

        # Prepare message and concatenate with image
        secret_enlarged = self.prepare_message(message_norm)
        inputs = torch.cat([secret_enlarged, image_norm], dim=1)  # (B, 6, 400, 400)

        # Encoder path (save activations for skip connections)
        c1 = F.relu(self.conv1(inputs))  # (B, 32, 400, 400)
        c2 = F.relu(self.conv2(c1))  # (B, 32, 200, 200)
        c3 = F.relu(self.conv3(c2))  # (B, 64, 100, 100)
        c4 = F.relu(self.conv4(c3))  # (B, 128, 50, 50)
        c5 = F.relu(self.conv5(c4))  # (B, 256, 25, 25)

        # Decoder path with skip connections
        # up6: upsample -> conv -> concat with c4
        x = F.interpolate(c5, scale_factor=2, mode="nearest")
        x = F.relu(self.up6(F.pad(x, (0, 1, 0, 1))))  # Pad to handle 2x2 conv
        x = torch.cat([c4, x], dim=1)  # (B, 256, 50, 50)
        x = F.relu(self.conv6(x))  # (B, 128, 50, 50)

        # up7: upsample -> conv -> concat with c3
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up7(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c3, x], dim=1)  # (B, 128, 100, 100)
        x = F.relu(self.conv7(x))  # (B, 64, 100, 100)

        # up8: upsample -> conv -> concat with c2
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up8(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c2, x], dim=1)  # (B, 64, 200, 200)
        x = F.relu(self.conv8(x))  # (B, 32, 200, 200)

        # up9: upsample -> conv -> concat with c1 AND inputs (original skip)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up9(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c1, x, inputs], dim=1)  # (B, 70, 400, 400)
        x = F.relu(self.conv9(x))  # (B, 32, 400, 400)

        # Output layers - raw residual (no activation on final layer)
        x = F.relu(self.conv10(x))
        residual = self.residual(x)  # (B, 3, 400, 400)

        # Add residual to original image and clamp to valid range
        encoded = image + residual
        encoded = torch.clamp(encoded, 0, 1)
        return encoded
