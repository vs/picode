"""PicodeFrame encoder network.

Generates a message-bearing frame around an image while preserving
the original center pixels exactly. Based on the StegaStamp U-Net
architecture with a hard mask to enforce zero modification of the
original image.

Key differences from StegaStamp encoder:
- 7 input channels (3 image + 1 mask + 3 message) instead of 6
- Hard mask output: center pixels are never modified
- Frame width is variable (passed as parameter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Encoder as BaseEncoder


class Encoder(BaseEncoder):
    """U-Net encoder that generates a message-bearing frame around an image.

    The encoder takes a reflection-padded image, a binary mask indicating
    the center region, and a message. It generates a natural-looking frame
    via outpainting while encoding the message into the frame pixels.
    A hard mask guarantees the original center pixels are never modified.

    Args:
        num_bits: Number of bits in the message (default: 96).
    """

    def __init__(self, num_bits: int = 96) -> None:
        super().__init__()
        self.num_bits = num_bits

        # Message preparation: num_bits -> 7500 -> (3, 50, 50) -> upsample to (400, 400)
        self.secret_dense = nn.Linear(num_bits, 7500)

        # Encoder (downsampling path) - no BatchNorm
        # 7 input channels: 3 (padded image) + 1 (mask) + 3 (message spatial)
        self.conv1 = nn.Conv2d(7, 32, 3, padding=1)
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
        self.conv9 = nn.Conv2d(71, 32, 3, padding=1)  # 32 + 32 + 7 = 71

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

    def _create_mask(self, image: Tensor, frame_width: int) -> Tensor:
        """Create a binary mask: 1 in center, 0 in frame border.

        Args:
            image: Image tensor to derive shape and device from (B, C, H, W).
            frame_width: Width of the frame border in pixels.

        Returns:
            Binary mask (B, 1, H, W) with 1 in center, 0 in border.
        """
        B, _, H, W = image.shape
        mask = torch.zeros(B, 1, H, W, device=image.device, dtype=image.dtype)
        fw = frame_width
        mask[:, :, fw:H - fw, fw:W - fw] = 1.0
        return mask

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

    def forward(self, image: Tensor, message: Tensor, **kwargs: int) -> Tensor:
        """Encode message into a frame around the image.

        Args:
            image: (B, 3, 400, 400) reflection-padded image in [0, 1].
            message: (B, num_bits) binary tensor.
            **kwargs: Must include frame_width (int) specifying the border width in pixels.

        Returns:
            Framed image (B, 3, 400, 400) with original center preserved.
        """
        frame_width: int = kwargs.get("frame_width", 16)

        # Normalize inputs (match original TF implementation)
        image_norm = image - 0.5
        message_norm = message - 0.5

        # Create mask (1 in center, 0 in border)
        mask = self._create_mask(image, frame_width)

        # Prepare message and concatenate with image and mask
        secret_enlarged = self.prepare_message(message_norm)
        inputs = torch.cat([image_norm, mask, secret_enlarged], dim=1)  # (B, 7, 400, 400)

        # Encoder path (save activations for skip connections)
        c1 = F.relu(self.conv1(inputs))  # (B, 32, 400, 400)
        c2 = F.relu(self.conv2(c1))  # (B, 32, 200, 200)
        c3 = F.relu(self.conv3(c2))  # (B, 64, 100, 100)
        c4 = F.relu(self.conv4(c3))  # (B, 128, 50, 50)
        c5 = F.relu(self.conv5(c4))  # (B, 256, 25, 25)

        # Decoder path with skip connections
        x = F.interpolate(c5, scale_factor=2, mode="nearest")
        x = F.relu(self.up6(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c4, x], dim=1)  # (B, 256, 50, 50)
        x = F.relu(self.conv6(x))  # (B, 128, 50, 50)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up7(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c3, x], dim=1)  # (B, 128, 100, 100)
        x = F.relu(self.conv7(x))  # (B, 64, 100, 100)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up8(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c2, x], dim=1)  # (B, 64, 200, 200)
        x = F.relu(self.conv8(x))  # (B, 32, 200, 200)

        # up9: skip with c1 AND inputs (original skip)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up9(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c1, x, inputs], dim=1)  # (B, 71, 400, 400)
        x = F.relu(self.conv9(x))  # (B, 32, 400, 400)

        # Output layers - raw residual
        _ = F.relu(self.conv10(x))  # Computed but not used (matches StegaStamp)
        residual = self.residual(x)  # (B, 3, 400, 400)

        # Add residual to padded image
        full_output = image + residual

        # Hard mask: preserve original center, use generated output in border
        framed = image * mask + full_output * (1 - mask)

        return framed
