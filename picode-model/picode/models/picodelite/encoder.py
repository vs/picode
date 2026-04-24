"""PicodeLite encoder network.

800x800 encoder with learned upsampling for reduced visual artifacts:
- Transposed convolutions for smooth message spatial expansion
- 6-level U-Net for high-resolution encoding
- No BatchNorm (matches StegaStamp)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Encoder as BaseEncoder


class Encoder(BaseEncoder):
    """PicodeLite encoder - high resolution with learned upsampling.

    Architecture optimized for 800x800 images with smooth message embedding:
    - Message preparation: Linear -> TransposedConv x4 -> Bilinear upsample
    - U-Net: 6 levels with skip connections for fine detail preservation

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

        # Message preparation network (learned upsampling)
        # 63 bits -> 48*16*16 -> 32x32 -> 64x64 -> 128x128 -> 256x256 -> bilinear -> HxW
        self.secret_dense = nn.Linear(num_bits, 48 * 16 * 16)

        # Transposed convolutions for smooth upsampling
        self.secret_up1 = nn.ConvTranspose2d(48, 32, 2, stride=2)  # 16->32
        self.secret_up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)  # 32->64
        self.secret_up3 = nn.ConvTranspose2d(16, 8, 2, stride=2)   # 64->128
        self.secret_up4 = nn.ConvTranspose2d(8, 3, 2, stride=2)    # 128->256

        # U-Net encoder (6 levels for 800x800)
        # Input: 6 channels (3 image + 3 message)
        self.conv1 = nn.Conv2d(6, 32, 3, padding=1)       # 800 -> 800
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)   # 800 -> 400
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)   # 400 -> 200
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 200 -> 100
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1) # 100 -> 50
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1) # 50 -> 25

        # U-Net decoder with skip connections
        self.up6 = nn.Conv2d(256, 256, 2, padding=0)  # After upsample
        self.conv6d = nn.Conv2d(512, 256, 3, padding=1)

        self.up5 = nn.Conv2d(256, 128, 2, padding=0)
        self.conv5d = nn.Conv2d(256, 128, 3, padding=1)

        self.up4 = nn.Conv2d(128, 64, 2, padding=0)
        self.conv4d = nn.Conv2d(128, 64, 3, padding=1)

        self.up3 = nn.Conv2d(64, 32, 2, padding=0)
        self.conv3d = nn.Conv2d(64, 32, 3, padding=1)

        self.up2 = nn.Conv2d(32, 32, 2, padding=0)
        # After up2: concat with c1 (32) + inputs (6) = 70 channels
        self.conv2d = nn.Conv2d(70, 32, 3, padding=1)

        # Output layers
        self.conv_out = nn.Conv2d(32, 32, 3, padding=1)
        self.residual = nn.Conv2d(32, 3, 1)

        # Initialize weights (no BatchNorm to init)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def prepare_message(self, message: Tensor, target_size: tuple[int, int]) -> Tensor:
        """Expand message bits to spatial feature map using learned upsampling.

        Args:
            message: (B, num_bits) binary tensor (already normalized to [-0.5, 0.5])
            target_size: (H, W) target spatial dimensions to match input image

        Returns:
            (B, 3, H, W) spatial tensor matching target_size
        """
        # Linear projection and reshape to spatial
        x = F.relu(self.secret_dense(message))  # (B, 48*16*16)
        x = x.view(-1, 48, 16, 16)  # (B, 48, 16, 16)

        # Learned upsampling via transposed convolutions
        x = F.relu(self.secret_up1(x))  # (B, 32, 32, 32)
        x = F.relu(self.secret_up2(x))  # (B, 16, 64, 64)
        x = F.relu(self.secret_up3(x))  # (B, 8, 128, 128)
        x = F.relu(self.secret_up4(x))  # (B, 3, 256, 256)

        # Final bilinear upsample to target size (derived from input image)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        return x

    def forward(self, image: Tensor, message: Tensor) -> Tensor:
        """Encode message into image.

        Args:
            image: (B, 3, H, W) in [0, 1] - must be divisible by 32 for U-Net skip connections
            message: (B, num_bits) binary tensor

        Returns:
            Encoded image (B, 3, H, W) same size as input - NOT clamped to allow gradient flow

        Raises:
            ValueError: If image dimensions are not divisible by 32.
        """
        h, w = image.shape[2], image.shape[3]
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError(
                f"PicodeLite encoder requires image dimensions divisible by 32 "
                f"(for U-Net skip connections). Got {h}×{w}. "
                f"Recommended: 800×800 (set model.encoder_size: 800 in config)."
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
        c5 = F.relu(self.conv5(c4))      # (B, 256, H/16, H/16)
        c6 = F.relu(self.conv6(c5))      # (B, 256, H/32, H/32) - bottleneck

        # Decoder path with skip connections

        # up6: upsample c6 -> conv -> concat with c5
        x = F.interpolate(c6, scale_factor=2, mode="nearest")  # (B, 256, H/16, H/16)
        x = F.relu(self.up6(F.pad(x, (0, 1, 0, 1))))  # Pad to handle 2x2 conv
        x = torch.cat([c5, x], dim=1)  # (B, 512, H/16, H/16)
        x = F.relu(self.conv6d(x))

        # up5: upsample -> conv -> concat with c4
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up5(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c4, x], dim=1)
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

        # Output
        x = F.relu(self.conv_out(x))
        residual = self.residual(x)

        # Add residual to original (no clamping during training)
        encoded = image + residual
        return encoded
