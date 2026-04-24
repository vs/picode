"""PicodeLite encoder network.

Key differences from StegaStamp encoder:
- 800x800 encoder resolution (vs 400x400)
- 6-level U-Net (vs 5-level)
- Learned upsampling via transposed convolutions (vs nearest-neighbor)
- 63 bits (vs 100 bits)
- No BatchNorm

Architecture:
- Message preparation: Linear -> reshape -> 4x TransposedConv2d -> bilinear upsample to 800x800
- Encoder path: 6 conv layers with stride 2 downsampling (800->400->200->100->50->25)
- Decoder path: 6 upsample+conv layers with skip connections
- Output: image + residual (no clamping)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Encoder as BaseEncoder


class Encoder(BaseEncoder):
    """U-Net encoder that embeds a bit message into an image.

    Architecture optimized for reduced visual artifacts:
    - Learned upsampling (transposed convolutions) for smooth message expansion
    - Higher resolution (800x800) for finer detail preservation
    - 6-level U-Net for deeper feature extraction

    Args:
        num_bits: Number of bits in the message (default: 63 for PicodeLite).
    """

    def __init__(self, num_bits: int = 63) -> None:
        super().__init__()
        self.num_bits = num_bits

        # Message preparation with learned upsampling:
        # 63 bits -> Linear(63, 48*16*16) -> reshape to (48, 16, 16)
        # -> TransposedConv 48->32 (4x4, stride 2) -> 32x32
        # -> TransposedConv 32->16 (4x4, stride 2) -> 64x64
        # -> TransposedConv 16->8 (4x4, stride 2) -> 128x128
        # -> TransposedConv 8->3 (4x4, stride 2) -> 256x256
        # -> Bilinear upsample to 800x800
        self.secret_dense = nn.Linear(num_bits, 48 * 16 * 16)
        self.secret_up1 = nn.ConvTranspose2d(48, 32, 4, stride=2, padding=1)  # 16->32
        self.secret_up2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)  # 32->64
        self.secret_up3 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)   # 64->128
        self.secret_up4 = nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1)    # 128->256

        # 6-level U-Net encoder path (800->400->200->100->50->25)
        # conv1: 6->32 (input = 3 channels image + 3 channels message)
        self.conv1 = nn.Conv2d(6, 32, 3, padding=1)
        # conv2: 32->32, stride 2 (800->400)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # conv3: 32->64, stride 2 (400->200)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        # conv4: 64->128, stride 2 (200->100)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        # conv5: 128->256, stride 2 (100->50)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        # conv6: 256->256, stride 2 (50->25) - bottleneck
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)

        # 6-level U-Net decoder path (25->800)
        # up6: 256->256 (2x2 conv after upsample), concat with c5 -> conv6d: 512->256
        self.up6 = nn.Conv2d(256, 256, 2, padding=0)
        self.conv6d = nn.Conv2d(512, 256, 3, padding=1)

        # up5: 256->128, concat with c4 -> conv5d: 256->128
        self.up5 = nn.Conv2d(256, 128, 2, padding=0)
        self.conv5d = nn.Conv2d(256, 128, 3, padding=1)

        # up4: 128->64, concat with c3 -> conv4d: 128->64
        self.up4 = nn.Conv2d(128, 64, 2, padding=0)
        self.conv4d = nn.Conv2d(128, 64, 3, padding=1)

        # up3: 64->32, concat with c2 -> conv3d: 64->32
        self.up3 = nn.Conv2d(64, 32, 2, padding=0)
        self.conv3d = nn.Conv2d(64, 32, 3, padding=1)

        # up2: 32->32, concat with c1 AND inputs -> conv2d: 70->32
        # 32 (from up2) + 32 (from c1) + 6 (from inputs) = 70
        self.up2 = nn.Conv2d(32, 32, 2, padding=0)
        self.conv2d = nn.Conv2d(70, 32, 3, padding=1)

        # Output layers: conv_out: 32->32, residual: 32->3
        self.conv_out = nn.Conv2d(32, 32, 3, padding=1)
        self.residual = nn.Conv2d(32, 3, 1)

        # Initialize weights (Kaiming normal, no BatchNorm)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal (He normal)."""
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
            image: (B, 3, H, W) in [0, 1] - typically 800x800 but size is derived from input
            message: (B, num_bits) binary tensor

        Returns:
            Encoded image (B, 3, H, W) same size as input - NOT clamped to allow gradient flow
        """
        # Normalize inputs (match original TF implementation)
        image_norm = image - 0.5
        message_norm = message - 0.5

        # Derive target size from input image (no hardcoded dimensions)
        target_size = (image.shape[2], image.shape[3])

        # Prepare message and concatenate with image
        secret_enlarged = self.prepare_message(message_norm, target_size)  # (B, 3, H, W)
        inputs = torch.cat([secret_enlarged, image_norm], dim=1)  # (B, 6, H, W)

        # Encoder path (save activations for skip connections)
        c1 = F.relu(self.conv1(inputs))  # (B, 32, 800, 800)
        c2 = F.relu(self.conv2(c1))      # (B, 32, 400, 400)
        c3 = F.relu(self.conv3(c2))      # (B, 64, 200, 200)
        c4 = F.relu(self.conv4(c3))      # (B, 128, 100, 100)
        c5 = F.relu(self.conv5(c4))      # (B, 256, 50, 50)
        c6 = F.relu(self.conv6(c5))      # (B, 256, 25, 25) - bottleneck

        # Decoder path with skip connections

        # up6: upsample c6 -> conv -> concat with c5
        x = F.interpolate(c6, scale_factor=2, mode="nearest")  # (B, 256, 50, 50)
        x = F.relu(self.up6(F.pad(x, (0, 1, 0, 1))))  # Pad to handle 2x2 conv
        x = torch.cat([c5, x], dim=1)  # (B, 512, 50, 50)
        x = F.relu(self.conv6d(x))     # (B, 256, 50, 50)

        # up5: upsample -> conv -> concat with c4
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # (B, 256, 100, 100)
        x = F.relu(self.up5(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c4, x], dim=1)  # (B, 256, 100, 100)
        x = F.relu(self.conv5d(x))     # (B, 128, 100, 100)

        # up4: upsample -> conv -> concat with c3
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # (B, 128, 200, 200)
        x = F.relu(self.up4(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c3, x], dim=1)  # (B, 128, 200, 200)
        x = F.relu(self.conv4d(x))     # (B, 64, 200, 200)

        # up3: upsample -> conv -> concat with c2
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # (B, 64, 400, 400)
        x = F.relu(self.up3(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c2, x], dim=1)  # (B, 64, 400, 400)
        x = F.relu(self.conv3d(x))     # (B, 32, 400, 400)

        # up2: upsample -> conv -> concat with c1 AND inputs (original skip)
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # (B, 32, 800, 800)
        x = F.relu(self.up2(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c1, x, inputs], dim=1)  # (B, 70, 800, 800)
        x = F.relu(self.conv2d(x))     # (B, 32, 800, 800)

        # Output layers - raw residual (no activation on final layer)
        x = F.relu(self.conv_out(x))   # (B, 32, 800, 800)
        residual = self.residual(x)    # (B, 3, 800, 800)

        # Add residual to original image
        # IMPORTANT: Do NOT clamp - this allows gradients to flow freely
        # and the encoder can temporarily overshoot during training.
        # L2/LPIPS losses naturally penalize out-of-range values.
        encoded: Tensor = image + residual
        return encoded
