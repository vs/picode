"""Picodeine encoder with AdaIN message injection + spatial message paths.

Uses a hybrid message injection strategy:
1. **Bottleneck spatial** (32ch at H/16): Low-res message expansion concatenated
   at the U-Net bottleneck. Provides coarse message structure in deep features.
2. **Output spatial skip** (32ch at H): Full-res message concatenated at the final
   decoder stage, giving the message a direct 2-layer path to the output residual.
   This bootstraps decoder learning (analogous to StegaStamp's inputs skip at conv9).
3. **AdaIN at 4 decoder layers** (H/8 → H): Fine-grained per-pixel modulation
   that hides the message imperceptibly once training progresses.

Key differences from PicodeLite/StegaStamp:
- 3-channel input (image only, no message spatial concat at input)
- MappingNetwork transforms 127 bits → 256-dim latent w
- AdaIN at 4 decoder layers for fine-grained modulation
- Spatial message at bottleneck + output skip for training bootstrap
- No BatchNorm (AdaIN replaces it in decoder path)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Encoder as BaseEncoder
from picode.models.picodeine.adain import AdaIN, MappingNetwork


class Encoder(BaseEncoder):
    """Picodeine encoder — U-Net with AdaIN + spatial message skip.

    Architecture:
    - MappingNetwork: 127 bits → 256-dim latent w (for AdaIN)
    - Spatial message: 127 bits → Linear → (32, 8, 8) → interpolate to target sizes
    - U-Net encoder: 5 levels, 3ch image input (no message concat at input)
    - Bottleneck: concat spatial message (32ch) with c5 (256ch) → 288ch
    - U-Net decoder: 4 levels with AdaIN(w) at each
    - Output stage: concat spatial message (32ch) at full res for direct message→output path
    - Output: image + residual (no clamping during training)

    Attributes:
        num_bits: Number of message bits to encode (default: 127).
        secret_channels: Number of channels for spatial message expansion (32).
    """

    def __init__(self, num_bits: int = 127, mapping_dim: int = 256) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.secret_channels = 32

        # Message mapping network: bits → latent w (for AdaIN)
        self.mapping = MappingNetwork(num_bits=num_bits, mapping_dim=mapping_dim)

        # Spatial message expansion: bits → low-res spatial tensor at bottleneck
        self.secret_dense = nn.Linear(num_bits, self.secret_channels * 8 * 8)

        # U-Net encoder (3ch image input — no message concat)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)               # H
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)    # H/2
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)    # H/4
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)   # H/8
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)  # H/16 (bottleneck)

        # U-Net decoder with skip connections
        # up5 takes 256 (c5) + 32 (spatial message) = 288 channels
        self.up5 = nn.Conv2d(256 + self.secret_channels, 128, 2, padding=0)
        self.conv5d = nn.Conv2d(256, 128, 3, padding=1)

        self.up4 = nn.Conv2d(128, 64, 2, padding=0)
        self.conv4d = nn.Conv2d(128, 64, 3, padding=1)

        self.up3 = nn.Conv2d(64, 32, 2, padding=0)
        self.conv3d = nn.Conv2d(64, 32, 3, padding=1)

        self.up2 = nn.Conv2d(32, 32, 2, padding=0)
        # After up2: concat c1 (32) + x (32) + msg_fullres (32) = 96 channels
        # The msg_fullres skip gives message a direct 2-layer path to output
        self.conv2d = nn.Conv2d(64 + self.secret_channels, 32, 3, padding=1)

        # AdaIN at each decoder layer (skip bottleneck)
        self.adain1 = AdaIN(mapping_dim, 128)  # After conv5d
        self.adain2 = AdaIN(mapping_dim, 64)   # After conv4d
        self.adain3 = AdaIN(mapping_dim, 32)   # After conv3d
        self.adain4 = AdaIN(mapping_dim, 32)   # After conv2d

        # Output: direct residual
        self.residual = nn.Conv2d(32, 3, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal, zero-init AdaIN projections.

        AdaIN projections start at zero (identity modulation) — the bottleneck
        spatial injection provides enough message signal to bootstrap decoder
        learning. AdaIN gradually learns fine-grained modulation during training.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init AdaIN projections: starts as identity modulation.
        # Bottleneck spatial injection bootstraps training; AdaIN refines later.
        for m in self.modules():
            if isinstance(m, AdaIN):
                nn.init.zeros_(m.projection.weight)
                nn.init.zeros_(m.projection.bias)

    def forward(self, image: Tensor, message: Tensor) -> Tensor:
        """Encode message into image using spatial skip + bottleneck + AdaIN injection.

        Args:
            image: (B, 3, H, W) in [0, 1] — must be divisible by 16.
            message: (B, num_bits) binary tensor.

        Returns:
            Encoded image (B, 3, H, W) — NOT clamped to allow gradient flow.

        Raises:
            ValueError: If image dimensions are not divisible by 16.
        """
        img_h, img_w = image.shape[2], image.shape[3]
        if img_h % 16 != 0 or img_w % 16 != 0:
            raise ValueError(
                f"Picodeine encoder requires image dimensions divisible by 16 "
                f"(for U-Net skip connections). Got {img_h}x{img_w}."
            )

        # Normalize inputs
        image_norm = image - 0.5
        message_norm = message - 0.5

        # Map message to latent w (for AdaIN)
        w = self.mapping(message_norm)  # (B, mapping_dim)

        # Encoder path
        c1 = F.relu(self.conv1(image_norm))  # (B, 32, H, H)
        c2 = F.relu(self.conv2(c1))          # (B, 32, H/2)
        c3 = F.relu(self.conv3(c2))          # (B, 64, H/4)
        c4 = F.relu(self.conv4(c3))          # (B, 128, H/8)
        c5 = F.relu(self.conv5(c4))          # (B, 256, H/16) bottleneck

        # Spatial message: compute once at base resolution, interpolate for each level
        msg_base = F.relu(self.secret_dense(message_norm))  # (B, 32*8*8)
        msg_base = msg_base.view(-1, self.secret_channels, 8, 8)  # (B, 32, 8, 8)

        # Bottleneck injection (H/16)
        msg_bottleneck = F.interpolate(
            msg_base, size=(c5.shape[2], c5.shape[3]),
            mode="bilinear", align_corners=False,
        )
        c5 = torch.cat([c5, msg_bottleneck], dim=1)  # (B, 288, H/16, W/16)

        # Full-res skip for output stage (H)
        msg_fullres = F.interpolate(
            msg_base, size=(img_h, img_w), mode="bilinear", align_corners=False,
        )

        # Decoder path with skip connections + AdaIN

        # up5: bottleneck (288ch) → concat with c4 → AdaIN
        x = F.interpolate(c5, scale_factor=2, mode="nearest")
        x = F.relu(self.up5(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c4, x], dim=1)
        x = F.relu(self.conv5d(x))
        x = self.adain1(x, w)

        # up4: concat with c3 → AdaIN
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up4(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c3, x], dim=1)
        x = F.relu(self.conv4d(x))
        x = self.adain2(x, w)

        # up3: concat with c2 → AdaIN
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up3(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c2, x], dim=1)
        x = F.relu(self.conv3d(x))
        x = self.adain3(x, w)

        # up2: concat with c1 + msg_fullres skip → AdaIN
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up2(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c1, x, msg_fullres], dim=1)  # (B, 96, H, W)
        x = F.relu(self.conv2d(x))
        x = self.adain4(x, w)

        # Residual output
        residual = self.residual(x)
        encoded = image + residual
        return encoded
