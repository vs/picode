"""Picodeine encoder: StegaStamp spatial backbone + AdaIN modulation.

Combines the proven StegaStamp spatial message injection (input-level concat +
output skip) with AdaIN modulation at 4 decoder layers for additional fine-grained
control. The spatial backbone ensures all U-Net features are message-dependent
from the first layer, preventing the training collapse that occurs with
decoder-only injection (bottleneck/output skip).

Message injection:
1. **Input spatial** (3ch at H): Message expanded to (3, 50, 50), upsampled to
   full resolution, concatenated with image → 6ch input. All encoder features
   (c1-c5) carry message information. Matches StegaStamp.
2. **Output skip** (6ch at H): The 6ch input (image + message) is concatenated
   at the final decoder stage, giving the message a direct 2-layer path to the
   output residual. Matches StegaStamp's conv9 inputs skip.
3. **AdaIN at 4 decoder layers** (H/8 → H): Picodeine-specific fine-grained
   per-pixel modulation via mapping network. Zero-init at start (identity),
   learns additional implicit modulation during training.

Key differences from StegaStamp:
- AdaIN at 4 decoder layers for implicit modulation
- MappingNetwork transforms 127 bits → 256-dim latent w
- No BatchNorm (AdaIN replaces it in decoder path)
- Supports variable image sizes (divisible by 16, not fixed 400x400)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Encoder as BaseEncoder
from picode.models.picodeine.adain import AdaIN, MappingNetwork


class Encoder(BaseEncoder):
    """Picodeine encoder — U-Net with input spatial + AdaIN modulation.

    Architecture:
    - Spatial message: 127 bits → Linear(7500) → (3, 50, 50) → upsample to (3, H, W)
    - Input: concat [image(3), message_spatial(3)] → 6ch
    - MappingNetwork: 127 bits → 256-dim latent w (for AdaIN)
    - U-Net encoder: 5 levels, 6ch input (all features are message-dependent)
    - U-Net decoder: 4 levels with AdaIN(w) at each
    - Output stage: concat [c1(32), x(32), inputs(6)] → 70ch (StegaStamp skip)
    - Output: image + residual (no clamping during training)

    Attributes:
        num_bits: Number of message bits to encode (default: 127).
    """

    def __init__(self, num_bits: int = 127, mapping_dim: int = 256) -> None:
        super().__init__()
        self.num_bits = num_bits

        # Spatial message expansion: bits → (3, 50, 50) → upsample to (3, H, W)
        self.secret_dense = nn.Linear(num_bits, 7500)

        # Message mapping network: bits → latent w (for AdaIN)
        self.mapping = MappingNetwork(num_bits=num_bits, mapping_dim=mapping_dim)

        # U-Net encoder (6ch input: 3 image + 3 message spatial)
        self.conv1 = nn.Conv2d(6, 32, 3, padding=1)               # H
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)    # H/2
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)    # H/4
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)   # H/8
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)  # H/16 (bottleneck)

        # U-Net decoder with skip connections
        self.up5 = nn.Conv2d(256, 128, 2, padding=0)
        self.conv5d = nn.Conv2d(256, 128, 3, padding=1)

        self.up4 = nn.Conv2d(128, 64, 2, padding=0)
        self.conv4d = nn.Conv2d(128, 64, 3, padding=1)

        self.up3 = nn.Conv2d(64, 32, 2, padding=0)
        self.conv3d = nn.Conv2d(64, 32, 3, padding=1)

        self.up2 = nn.Conv2d(32, 32, 2, padding=0)
        # After up2: concat c1 (32) + x (32) + inputs (6) = 70 channels
        # The inputs skip gives message a direct 2-layer path to output
        self.conv2d = nn.Conv2d(70, 32, 3, padding=1)

        # AdaIN at each decoder layer
        self.adain1 = AdaIN(mapping_dim, 128)  # After conv5d
        self.adain2 = AdaIN(mapping_dim, 64)   # After conv4d
        self.adain3 = AdaIN(mapping_dim, 32)   # After conv3d
        self.adain4 = AdaIN(mapping_dim, 32)   # After conv2d

        # Output: direct residual
        self.residual = nn.Conv2d(32, 3, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal, zero-init AdaIN projections.

        AdaIN projections start at zero (identity modulation). The input-level
        spatial injection ensures all features are message-dependent from the
        start. AdaIN gradually learns additional modulation during training.
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
        # Input spatial provides message signal; AdaIN refines later.
        for m in self.modules():
            if isinstance(m, AdaIN):
                nn.init.zeros_(m.projection.weight)
                nn.init.zeros_(m.projection.bias)

    def prepare_message(self, message_norm: Tensor, target_h: int, target_w: int) -> Tensor:
        """Expand message bits to spatial feature map.

        Args:
            message_norm: (B, num_bits) normalized message (centered at 0).
            target_h: Target height.
            target_w: Target width.

        Returns:
            (B, 3, target_h, target_w) spatial message tensor.
        """
        x = F.relu(self.secret_dense(message_norm))  # (B, 7500)
        x = x.view(-1, 3, 50, 50)                    # (B, 3, 50, 50)
        x = F.interpolate(x, size=(target_h, target_w), mode="nearest")
        return x

    def forward(self, image: Tensor, message: Tensor) -> Tensor:
        """Encode message into image using input spatial + AdaIN injection.

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

        # Prepare message spatial and concatenate with image (StegaStamp pattern)
        secret_enlarged = self.prepare_message(message_norm, img_h, img_w)
        inputs = torch.cat([secret_enlarged, image_norm], dim=1)  # (B, 6, H, W)

        # Map message to latent w (for AdaIN)
        w = self.mapping(message_norm)  # (B, mapping_dim)

        # Encoder path — all features are message-dependent
        c1 = F.relu(self.conv1(inputs))  # (B, 32, H, H)
        c2 = F.relu(self.conv2(c1))      # (B, 32, H/2)
        c3 = F.relu(self.conv3(c2))      # (B, 64, H/4)
        c4 = F.relu(self.conv4(c3))      # (B, 128, H/8)
        c5 = F.relu(self.conv5(c4))      # (B, 256, H/16) bottleneck

        # Decoder path with skip connections + AdaIN

        # up5: bottleneck → concat with c4 → AdaIN
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

        # up2: concat with c1 + inputs skip → AdaIN
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up2(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c1, x, inputs], dim=1)  # (B, 70, H, W)
        x = F.relu(self.conv2d(x))
        x = self.adain4(x, w)

        # Residual output
        residual = self.residual(x)
        encoded = image + residual
        return encoded
