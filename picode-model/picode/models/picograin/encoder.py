"""PicoGrain encoder: noise modulation with luminance-adaptive film grain.

Architecture:
- PicoTrust U-Net backbone (parameterized size)
- E_post refinement -> smooth 1-channel envelope
- Noise modulation: envelope * N(0,1) noise * luminance mask
- Softsign amplitude bound * strength
- Grayscale residual broadcast to RGB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Encoder as BaseEncoder


class Encoder(BaseEncoder):
    """U-Net encoder with noise modulation for grain-textured steganography.

    The U-Net produces a smooth spatial envelope. This envelope is multiplied
    by a random noise carrier and a luminance-derived mask to produce a
    grain-textured residual. The message is encoded in the envelope; the
    noise carrier provides the grain aesthetic.

    Args:
        num_bits: Number of bits in the message (default: 127 for BCH).
        image_size: Target image size (default: 512).
        strength: Residual amplitude bound (softsign scaling). Higher than
            PicoTrust since grain is intentionally visible (~0.05-0.15).
        lum_floor: Minimum luminance mask value for dark areas (default: 0.1).
        lum_gamma: Gamma curve for luminance mask (default: 0.8).
    """

    def __init__(
        self,
        num_bits: int = 127,
        image_size: int = 512,
        strength: float | None = None,
        lum_floor: float = 0.1,
        lum_gamma: float = 0.8,
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.image_size = image_size
        self.strength = strength
        self.lum_floor = lum_floor
        self.lum_gamma = lum_gamma

        # Message preparation: num_bits -> 7500 -> (3, 50, 50) -> upsample
        self.secret_dense = nn.Linear(num_bits, 7500)

        # Encoder (downsampling path) — no normalization
        self.conv1 = nn.Conv2d(6, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        # Decoder (upsampling path) — no normalization
        self.up6 = nn.Conv2d(256, 128, 2, padding=0)
        self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.up7 = nn.Conv2d(128, 64, 2, padding=0)
        self.conv7 = nn.Conv2d(128, 64, 3, padding=1)
        self.up8 = nn.Conv2d(64, 32, 2, padding=0)
        self.conv8 = nn.Conv2d(64, 32, 3, padding=1)
        self.up9 = nn.Conv2d(32, 32, 2, padding=0)
        self.conv9 = nn.Conv2d(70, 32, 3, padding=1)  # 32 + 32 + 6 = 70

        # E_post: outputs smooth 1-channel envelope
        self.e_post = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1),
            nn.SiLU(),
            nn.Conv2d(16, 1, 1),  # Smooth envelope (no activation)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal. Zero-init E_post final layer."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Zero-init so envelope starts at zero -> residual starts at zero
        last_conv = self.e_post[-1]
        assert isinstance(last_conv, nn.Conv2d)
        nn.init.zeros_(last_conv.weight)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

    def prepare_message(self, message: Tensor) -> Tensor:
        """Expand message bits to spatial feature map.

        Args:
            message: (B, num_bits) normalized to [-0.5, 0.5].

        Returns:
            (B, 3, image_size, image_size) spatial tensor.
        """
        x = F.relu(self.secret_dense(message))
        x = x.view(-1, 3, 50, 50)
        x = F.interpolate(
            x, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )
        return x

    def compute_luminance_mask(self, image: Tensor) -> Tensor:
        """Compute luminance-adaptive grain mask from cover image.

        Bright areas get more grain, dark areas get minimal grain (floor).

        Args:
            image: (B, 3, H, W) in [0, 1].

        Returns:
            (B, 1, H, W) mask in [floor, 1.0].
        """
        lum = (
            0.299 * image[:, 0:1]
            + 0.587 * image[:, 1:2]
            + 0.114 * image[:, 2:3]
        )
        mask = self.lum_floor + (1.0 - self.lum_floor) * lum.pow(self.lum_gamma)
        return mask

    def forward(self, image: Tensor, message: Tensor) -> dict[str, Tensor] | Tensor:
        """Encode message into image using noise-modulated grain.

        Args:
            image: (B, 3, H, W) in [0, 1].
            message: (B, num_bits) binary tensor.

        Returns:
            Dict with "encoded" (B, 3, H, W), "envelope" (B, 1, H, W),
            and "lum_mask" (B, 1, H, W).
        """
        image_norm = image - 0.5
        message_norm = message - 0.5

        secret_enlarged = self.prepare_message(message_norm)
        inputs = torch.cat([secret_enlarged, image_norm], dim=1)

        # U-Net encoder path
        c1 = F.relu(self.conv1(inputs))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        c4 = F.relu(self.conv4(c3))
        c5 = F.relu(self.conv5(c4))

        # U-Net decoder path with skip connections
        x = F.interpolate(c5, scale_factor=2, mode="nearest")
        x = F.relu(self.up6(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c4, x], dim=1)
        x = F.relu(self.conv6(x))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up7(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c3, x], dim=1)
        x = F.relu(self.conv7(x))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up8(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c2, x], dim=1)
        x = F.relu(self.conv8(x))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up9(F.pad(x, (0, 1, 0, 1))))
        x = torch.cat([c1, x, inputs], dim=1)
        x = F.relu(self.conv9(x))

        # E_post: smooth 1-channel envelope
        envelope = self.e_post(x)

        # Noise modulation
        noise = torch.randn_like(envelope)
        modulated = envelope * noise

        # Luminance mask
        lum_mask = self.compute_luminance_mask(image)
        masked = modulated * lum_mask

        # Softsign amplitude bound + strength scaling
        if self.strength is not None:
            residual_1ch = self.strength * masked / (1.0 + masked.abs())
        else:
            residual_1ch = masked

        # Broadcast to 3 channels
        residual = residual_1ch.expand(-1, 3, -1, -1)
        encoded: Tensor = image + residual

        if self.strength is not None:
            return {
                "encoded": encoded,
                "envelope": envelope,
                "lum_mask": lum_mask,
            }
        return encoded
