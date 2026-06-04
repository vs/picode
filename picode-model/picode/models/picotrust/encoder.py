"""PicoTrust encoder: StegaStamp U-Net with TrustMark E_post refinement.

Architecture:
- StegaStamp U-Net backbone (parameterized size, default 256x256)
- E_post post-processing: Conv(32->32, 3x3) + ReLU -> Conv(32->16, 1x1) + SiLU -> Conv(16->3, 1x1)
- Residual encoding: encoded = image + residual
- No normalization layers (matches StegaStamp)
- Kaiming initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Encoder as BaseEncoder


class Encoder(BaseEncoder):
    """U-Net encoder with E_post refinement that embeds a bit message into an image.

    Based on StegaStamp encoder with TrustMark post-processing network (E_post).
    The E_post block replaces StegaStamp's single residual conv with a 3-layer
    refinement that improves PSNR by ~1.6 dB.

    Args:
        num_bits: Number of bits in the message (default: 100).
        image_size: Target image size (default: 256).
    """

    def __init__(
        self,
        num_bits: int = 100,
        image_size: int = 256,
        strength: float | None = None,
        use_mask: bool = False,
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.image_size = image_size
        self.strength = strength
        self.use_mask = use_mask

        # Message preparation: num_bits -> 7500 -> (3, 50, 50) -> upsample to image_size
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

        # E_post: TrustMark post-processing network
        # Replaces StegaStamp's single residual conv with a 3-layer refinement block
        # Outputs 1 channel (luminance-only) → broadcast to RGB = zero colour shift
        self.e_post = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),  # Spatial refinement
            nn.ReLU(),
            nn.Conv2d(32, 16, 1),             # Channel reduction
            nn.SiLU(),
            nn.Conv2d(16, 1, 1),              # Grayscale residual (no activation)
        )

        # Learned spatial mask (PicoTrust v2)
        self.mask_head: nn.Sequential | None = None
        if use_mask:
            self.mask_head = nn.Sequential(
                nn.Conv2d(32, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid(),
            )

        # Initialize weights (Kaiming normal)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal (He normal).

        Special case: E_post's final Conv2d (16→3) is zero-initialized so that
        tanh(0) = 0 and the residual starts at zero.  This keeps gradients in
        tanh's linear region and avoids the saturation / gradient-death problem
        that Kaiming init causes when ``strength * tanh(...)`` is used.
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

        # Zero-init E_post's last layer so residual starts at zero
        last_conv = self.e_post[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def prepare_message(self, message: Tensor) -> Tensor:
        """Expand message bits to spatial feature map.

        Args:
            message: (B, num_bits) binary tensor (already normalized to [-0.5, 0.5]).

        Returns:
            (B, 3, image_size, image_size) spatial tensor.
        """
        x = F.relu(self.secret_dense(message))  # (B, 7500)
        x = x.view(-1, 3, 50, 50)  # (B, 3, 50, 50)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="nearest")
        return x

    def forward(self, image: Tensor, message: Tensor) -> dict[str, Tensor] | Tensor:
        """Encode message into image.

        Args:
            image: (B, 3, H, W) in [0, 1].
            message: (B, num_bits) binary tensor.

        Returns:
            When strength is set: dict with "encoded" and optionally "mask".
            When strength is None: Encoded image tensor (B, 3, H, W) (backward compat).
        """
        # Normalize inputs (match StegaStamp)
        image_norm = image - 0.5
        message_norm = message - 0.5

        # Prepare message and concatenate with image
        secret_enlarged = self.prepare_message(message_norm)
        inputs = torch.cat([secret_enlarged, image_norm], dim=1)  # (B, 6, H, W)

        # Encoder path (save activations for skip connections)
        c1 = F.relu(self.conv1(inputs))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        c4 = F.relu(self.conv4(c3))
        c5 = F.relu(self.conv5(c4))

        # Decoder path with skip connections
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
        x = torch.cat([c1, x, inputs], dim=1)  # 32 + 32 + 6 = 70 (message skip)
        x = F.relu(self.conv9(x))

        # E_post: spatial refinement -> raw 1-channel residual
        raw_residual = self.e_post(x)  # (B, 1, H, W)

        # Broadcast to 3 channels (grayscale residual → no colour shift)
        raw_residual = raw_residual.expand(-1, 3, -1, -1)

        # Apply amplitude bound if configured
        if self.strength is not None:
            # Softsign-like bound: strength * x / (1 + |x|)
            # Unlike tanh, gradient 1/(1+|x|)^2 never reaches zero —
            # encoder always has signal to adjust spatial patterns.
            residual = self.strength * raw_residual / (1.0 + raw_residual.abs())
        else:
            residual = raw_residual

        # Apply spatial mask if enabled (independent of strength)
        mask: Tensor | None = None
        if self.mask_head is not None:
            mask = self.mask_head(x)  # (B, 1, H, W) in [0, 1]
            residual = residual * mask

        encoded = image + residual

        # Return dict if v2 features active, tensor for backward compat
        if mask is not None or self.strength is not None:
            result: dict[str, Tensor] = {"encoded": encoded}
            if mask is not None:
                result["mask"] = mask
            return result
        else:
            return encoded
