"""Picode_v2 encoder with artifact reduction.

Key features:
- MessageExpander with bilinear upsampling (no checkerboard)
- Content-adaptive residual scaling (hide in textures)
- tanh-bounded residual (controlled perturbation)
- Bilinear upsampling throughout U-Net decoder path
- GroupNorm + LeakyReLU (server-side, not mobile)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.base import Encoder as BaseEncoder
from picode.models.picode_v2.blocks import MessageExpander


def compute_activity_map(image: Tensor, kernel_size: int = 3) -> Tensor:
    """Compute local activity (edge magnitude) for content-adaptive scaling.

    Higher values in textured/edge regions, lower in smooth regions.

    Args:
        image: (B, 3, H, W) input image.
        kernel_size: Size of the Sobel-like filter window.

    Returns:
        (B, 1, H, W) activity map normalized to [0, 1].
    """
    # Convert to grayscale
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]

    # Sobel filters for edge detection
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=image.dtype, device=image.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=image.dtype, device=image.device
    ).view(1, 1, 3, 3)

    # Compute gradients
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)

    # Edge magnitude
    activity = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

    # Normalize to [0, 1] per image
    b = activity.shape[0]
    activity_flat = activity.view(b, -1)
    min_vals = activity_flat.min(dim=1, keepdim=True).values.view(b, 1, 1, 1)
    max_vals = activity_flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1)
    activity = (activity - min_vals) / (max_vals - min_vals + 1e-8)

    return activity


class Encoder(BaseEncoder):
    """U-Net encoder with artifact reduction.

    Embeds a bit message into an image using a U-Net architecture.
    Features content-adaptive residual scaling and tanh-bounded output.

    Args:
        num_bits: Number of bits in the message (default: 100).
        residual_scale: Maximum residual magnitude (default: 0.1).
    """

    def __init__(self, num_bits: int = 100, residual_scale: float = 0.1) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.residual_scale = residual_scale

        # Message expansion using learned upsampling
        self.message_expander = MessageExpander(num_bits=num_bits)

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

        # Decoder (upsampling path) with GroupNorm and bilinear upsampling
        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
        )
        self.norm6a = nn.GroupNorm(8, 128)
        self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.norm6b = nn.GroupNorm(8, 128)

        self.up7 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
        )
        self.norm7a = nn.GroupNorm(8, 64)
        self.conv7 = nn.Conv2d(128, 64, 3, padding=1)
        self.norm7b = nn.GroupNorm(8, 64)

        self.up8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
        )
        self.norm8a = nn.GroupNorm(8, 32)
        self.conv8 = nn.Conv2d(64, 32, 3, padding=1)
        self.norm8b = nn.GroupNorm(8, 32)

        self.up9 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 32, 3, padding=1),
        )
        self.norm9a = nn.GroupNorm(8, 32)
        self.conv9 = nn.Conv2d(70, 32, 3, padding=1)  # 32 + 32 + 6 = 70
        self.norm9b = nn.GroupNorm(8, 32)

        # Output layers
        self.conv10 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm10 = nn.GroupNorm(8, 32)
        self.residual_conv = nn.Conv2d(32, 3, 1)

        # Activation
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal for LeakyReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.2
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.2
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

        # Expand message using learned upsampling
        secret_enlarged = self.message_expander(message_norm)
        inputs = torch.cat([secret_enlarged, image_norm], dim=1)  # (B, 6, 400, 400)

        # Encoder path (save activations for skip connections)
        c1 = self.act(self.norm1(self.conv1(inputs)))  # (B, 32, 400, 400)
        c2 = self.act(self.norm2(self.conv2(c1)))      # (B, 32, 200, 200)
        c3 = self.act(self.norm3(self.conv3(c2)))      # (B, 64, 100, 100)
        c4 = self.act(self.norm4(self.conv4(c3)))      # (B, 128, 50, 50)
        c5 = self.act(self.norm5(self.conv5(c4)))      # (B, 256, 25, 25)

        # Decoder path with skip connections and bilinear upsampling
        x = self.act(self.norm6a(self.up6(c5)))
        x = torch.cat([c4, x], dim=1)
        x = self.act(self.norm6b(self.conv6(x)))

        x = self.act(self.norm7a(self.up7(x)))
        x = torch.cat([c3, x], dim=1)
        x = self.act(self.norm7b(self.conv7(x)))

        x = self.act(self.norm8a(self.up8(x)))
        x = torch.cat([c2, x], dim=1)
        x = self.act(self.norm8b(self.conv8(x)))

        x = self.act(self.norm9a(self.up9(x)))
        x = torch.cat([c1, x, inputs], dim=1)  # (B, 70, 400, 400)
        x = self.act(self.norm9b(self.conv9(x)))

        # Output layers
        x = self.act(self.norm10(self.conv10(x)))
        raw_residual = self.residual_conv(x)

        # Apply tanh for bounded residual
        bounded_residual = torch.tanh(raw_residual) * self.residual_scale

        # Content-adaptive scaling: larger residuals in textured regions
        activity_map = compute_activity_map(image)
        # Scale from [0.3, 1.0] based on activity (never fully zero)
        scaling = 0.3 + 0.7 * activity_map
        adaptive_residual = bounded_residual * scaling

        # Add residual and clamp
        encoded = image + adaptive_residual
        return torch.clamp(encoded, 0, 1)
