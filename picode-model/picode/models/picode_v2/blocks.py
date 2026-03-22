"""Building blocks for picode_v2 mobile-optimized model."""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block.

    Structure: 1x1 expand -> 3x3 depthwise -> 1x1 project
    Uses BatchNorm + ReLU6 for mobile compatibility.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        stride: Stride for depthwise conv (1 or 2).
        expand_ratio: Expansion factor for hidden channels.
    """

    def __init__(
        self, in_ch: int, out_ch: int, stride: int, expand_ratio: int = 6
    ) -> None:
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_residual = stride == 1 and in_ch == out_ch

        layers: list[nn.Module] = []

        # Expand (1x1 conv) - skip if expand_ratio == 1
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ])

        # Depthwise (3x3 conv, groups=hidden)
        layers.extend([
            nn.Conv2d(
                hidden, hidden, 3, stride=stride, padding=1,
                groups=hidden, bias=False
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
        ])

        # Project (1x1 conv, linear - no activation)
        layers.extend([
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with optional residual connection."""
        out: Tensor = self.conv(x)
        if self.use_residual:
            out = x + out
        return out


class MessageExpander(nn.Module):
    """Progressive learned upsampling for message expansion.

    Uses Upsample + Conv2d instead of ConvTranspose2d to avoid
    checkerboard artifacts (Odena et al., "Deconvolution and Checkerboard Artifacts").

    Args:
        num_bits: Number of message bits.
        hidden_ch: Hidden channel dimension.
    """

    def __init__(self, num_bits: int = 100, hidden_ch: int = 64) -> None:
        super().__init__()
        self.dense = nn.Linear(num_bits, hidden_ch * 5 * 5)

        # Progressive upsampling: 5x5 -> 25x25 -> 100x100 -> 400x400
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=5, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.GroupNorm(8, hidden_ch),
            nn.LeakyReLU(0.2),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_ch, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.2),
        )
        self.refine = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, message: Tensor) -> Tensor:
        """Expand message to spatial feature map."""
        x = F.leaky_relu(self.dense(message), 0.2)
        x = x.view(-1, 64, 5, 5)
        x = self.up1(x)   # 5x5 -> 25x25
        x = self.up2(x)   # 25x25 -> 100x100
        x = self.up3(x)   # 100x100 -> 400x400
        out: Tensor = self.refine(x)
        return out
