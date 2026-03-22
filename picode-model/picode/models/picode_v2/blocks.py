"""Building blocks for picode_v2 mobile-optimized model."""

import torch.nn as nn
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
