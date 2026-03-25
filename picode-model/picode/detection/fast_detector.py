# picode/detection/fast_detector.py
"""FastDetector model for mobile-optimized watermark detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor


class FastDetectorModel(nn.Module):
    """Single-pass watermark detector with quadrilateral output.

    Architecture:
        - MobileNetV3-Small backbone (mobile-safe ops: BatchNorm, ReLU6)
        - Global average pooling
        - Three heads: classification, corner regression, confidence

    Args:
        input_size: Expected input image size (default 320)
        pretrained: Use ImageNet pretrained backbone

    Example:
        >>> model = FastDetectorModel(input_size=320)
        >>> x = torch.rand(1, 3, 320, 320)
        >>> output = model(x)
        >>> output["is_watermark"].shape
        torch.Size([1, 1])
        >>> output["corners"].shape
        torch.Size([1, 8])
    """

    def __init__(self, input_size: int = 320, pretrained: bool = True) -> None:
        super().__init__()
        self.input_size = input_size

        # MobileNetV3-Small backbone
        # Uses BatchNorm (fuses with conv) and ReLU6/HardSwish (hardware accelerated)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features  # Output: 576 channels

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classification head: is there a watermark?
        self.cls_head = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # Corner regression head: 4 corners x 2 coords = 8 values
        self.corner_head = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 8),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

        # Corner confidence head
        self.conf_head = nn.Sequential(
            nn.Linear(576, 64),
            nn.ReLU6(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Initialize heads
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize head weights."""
        for module in [self.cls_head, self.corner_head, self.conf_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Input images (B, 3, H, W) normalized to [0, 1]

        Returns:
            Dictionary with:
                - is_watermark: (B, 1) classification logits
                - corners: (B, 8) normalized corner coordinates [0, 1]
                - corner_confidence: (B, 1) confidence score [0, 1]
        """
        features = self.features(x)  # (B, 576, H/32, W/32)
        pooled = self.pool(features).flatten(1)  # (B, 576)

        return {
            "is_watermark": self.cls_head(pooled),
            "corners": self.corner_head(pooled),
            "corner_confidence": self.conf_head(pooled),
        }
