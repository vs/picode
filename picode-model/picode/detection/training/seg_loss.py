# picode/detection/training/seg_loss.py
"""Segmentation loss for pixel-wise watermark region detection."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SegDetectionLoss(nn.Module):
    """Pixel-wise BCE loss for segmentation detector.

    The encoded watermark region is typically much smaller than the background,
    so a positive class weight compensates for this imbalance.

    Args:
        pos_weight: Weight applied to positive (encoded region) pixels.
            Higher values increase recall at the cost of precision.

    Example:
        >>> loss_fn = SegDetectionLoss(pos_weight=2.0)
        >>> pred = torch.rand(2, 1, 80, 80)   # model output (post-sigmoid)
        >>> target = torch.randint(0, 2, (2, 1, 320, 320)).float()
        >>> losses = loss_fn(pred, target)
        >>> losses["total"].backward()
    """

    def __init__(self, pos_weight: float = 2.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: Tensor, target: Tensor) -> dict[str, Tensor]:
        """Compute pixel-wise weighted BCE loss.

        Args:
            pred: Predicted mask (B, 1, H, W) in [0, 1] — already sigmoid.
            target: Ground-truth mask (B, 1, H', W') binary {0, 1}.
                If H' != H or W' != W the target is resized to match pred.

        Returns:
            Dictionary with keys:
                - ``total``: weighted BCE loss (scalar)
                - ``bce``: same value (alias for logging)
        """
        # Resize target to match prediction spatial size if needed
        if target.shape[-2:] != pred.shape[-2:]:
            target = F.interpolate(
                target,
                size=pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)

        # Build per-pixel weight map: positive pixels get pos_weight, negatives get 1.0
        weight = target * (self.pos_weight - 1.0) + 1.0  # (B, 1, H, W)

        # BCE expects inputs in [0, 1] (pred is already post-sigmoid)
        bce = F.binary_cross_entropy(pred, target, weight=weight)

        return {"total": bce, "bce": bce}
