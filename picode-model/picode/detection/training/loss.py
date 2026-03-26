# picode/detection/training/loss.py
"""Loss functions for FastDetector training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DetectionLoss(nn.Module):
    """Combined loss for classification + corner regression.

    Args:
        cls_weight: Weight for classification loss
        corner_weight: Weight for corner regression loss
        conf_weight: Weight for confidence loss

    Example:
        >>> loss_fn = DetectionLoss(cls_weight=1.0, corner_weight=5.0)
        >>> pred = {"is_watermark": logits, "corners": corners, "corner_confidence": conf}
        >>> target = {"is_watermark": labels, "corners": gt_corners, "has_corners": mask}
        >>> losses = loss_fn(pred, target)
        >>> losses["total"].backward()
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        corner_weight: float = 5.0,
        conf_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.corner_weight = corner_weight
        self.conf_weight = conf_weight

        self.cls_loss = nn.BCEWithLogitsLoss()
        self.corner_loss = nn.SmoothL1Loss(reduction="none")

    def forward(
        self,
        pred: dict[str, Tensor],
        target: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute combined loss.

        Args:
            pred: Model predictions with keys:
                - is_watermark: (B, 1) classification logits
                - corners: (B, 8) predicted corners [0, 1]
                - corner_confidence: (B, 1) confidence [0, 1]
            target: Ground truth with keys:
                - is_watermark: (B,) binary labels
                - corners: (B, 8) ground truth corners
                - has_corners: (B,) mask for valid corners (positive samples)

        Returns:
            Dictionary with total, cls, corner, conf losses
        """
        device = pred["is_watermark"].device

        # Classification loss (all samples)
        cls_loss = self.cls_loss(
            pred["is_watermark"].squeeze(-1),
            target["is_watermark"].to(device),
        )

        # Corner loss (only positive samples with valid corners)
        has_corners = target["has_corners"].bool().to(device)

        if has_corners.any():
            corner_loss = self.corner_loss(
                pred["corners"][has_corners],
                target["corners"][has_corners].to(device),
            ).mean()

            # Confidence should reflect corner accuracy
            corner_errors = (
                pred["corners"][has_corners] - target["corners"][has_corners].to(device)
            ).abs().mean(dim=1)
            # High error -> low confidence target, low error -> high confidence target
            conf_target = (1 - corner_errors * 5).clamp(0, 1)
            conf_loss = F.mse_loss(
                pred["corner_confidence"][has_corners].squeeze(-1),
                conf_target,
            )
        else:
            corner_loss = torch.tensor(0.0, device=device)
            conf_loss = torch.tensor(0.0, device=device)

        # Combined loss
        total = (
            self.cls_weight * cls_loss
            + self.corner_weight * corner_loss
            + self.conf_weight * conf_loss
        )

        return {
            "total": total,
            "cls": cls_loss,
            "corner": corner_loss,
            "conf": conf_loss,
        }
