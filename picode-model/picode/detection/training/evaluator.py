# picode/detection/training/evaluator.py
"""Evaluation metrics for FastDetector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


def compute_iou(corners1: Tensor, corners2: Tensor) -> float:
    """Compute IoU between two axis-aligned bounding boxes from corners.

    Args:
        corners1: (8,) tensor of [x1,y1, x2,y2, x3,y3, x4,y4] corners
        corners2: (8,) tensor of corners

    Returns:
        Intersection over Union [0, 1]
    """
    # Extract bounding box from corners (axis-aligned)
    x1_1 = corners1[0::2].min().item()
    y1_1 = corners1[1::2].min().item()
    x2_1 = corners1[0::2].max().item()
    y2_1 = corners1[1::2].max().item()

    x1_2 = corners2[0::2].min().item()
    y1_2 = corners2[1::2].min().item()
    x2_2 = corners2[0::2].max().item()
    y2_2 = corners2[1::2].max().item()

    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)

    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


@dataclass
class DetectionMetrics:
    """Detection evaluation metrics.

    Attributes:
        precision: True positives / (True positives + False positives)
        recall: True positives / (True positives + False negatives)
        f1: 2 * precision * recall / (precision + recall)
        accuracy: (TP + TN) / total
        mean_iou: Mean IoU for positive samples
    """

    precision: float
    recall: float
    f1: float
    accuracy: float
    mean_iou: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "mean_iou": self.mean_iou,
        }


class DetectionEvaluator:
    """Evaluator for FastDetector model.

    Computes classification and localization metrics.

    Args:
        model: FastDetectorModel to evaluate
        threshold: Classification threshold
        iou_threshold: IoU threshold for correct localization
        device: Device to run evaluation on

    Example:
        >>> evaluator = DetectionEvaluator(model, threshold=0.5)
        >>> metrics = evaluator.evaluate_dataset(test_loader)
        >>> print(f"Precision: {metrics.precision:.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.5,
        iou_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.device = device

    def evaluate_batch(
        self, batch: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Evaluate a single batch.

        Args:
            batch: Dict with image, is_watermark, corners, has_corners

        Returns:
            Tuple of (predictions dict, targets dict)
        """
        images = batch["image"].to(self.device)

        with torch.no_grad():
            outputs = self.model(images)

        # Convert logits to predictions
        pred_probs = torch.sigmoid(outputs["is_watermark"]).squeeze(-1)
        pred_cls = (pred_probs >= self.threshold).float()

        predictions = {
            "pred_cls": pred_cls,
            "pred_probs": pred_probs,
            "pred_corners": outputs["corners"],
            "pred_conf": outputs["corner_confidence"].squeeze(-1),
        }

        targets = {
            "is_watermark": batch["is_watermark"],
            "corners": batch["corners"],
            "has_corners": batch["has_corners"],
        }

        return predictions, targets

    def compute_metrics(
        self,
        predictions: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> DetectionMetrics:
        """Compute metrics from predictions and targets.

        Args:
            predictions: Dict with pred_cls, pred_corners
            targets: Dict with is_watermark, corners, has_corners

        Returns:
            DetectionMetrics instance
        """
        pred_cls = predictions["pred_cls"]
        true_cls = targets["is_watermark"]

        # Classification metrics
        tp = ((pred_cls == 1) & (true_cls == 1)).sum().item()
        fp = ((pred_cls == 1) & (true_cls == 0)).sum().item()
        fn = ((pred_cls == 0) & (true_cls == 1)).sum().item()
        tn = ((pred_cls == 0) & (true_cls == 0)).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

        # IoU for positive samples
        has_corners = targets["has_corners"].bool()
        if has_corners.any():
            ious = []
            pred_corners = predictions["pred_corners"]
            true_corners = targets["corners"]

            for i in range(len(pred_corners)):
                if has_corners[i]:
                    iou = compute_iou(pred_corners[i], true_corners[i])
                    ious.append(iou)

            mean_iou = sum(ious) / len(ious) if ious else 0.0
        else:
            mean_iou = 0.0

        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            mean_iou=mean_iou,
        )

    def evaluate_dataset(self, loader: DataLoader[dict[str, Tensor]]) -> DetectionMetrics:
        """Evaluate on entire dataset.

        Args:
            loader: DataLoader for evaluation

        Returns:
            Aggregated DetectionMetrics
        """
        all_pred_cls = []
        all_pred_corners = []
        all_true_cls = []
        all_true_corners = []
        all_has_corners = []

        for batch in loader:
            predictions, targets = self.evaluate_batch(batch)

            all_pred_cls.append(predictions["pred_cls"])
            all_pred_corners.append(predictions["pred_corners"])
            all_true_cls.append(targets["is_watermark"])
            all_true_corners.append(targets["corners"])
            all_has_corners.append(targets["has_corners"])

        # Concatenate all
        aggregated_preds = {
            "pred_cls": torch.cat(all_pred_cls),
            "pred_corners": torch.cat(all_pred_corners),
        }
        aggregated_targets = {
            "is_watermark": torch.cat(all_true_cls),
            "corners": torch.cat(all_true_corners),
            "has_corners": torch.cat(all_has_corners),
        }

        return self.compute_metrics(aggregated_preds, aggregated_targets)
