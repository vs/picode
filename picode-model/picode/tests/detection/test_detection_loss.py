# picode/tests/detection/test_detection_loss.py
"""Tests for detection loss functions."""

import pytest
import torch
from torch import Tensor

from picode.detection.training.loss import DetectionLoss


class TestDetectionLoss:
    @pytest.fixture
    def loss_fn(self) -> DetectionLoss:
        return DetectionLoss(cls_weight=1.0, corner_weight=5.0, conf_weight=0.5)

    @pytest.fixture
    def pred_positive(self) -> dict[str, Tensor]:
        """Predictions for positive samples (watermark detected, good corners)."""
        return {
            "is_watermark": torch.tensor([[2.0], [1.5]]),  # High logits
            "corners": torch.tensor(
                [[0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
                 [0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8]]
            ),
            "corner_confidence": torch.tensor([[0.9], [0.8]]),
        }

    @pytest.fixture
    def target_positive(self) -> dict[str, Tensor]:
        """Ground truth for positive samples."""
        return {
            "is_watermark": torch.tensor([1.0, 1.0]),
            "corners": torch.tensor(
                [[0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
                 [0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8]]
            ),
            "has_corners": torch.tensor([1.0, 1.0]),
        }

    def test_loss_creation(self, loss_fn: DetectionLoss) -> None:
        assert loss_fn.cls_weight == 1.0
        assert loss_fn.corner_weight == 5.0
        assert loss_fn.conf_weight == 0.5

    def test_loss_forward_returns_dict(
        self,
        loss_fn: DetectionLoss,
        pred_positive: dict[str, Tensor],
        target_positive: dict[str, Tensor],
    ) -> None:
        result = loss_fn(pred_positive, target_positive)

        assert "total" in result
        assert "cls" in result
        assert "corner" in result
        assert "conf" in result

    def test_loss_values_are_scalars(
        self,
        loss_fn: DetectionLoss,
        pred_positive: dict[str, Tensor],
        target_positive: dict[str, Tensor],
    ) -> None:
        result = loss_fn(pred_positive, target_positive)

        assert result["total"].dim() == 0
        assert result["cls"].dim() == 0
        assert result["corner"].dim() == 0

    def test_loss_positive_values(
        self,
        loss_fn: DetectionLoss,
        pred_positive: dict[str, Tensor],
        target_positive: dict[str, Tensor],
    ) -> None:
        result = loss_fn(pred_positive, target_positive)

        assert result["total"] >= 0
        assert result["cls"] >= 0
        assert result["corner"] >= 0

    def test_perfect_prediction_low_loss(self, loss_fn: DetectionLoss) -> None:
        """Perfect predictions should have very low loss."""
        pred = {
            "is_watermark": torch.tensor([[10.0]]),  # Very confident positive
            "corners": torch.tensor([[0.1, 0.2, 0.9, 0.2, 0.9, 0.8, 0.1, 0.8]]),
            "corner_confidence": torch.tensor([[0.95]]),
        }
        target = {
            "is_watermark": torch.tensor([1.0]),
            "corners": torch.tensor([[0.1, 0.2, 0.9, 0.2, 0.9, 0.8, 0.1, 0.8]]),
            "has_corners": torch.tensor([1.0]),
        }

        result = loss_fn(pred, target)

        # Corner loss should be near zero for perfect match
        assert result["corner"] < 0.01

    def test_negative_samples_no_corner_loss(self, loss_fn: DetectionLoss) -> None:
        """Negative samples should not contribute to corner loss."""
        pred = {
            "is_watermark": torch.tensor([[-2.0], [-1.5]]),
            "corners": torch.tensor(
                [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # Random corners
                 [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]]
            ),
            "corner_confidence": torch.tensor([[0.1], [0.1]]),
        }
        target = {
            "is_watermark": torch.tensor([0.0, 0.0]),
            "corners": torch.zeros(2, 8),  # Ignored
            "has_corners": torch.tensor([0.0, 0.0]),
        }

        result = loss_fn(pred, target)

        # Corner loss should be zero (no valid corners)
        assert result["corner"] == 0.0

    def test_gradient_flow(
        self,
        loss_fn: DetectionLoss,
        pred_positive: dict[str, Tensor],
        target_positive: dict[str, Tensor],
    ) -> None:
        # Make predictions require grad
        pred = {k: v.clone().requires_grad_(True) for k, v in pred_positive.items()}

        result = loss_fn(pred, target_positive)
        result["total"].backward()

        assert pred["is_watermark"].grad is not None
        assert pred["corners"].grad is not None

    def test_mixed_batch(self, loss_fn: DetectionLoss) -> None:
        """Batch with both positive and negative samples."""
        pred = {
            "is_watermark": torch.tensor([[2.0], [-2.0]]),  # 1 pos, 1 neg
            "corners": torch.tensor(
                [[0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
                 [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
            ),
            "corner_confidence": torch.tensor([[0.9], [0.1]]),
        }
        target = {
            "is_watermark": torch.tensor([1.0, 0.0]),
            "corners": torch.tensor(
                [[0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            ),
            "has_corners": torch.tensor([1.0, 0.0]),
        }

        result = loss_fn(pred, target)

        # Should compute without error
        assert result["total"].isfinite()
        assert result["cls"].isfinite()
        assert result["corner"].isfinite()
