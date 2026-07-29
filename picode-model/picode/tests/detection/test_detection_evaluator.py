# picode/tests/detection/test_detection_evaluator.py
"""Tests for detection evaluation metrics."""

import pytest
import torch

from picode.detection.fast_detector import FastDetectorModel
from picode.detection.training.evaluator import (
    DetectionEvaluator,
    DetectionMetrics,
    compute_iou,
)


class TestComputeIoU:
    def test_identical_boxes_iou_one(self) -> None:
        corners1 = torch.tensor([0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9])
        corners2 = torch.tensor([0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9])
        iou = compute_iou(corners1, corners2)
        assert iou == pytest.approx(1.0, abs=0.01)

    def test_no_overlap_iou_zero(self) -> None:
        corners1 = torch.tensor([0.0, 0.0, 0.3, 0.0, 0.3, 0.3, 0.0, 0.3])
        corners2 = torch.tensor([0.7, 0.7, 1.0, 0.7, 1.0, 1.0, 0.7, 1.0])
        iou = compute_iou(corners1, corners2)
        assert iou == pytest.approx(0.0, abs=0.01)

    def test_partial_overlap(self) -> None:
        corners1 = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5])
        corners2 = torch.tensor([0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75])
        iou = compute_iou(corners1, corners2)
        # Should be between 0 and 1
        assert 0.0 < iou < 1.0


class TestDetectionMetrics:
    def test_metrics_creation(self) -> None:
        metrics = DetectionMetrics(
            precision=0.95,
            recall=0.90,
            f1=0.925,
            accuracy=0.93,
            mean_iou=0.85,
        )
        assert metrics.precision == 0.95
        assert metrics.recall == 0.90
        assert metrics.f1 == 0.925

    def test_metrics_to_dict(self) -> None:
        metrics = DetectionMetrics(
            precision=0.95,
            recall=0.90,
            f1=0.925,
            accuracy=0.93,
            mean_iou=0.85,
        )
        d = metrics.to_dict()

        assert "precision" in d
        assert "recall" in d
        assert "f1" in d
        assert d["precision"] == 0.95


class TestDetectionEvaluator:
    @pytest.fixture
    def model(self) -> FastDetectorModel:
        return FastDetectorModel(input_size=320, pretrained=False)

    @pytest.fixture
    def evaluator(self, model: FastDetectorModel) -> DetectionEvaluator:
        return DetectionEvaluator(model=model, threshold=0.5, device="cpu")

    def test_evaluator_creation(self, evaluator: DetectionEvaluator) -> None:
        assert evaluator.threshold == 0.5
        assert evaluator.device == "cpu"

    def test_evaluate_batch_returns_predictions(
        self, evaluator: DetectionEvaluator
    ) -> None:
        batch = {
            "image": torch.rand(4, 3, 320, 320),
            "is_watermark": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "corners": torch.rand(4, 8),
            "has_corners": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        }

        predictions, targets = evaluator.evaluate_batch(batch)

        assert "pred_cls" in predictions
        assert "pred_corners" in predictions
        assert len(predictions["pred_cls"]) == 4

    def test_compute_metrics_returns_metrics(
        self, evaluator: DetectionEvaluator
    ) -> None:
        # Simulate some predictions
        predictions = {
            "pred_cls": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "pred_corners": torch.rand(4, 8),
        }
        targets = {
            "is_watermark": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "corners": torch.rand(4, 8),
            "has_corners": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        }

        metrics = evaluator.compute_metrics(predictions, targets)

        assert isinstance(metrics, DetectionMetrics)
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0

    def test_evaluate_dataset_runs(self, evaluator: DetectionEvaluator) -> None:
        from torch.utils.data import DataLoader

        # Create simple dataset
        images = torch.rand(8, 3, 320, 320)
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        corners = torch.rand(8, 8)
        has_corners = labels.clone()

        class SimpleDataset:
            def __init__(self):
                pass

            def __len__(self):
                return 8

            def __getitem__(self, idx):
                return {
                    "image": images[idx],
                    "is_watermark": labels[idx],
                    "corners": corners[idx],
                    "has_corners": has_corners[idx],
                }

        loader = DataLoader(SimpleDataset(), batch_size=2)
        metrics = evaluator.evaluate_dataset(loader)

        assert isinstance(metrics, DetectionMetrics)
