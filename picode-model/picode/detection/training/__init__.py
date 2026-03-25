"""Training infrastructure for detection models."""

from picode.detection.training.dataset import DetectionDataset
from picode.detection.training.loss import DetectionLoss

__all__ = ["DetectionDataset", "DetectionLoss"]
