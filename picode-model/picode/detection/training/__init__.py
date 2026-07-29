"""Training infrastructure for detection models."""

from picode.detection.training.augmentation import (
    DetectionAugmentation,
    GeometricAugmentation,
    PhotometricAugmentation,
)
from picode.detection.training.dataset import DetectionDataset
from picode.detection.training.evaluator import (
    DetectionEvaluator,
    DetectionMetrics,
    compute_iou,
)
from picode.detection.training.hard_negative import (
    HardNegativeDataset,
    HardNegativeTransform,
)
from picode.detection.training.loss import DetectionLoss
from picode.detection.training.pregenerated_dataset import PregeneratedDetectionDataset
from picode.detection.training.trainer import DetectionTrainer

__all__ = [
    # Augmentation
    "DetectionAugmentation",
    "GeometricAugmentation",
    "PhotometricAugmentation",
    # Dataset
    "DetectionDataset",
    "PregeneratedDetectionDataset",
    "HardNegativeDataset",
    "HardNegativeTransform",
    # Loss
    "DetectionLoss",
    # Training
    "DetectionTrainer",
    # Evaluation
    "DetectionEvaluator",
    "DetectionMetrics",
    "compute_iou",
]
