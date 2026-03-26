"""Detection module for watermark detection in images and video."""

from picode.detection.confidence import compute_confidence
from picode.detection.detector import Detector
from picode.detection.fast_detector import FastDetector, FastDetectorModel
from picode.detection.rectifier import Rectifier
from picode.detection.types import Detection, Point, Quadrilateral
from picode.detection.window import Window, WindowGenerator

__all__ = [
    # Types
    "Detection",
    "Point",
    "Quadrilateral",
    # Existing (slow) detector
    "compute_confidence",
    "Detector",
    "Window",
    "WindowGenerator",
    # Fast detector (new)
    "FastDetector",
    "FastDetectorModel",
    # Utilities
    "Rectifier",
]

# Backward compatibility alias
SlowDetector = Detector
