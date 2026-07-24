# picode/tests/detection/test_seg_detector.py
"""Tests for segmentation-based detector."""

from __future__ import annotations

import torch

from picode.detection.seg_detector import SegDetector, SegDetectorModel


class TestSegDetectorModel:
    def test_forward_shape(self) -> None:
        model = SegDetectorModel(input_size=320)
        x = torch.rand(2, 3, 320, 320)
        mask = model(x)
        assert mask.shape == (2, 1, 80, 80)

    def test_output_range(self) -> None:
        model = SegDetectorModel(input_size=320)
        x = torch.rand(1, 3, 320, 320)
        mask = model(x)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_different_input_sizes(self) -> None:
        model = SegDetectorModel(input_size=256)
        x = torch.rand(1, 3, 256, 256)
        mask = model(x)
        assert mask.shape == (1, 1, 64, 64)


class TestSegDetector:
    def test_extract_corners_from_mask(self) -> None:
        detector = SegDetector.__new__(SegDetector)
        detector.threshold = 0.5
        detector.min_area_ratio = 0.05
        mask = torch.zeros(1, 1, 80, 80)
        mask[0, 0, 10:70, 15:65] = 1.0
        corners = detector._extract_corners(mask[0, 0], original_size=(320, 320))
        assert corners is not None
        assert corners.shape == (4, 2)
        assert all(0 <= corners[i, 0] <= 320 for i in range(4))
        assert all(0 <= corners[i, 1] <= 320 for i in range(4))

    def test_no_detection_on_empty_mask(self) -> None:
        detector = SegDetector.__new__(SegDetector)
        detector.threshold = 0.5
        detector.min_area_ratio = 0.05
        mask = torch.zeros(1, 1, 80, 80)
        corners = detector._extract_corners(mask[0, 0], original_size=(320, 320))
        assert corners is None
