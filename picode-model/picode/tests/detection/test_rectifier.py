# picode/tests/detection/test_rectifier.py
"""Tests for perspective rectification."""

import pytest
import torch
from torch import Tensor

from picode.detection.rectifier import Rectifier
from picode.detection.types import Point, Quadrilateral


class TestRectifier:
    @pytest.fixture
    def rectifier(self) -> Rectifier:
        return Rectifier(output_size=400)

    @pytest.fixture
    def sample_image(self) -> Tensor:
        """Sample 640x480 image."""
        return torch.rand(3, 480, 640)

    @pytest.fixture
    def unit_quad(self) -> Quadrilateral:
        """Quadrilateral covering full image."""
        return Quadrilateral(
            top_left=Point(0.0, 0.0),
            top_right=Point(640.0, 0.0),
            bottom_right=Point(640.0, 480.0),
            bottom_left=Point(0.0, 480.0),
        )

    @pytest.fixture
    def center_quad(self) -> Quadrilateral:
        """Quadrilateral in center of image."""
        return Quadrilateral(
            top_left=Point(100.0, 100.0),
            top_right=Point(500.0, 100.0),
            bottom_right=Point(500.0, 400.0),
            bottom_left=Point(100.0, 400.0),
        )

    def test_rectifier_creation(self, rectifier: Rectifier) -> None:
        assert rectifier.output_size == 400
        assert rectifier.dst_corners.shape == (4, 2)

    def test_rectify_output_shape(
        self, rectifier: Rectifier, sample_image: Tensor, center_quad: Quadrilateral
    ) -> None:
        rectified = rectifier.rectify(sample_image, center_quad)

        assert rectified.shape == (3, 400, 400)

    def test_rectify_output_range(
        self, rectifier: Rectifier, sample_image: Tensor, center_quad: Quadrilateral
    ) -> None:
        rectified = rectifier.rectify(sample_image, center_quad)

        assert rectified.min() >= 0.0
        assert rectified.max() <= 1.0

    def test_rectify_preserves_dtype(
        self, rectifier: Rectifier, sample_image: Tensor, center_quad: Quadrilateral
    ) -> None:
        rectified = rectifier.rectify(sample_image, center_quad)

        assert rectified.dtype == torch.float32

    def test_rectify_with_tensor_corners(
        self, rectifier: Rectifier, sample_image: Tensor
    ) -> None:
        corners_tensor = torch.tensor(
            [100.0, 100.0, 500.0, 100.0, 500.0, 400.0, 100.0, 400.0]
        )
        rectified = rectifier.rectify(sample_image, corners_tensor)

        assert rectified.shape == (3, 400, 400)

    def test_rectify_perspective_distorted(self, rectifier: Rectifier) -> None:
        """Test with perspective-distorted quadrilateral."""
        # Create a simple gradient image
        image = torch.zeros(3, 400, 400)
        image[0, :, :] = torch.linspace(0, 1, 400).unsqueeze(0)  # Red gradient

        # Perspective quadrilateral (trapezoid)
        quad = Quadrilateral(
            top_left=Point(50.0, 50.0),
            top_right=Point(350.0, 60.0),  # Slightly lower
            bottom_right=Point(380.0, 350.0),  # More to the right
            bottom_left=Point(20.0, 340.0),  # More to the left
        )

        rectified = rectifier.rectify(image, quad)

        assert rectified.shape == (3, 400, 400)
        assert rectified.isfinite().all()

    def test_rectify_batch(self, rectifier: Rectifier) -> None:
        images = [torch.rand(3, 480, 640) for _ in range(3)]
        quads = [
            Quadrilateral(
                top_left=Point(100.0 + i * 10, 100.0),
                top_right=Point(500.0, 100.0),
                bottom_right=Point(500.0, 400.0),
                bottom_left=Point(100.0 + i * 10, 400.0),
            )
            for i in range(3)
        ]

        rectified = rectifier.rectify_batch(images, quads)

        assert rectified.shape == (3, 3, 400, 400)

    def test_custom_output_size(self, sample_image: Tensor, center_quad: Quadrilateral) -> None:
        rectifier = Rectifier(output_size=256)
        rectified = rectifier.rectify(sample_image, center_quad)

        assert rectified.shape == (3, 256, 256)
