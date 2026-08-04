# picode/tests/detection/test_augmentation.py
"""Tests for detection augmentation pipeline."""

import random

import pytest
import torch
from torch import Tensor

from picode.detection.training.augmentation import (
    DetectionAugmentation,
    GeometricAugmentation,
    PhotometricAugmentation,
)


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Augmentations draw from ``random``; a few percent of draws are no-ops, so seed
    every test to keep "modifies" assertions deterministic."""
    random.seed(0)
    torch.manual_seed(0)


class TestPhotometricAugmentation:
    @pytest.fixture
    def augment(self) -> PhotometricAugmentation:
        return PhotometricAugmentation(p=1.0)  # Always apply for testing

    @pytest.fixture
    def sample_image(self) -> Tensor:
        return torch.rand(3, 320, 320)

    def test_output_shape_preserved(
        self, augment: PhotometricAugmentation, sample_image: Tensor
    ) -> None:
        output = augment(sample_image)
        assert output.shape == sample_image.shape

    def test_output_range_valid(
        self, augment: PhotometricAugmentation, sample_image: Tensor
    ) -> None:
        output = augment(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_modifies_image(
        self, augment: PhotometricAugmentation, sample_image: Tensor
    ) -> None:
        output = augment(sample_image)
        # Should modify the image (not identical)
        assert not torch.allclose(output, sample_image)

    def test_probability_zero_no_change(self, sample_image: Tensor) -> None:
        augment = PhotometricAugmentation(p=0.0)
        output = augment(sample_image)
        assert torch.allclose(output, sample_image)


class TestGeometricAugmentation:
    @pytest.fixture
    def augment(self) -> GeometricAugmentation:
        return GeometricAugmentation(
            perspective_strength=(0.05, 0.1),
            rotation_degrees=(-15, 15),
            p=1.0,
        )

    @pytest.fixture
    def sample_image(self) -> Tensor:
        return torch.rand(3, 320, 320)

    @pytest.fixture
    def sample_corners(self) -> Tensor:
        """Normalized corners for a centered rectangle."""
        return torch.tensor([
            0.1, 0.1,  # TL
            0.9, 0.1,  # TR
            0.9, 0.9,  # BR
            0.1, 0.9,  # BL
        ])

    def test_output_shapes(
        self, augment: GeometricAugmentation, sample_image: Tensor, sample_corners: Tensor
    ) -> None:
        img_out, corners_out = augment(sample_image, sample_corners)
        assert img_out.shape == sample_image.shape
        assert corners_out.shape == sample_corners.shape

    def test_corners_still_valid(
        self, augment: GeometricAugmentation, sample_image: Tensor, sample_corners: Tensor
    ) -> None:
        _, corners_out = augment(sample_image, sample_corners)
        # Corners should remain in [0, 1] range
        assert corners_out.min() >= 0.0
        assert corners_out.max() <= 1.0

    def test_modifies_corners(
        self, augment: GeometricAugmentation, sample_image: Tensor, sample_corners: Tensor
    ) -> None:
        _, corners_out = augment(sample_image, sample_corners)
        # Should modify corners
        assert not torch.allclose(corners_out, sample_corners)


class TestDetectionAugmentation:
    @pytest.fixture
    def augment(self) -> DetectionAugmentation:
        return DetectionAugmentation(
            photometric_p=1.0,
            geometric_p=1.0,
        )

    @pytest.fixture
    def sample_data(self) -> dict[str, Tensor]:
        return {
            "image": torch.rand(3, 320, 320),
            "is_watermark": torch.tensor(1.0),
            "corners": torch.tensor([0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]),
            "has_corners": torch.tensor(1.0),
        }

    def test_augments_positive_sample(
        self, augment: DetectionAugmentation, sample_data: dict[str, Tensor]
    ) -> None:
        output = augment(sample_data)

        assert "image" in output
        assert "corners" in output
        assert output["image"].shape == sample_data["image"].shape
        assert output["corners"].shape == sample_data["corners"].shape

    def test_preserves_labels(
        self, augment: DetectionAugmentation, sample_data: dict[str, Tensor]
    ) -> None:
        output = augment(sample_data)

        assert output["is_watermark"] == sample_data["is_watermark"]
        assert output["has_corners"] == sample_data["has_corners"]

    def test_negative_sample_no_corner_augment(
        self, augment: DetectionAugmentation
    ) -> None:
        """Negative samples shouldn't have corner augmentation applied."""
        data = {
            "image": torch.rand(3, 320, 320),
            "is_watermark": torch.tensor(0.0),
            "corners": torch.zeros(8),
            "has_corners": torch.tensor(0.0),
        }
        output = augment(data)

        # Corners should remain zero for negative samples
        assert torch.allclose(output["corners"], data["corners"])

    def test_output_range_valid(
        self, augment: DetectionAugmentation, sample_data: dict[str, Tensor]
    ) -> None:
        output = augment(sample_data)

        assert output["image"].min() >= 0.0
        assert output["image"].max() <= 1.0
        assert output["corners"].min() >= 0.0
        assert output["corners"].max() <= 1.0
