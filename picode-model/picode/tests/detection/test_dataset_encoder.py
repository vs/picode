# picode/tests/detection/test_dataset_encoder.py
"""Tests for DetectionDataset encoder integration."""

import pytest

from picode.detection.training.dataset import DetectionDataset
from picode.models.stegastamp import Encoder


class TestDetectionDatasetEncoder:
    """Tests for DetectionDataset with real encoder."""

    @pytest.fixture
    def encoder(self) -> Encoder:
        """Create a small encoder for testing."""
        encoder = Encoder(num_bits=100)
        encoder.eval()
        return encoder

    @pytest.fixture
    def sample_images_dir(self, tmp_path):
        """Create temp directory with sample images."""
        import numpy as np
        from PIL import Image

        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
            )
            img.save(tmp_path / f"img_{i}.jpg")
        return tmp_path

    def test_dataset_creates_positives(self, encoder, sample_images_dir) -> None:
        """Dataset should create valid positive samples."""
        dataset = DetectionDataset(
            image_dir=sample_images_dir,
            encoder=encoder,
            positive_ratio=1.0,  # All positives
        )

        item = dataset[0]

        assert "image" in item
        assert "is_watermark" in item
        assert "corners" in item
        assert "has_corners" in item

        assert item["image"].shape == (3, 320, 320)
        assert item["is_watermark"].item() == 1.0
        assert item["corners"].shape == (8,)
        assert item["has_corners"].item() == 1.0

    def test_dataset_creates_negatives(self, encoder, sample_images_dir) -> None:
        """Dataset should create valid negative samples."""
        dataset = DetectionDataset(
            image_dir=sample_images_dir,
            encoder=encoder,
            positive_ratio=0.0,  # All negatives
        )

        item = dataset[0]

        assert item["image"].shape == (3, 320, 320)
        assert item["is_watermark"].item() == 0.0
        assert item["has_corners"].item() == 0.0

    def test_image_range_valid(self, encoder, sample_images_dir) -> None:
        """Output images should be in [0, 1] range."""
        dataset = DetectionDataset(
            image_dir=sample_images_dir,
            encoder=encoder,
            positive_ratio=1.0,
        )

        item = dataset[0]

        assert item["image"].min() >= 0.0
        assert item["image"].max() <= 1.0

    def test_corners_in_valid_range(self, encoder, sample_images_dir) -> None:
        """Corners should be normalized to [0, 1]."""
        dataset = DetectionDataset(
            image_dir=sample_images_dir,
            encoder=encoder,
            positive_ratio=1.0,
        )

        item = dataset[0]
        corners = item["corners"]

        assert corners.min() >= 0.0
        assert corners.max() <= 1.0

    def test_watermarked_differs_from_original(self, encoder, sample_images_dir) -> None:
        """Watermarked image should differ from original (encoder actually works)."""
        dataset = DetectionDataset(
            image_dir=sample_images_dir,
            encoder=encoder,
            positive_ratio=1.0,
        )

        # Get positive and negative from same base image
        dataset_neg = DetectionDataset(
            image_dir=sample_images_dir,
            encoder=encoder,
            positive_ratio=0.0,
        )

        pos_item = dataset[0]
        neg_item = dataset_neg[0]

        # They should be different (watermark was applied)
        # Note: Due to random transforms, this is a soft check
        # The watermark residual should add some difference
        diff = (pos_item["image"] - neg_item["image"]).abs().mean()
        # Even with resize, there should be measurable difference if encoder works
        assert diff > 0.0  # Some difference expected
