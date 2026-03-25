# picode/tests/detection/test_detection_dataset.py
"""Tests for detection training dataset."""

import pytest
import torch
from torch import Tensor
from unittest.mock import Mock

from picode.detection.training.dataset import DetectionDataset


class TestDetectionDataset:
    @pytest.fixture
    def mock_encoder(self) -> Mock:
        """Mock encoder that returns input with small perturbation."""
        encoder = Mock()
        encoder.return_value = torch.rand(1, 3, 400, 400)
        return encoder

    @pytest.fixture
    def sample_images(self, tmp_path) -> str:
        """Create temporary image directory with sample images."""
        from PIL import Image
        import numpy as np

        img_dir = tmp_path / "images"
        img_dir.mkdir()

        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
            )
            img.save(img_dir / f"img_{i}.jpg")

        return str(img_dir)

    def test_dataset_creation(
        self, mock_encoder: Mock, sample_images: str
    ) -> None:
        dataset = DetectionDataset(
            image_dir=sample_images,
            encoder=mock_encoder,
            num_bits=100,
            positive_ratio=0.5,
        )

        assert len(dataset) > 0

    def test_dataset_getitem_returns_dict(
        self, mock_encoder: Mock, sample_images: str
    ) -> None:
        dataset = DetectionDataset(
            image_dir=sample_images,
            encoder=mock_encoder,
            positive_ratio=1.0,  # All positive
        )

        item = dataset[0]

        assert "image" in item
        assert "is_watermark" in item
        assert "corners" in item
        assert "has_corners" in item

    def test_positive_sample_shape(
        self, mock_encoder: Mock, sample_images: str
    ) -> None:
        dataset = DetectionDataset(
            image_dir=sample_images,
            encoder=mock_encoder,
            positive_ratio=1.0,
            input_size=320,
        )

        item = dataset[0]

        assert item["image"].shape == (3, 320, 320)
        assert item["corners"].shape == (8,)
        assert item["is_watermark"].shape == ()
        assert item["has_corners"].shape == ()

    def test_positive_sample_values(
        self, mock_encoder: Mock, sample_images: str
    ) -> None:
        dataset = DetectionDataset(
            image_dir=sample_images,
            encoder=mock_encoder,
            positive_ratio=1.0,
        )

        item = dataset[0]

        assert item["is_watermark"] == 1.0
        assert item["has_corners"] == 1.0
        # Corners should be normalized [0, 1]
        assert item["corners"].min() >= 0.0
        assert item["corners"].max() <= 1.0

    def test_negative_sample_values(
        self, mock_encoder: Mock, sample_images: str
    ) -> None:
        dataset = DetectionDataset(
            image_dir=sample_images,
            encoder=mock_encoder,
            positive_ratio=0.0,  # All negative
        )

        item = dataset[0]

        assert item["is_watermark"] == 0.0
        assert item["has_corners"] == 0.0

    def test_image_range(
        self, mock_encoder: Mock, sample_images: str
    ) -> None:
        dataset = DetectionDataset(
            image_dir=sample_images,
            encoder=mock_encoder,
            positive_ratio=0.5,
        )

        item = dataset[0]

        assert item["image"].min() >= 0.0
        assert item["image"].max() <= 1.0

    def test_perspective_range(
        self, mock_encoder: Mock, sample_images: str
    ) -> None:
        dataset = DetectionDataset(
            image_dir=sample_images,
            encoder=mock_encoder,
            positive_ratio=1.0,
            perspective_strength=(0.05, 0.1),
        )

        # Sample multiple times to check perspective variation
        corners_list = [dataset[i]["corners"] for i in range(5)]

        # Should have variation due to random perspective
        corners_stack = torch.stack(corners_list)
        variation = corners_stack.std(dim=0).mean()
        assert variation > 0.01  # Some variation expected
