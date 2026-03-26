# picode/tests/detection/test_hard_negative.py
"""Tests for hard negative sampling."""

import pytest
import torch
from torch import Tensor
from PIL import Image
import numpy as np

from picode.detection.training.hard_negative import (
    HardNegativeTransform,
    HardNegativeDataset,
)


class TestHardNegativeTransform:
    @pytest.fixture
    def transform(self) -> HardNegativeTransform:
        return HardNegativeTransform()

    @pytest.fixture
    def sample_image(self) -> Tensor:
        return torch.rand(3, 320, 320)

    def test_jpeg_artifacts(self, transform: HardNegativeTransform, sample_image: Tensor) -> None:
        output = transform.apply_jpeg_artifacts(sample_image, quality=30)
        assert output.shape == sample_image.shape
        assert output.min() >= 0.0
        assert output.max() <= 1.0
        # Should modify the image
        assert not torch.allclose(output, sample_image)

    def test_resize_artifacts(self, transform: HardNegativeTransform, sample_image: Tensor) -> None:
        output = transform.apply_resize_artifacts(sample_image, scale=0.5)
        assert output.shape == sample_image.shape
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_filter_effects(self, transform: HardNegativeTransform, sample_image: Tensor) -> None:
        output = transform.apply_filter_effect(sample_image, effect="sharpen")
        assert output.shape == sample_image.shape
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_random_transform(self, transform: HardNegativeTransform, sample_image: Tensor) -> None:
        output = transform(sample_image)
        assert output.shape == sample_image.shape
        assert output.min() >= 0.0
        assert output.max() <= 1.0


class TestHardNegativeDataset:
    @pytest.fixture
    def sample_images(self, tmp_path) -> str:
        """Create temporary image directory with sample images."""
        img_dir = tmp_path / "negatives"
        img_dir.mkdir()

        for i in range(10):
            img = Image.fromarray(
                np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
            )
            img.save(img_dir / f"neg_{i}.jpg")

        return str(img_dir)

    def test_dataset_creation(self, sample_images: str) -> None:
        dataset = HardNegativeDataset(
            image_dir=sample_images,
            input_size=320,
        )
        assert len(dataset) > 0

    def test_getitem_returns_dict(self, sample_images: str) -> None:
        dataset = HardNegativeDataset(
            image_dir=sample_images,
            input_size=320,
        )
        item = dataset[0]

        assert "image" in item
        assert "is_watermark" in item
        assert "corners" in item
        assert "has_corners" in item

    def test_negative_labels(self, sample_images: str) -> None:
        dataset = HardNegativeDataset(
            image_dir=sample_images,
            input_size=320,
        )
        item = dataset[0]

        # All samples should be negative
        assert item["is_watermark"] == 0.0
        assert item["has_corners"] == 0.0
        assert torch.all(item["corners"] == 0.0)

    def test_output_shape(self, sample_images: str) -> None:
        dataset = HardNegativeDataset(
            image_dir=sample_images,
            input_size=320,
        )
        item = dataset[0]

        assert item["image"].shape == (3, 320, 320)
        assert item["corners"].shape == (8,)

    def test_transform_applied(self, sample_images: str) -> None:
        dataset = HardNegativeDataset(
            image_dir=sample_images,
            input_size=320,
            transform_p=1.0,  # Always apply
        )

        # Get same image twice - should be different due to random transforms
        torch.manual_seed(42)
        item1 = dataset[0]
        torch.manual_seed(123)
        item2 = dataset[0]

        # Images should differ due to random transforms
        assert not torch.allclose(item1["image"], item2["image"])
