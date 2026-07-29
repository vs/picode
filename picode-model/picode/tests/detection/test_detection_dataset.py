# picode/tests/detection/test_detection_dataset.py
"""Tests for detection training dataset."""

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from picode.detection.training.dataset import DetectionDataset


class _MockEncoder(nn.Module):
    """Mock encoder that returns input with small perturbation."""

    def __init__(self) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, image: Tensor, message: Tensor) -> Tensor:
        return image + torch.randn_like(image) * 0.01


class TestDetectionDataset:
    @pytest.fixture
    def mock_encoder(self) -> nn.Module:
        """Mock encoder that returns input with small perturbation."""
        encoder = _MockEncoder()
        encoder.eval()
        return encoder

    @pytest.fixture
    def sample_images(self, tmp_path) -> str:
        """Create temporary image directory with sample images."""
        import numpy as np
        from PIL import Image

        img_dir = tmp_path / "images"
        img_dir.mkdir()

        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
            )
            img.save(img_dir / f"img_{i}.jpg")

        return str(img_dir)

    def test_dataset_creation(
        self, mock_encoder: nn.Module, sample_images: str
    ) -> None:
        dataset = DetectionDataset(
            image_dir=sample_images,
            encoder=mock_encoder,
            num_bits=100,
            positive_ratio=0.5,
        )

        assert len(dataset) > 0

    def test_dataset_getitem_returns_dict(
        self, mock_encoder: nn.Module, sample_images: str
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
        self, mock_encoder: nn.Module, sample_images: str
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
        self, mock_encoder: nn.Module, sample_images: str
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
        self, mock_encoder: nn.Module, sample_images: str
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
        self, mock_encoder: nn.Module, sample_images: str
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
        self, mock_encoder: nn.Module, sample_images: str
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

    def test_sobel_mask_attenuates_residual(
        self, mock_encoder: nn.Module, sample_images: str
    ) -> None:
        """Sobel mask should produce valid images."""
        ds_mask = DetectionDataset(
            image_dir=sample_images,
            encoder=mock_encoder,
            positive_ratio=1.0,
            sobel_mask_sigma=5.0,
            sobel_mask_floor=0.85,
        )
        item_mask = ds_mask[0]

        assert item_mask["image"].shape == (3, 320, 320)
        assert item_mask["image"].min() >= 0.0
        assert item_mask["image"].max() <= 1.0

    def test_strength_sampling_varies_residual(
        self, mock_encoder: nn.Module, sample_images: str
    ) -> None:
        """Different strength values should produce different encoded images."""
        ds_low = DetectionDataset(
            image_dir=sample_images,
            encoder=mock_encoder,
            positive_ratio=1.0,
            strength_values=[0.1],
        )
        ds_high = DetectionDataset(
            image_dir=sample_images,
            encoder=mock_encoder,
            positive_ratio=1.0,
            strength_values=[1.0],
        )

        item_low = ds_low[0]
        item_high = ds_high[0]

        assert item_low["image"].shape == (3, 320, 320)
        assert item_high["image"].shape == (3, 320, 320)

    def test_backward_compat_no_sobel(
        self, mock_encoder: nn.Module, sample_images: str
    ) -> None:
        """Dataset without Sobel params should work exactly as before."""
        dataset = DetectionDataset(
            image_dir=sample_images,
            encoder=mock_encoder,
            num_bits=100,
            positive_ratio=1.0,
        )

        item = dataset[0]

        assert item["image"].shape == (3, 320, 320)
        assert item["is_watermark"] == 1.0
        assert item["has_corners"] == 1.0
        assert item["corners"].shape == (8,)
