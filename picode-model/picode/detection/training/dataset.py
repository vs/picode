# picode/detection/training/dataset.py
"""Synthetic dataset for FastDetector training."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class DetectionDataset(Dataset):
    """Synthetic dataset for training FastDetector.

    Generates positive samples by encoding watermarks into images and
    applying perspective transforms. Negative samples are clean images.

    Args:
        image_dir: Directory containing training images
        encoder: Trained encoder model for watermarking
        num_bits: Number of message bits (default 100)
        positive_ratio: Ratio of positive (watermarked) samples
        input_size: Output image size for detector (default 320)
        perspective_strength: Range of perspective distortion (min, max)
        transform: Optional additional transforms

    Example:
        >>> encoder = Encoder.load("encoder.pt")
        >>> dataset = DetectionDataset("./images", encoder, positive_ratio=0.5)
        >>> item = dataset[0]
        >>> item["image"].shape
        torch.Size([3, 320, 320])
    """

    def __init__(
        self,
        image_dir: str | Path,
        encoder: nn.Module,
        num_bits: int = 100,
        positive_ratio: float = 0.5,
        input_size: int = 320,
        perspective_strength: tuple[float, float] = (0.0, 0.15),
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.encoder = encoder
        self.num_bits = num_bits
        self.positive_ratio = positive_ratio
        self.input_size = input_size
        self.perspective_strength = perspective_strength
        self.transform = transform

        # Collect image paths
        self.image_paths = list(self.image_dir.glob("*.jpg")) + list(
            self.image_dir.glob("*.png")
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        # Image loading transform
        self._load_transform = transforms.Compose(
            [
                transforms.Resize((400, 400)),  # Encoder input size
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        # Load image
        img_path = self.image_paths[idx % len(self.image_paths)]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self._load_transform(image)

        is_positive = random.random() < self.positive_ratio

        if is_positive:
            return self._generate_positive(image_tensor)
        else:
            return self._generate_negative(image_tensor)

    def _generate_positive(self, image: Tensor) -> dict[str, Tensor]:
        """Generate watermarked image with perspective transform."""
        # Encode watermark
        message = torch.randint(0, 2, (1, self.num_bits)).float()
        with torch.no_grad():
            # Encoder returns unclamped image (allows gradients during training)
            # We clamp to [0, 1] for detector training dataset
            watermarked = self.encoder(image.unsqueeze(0), message)
            watermarked = torch.clamp(watermarked, 0, 1)
        watermarked = watermarked.squeeze(0)

        # Generate random perspective corners
        corners = self._random_perspective_corners()

        # Apply perspective transform to watermarked image
        # For now, we skip actual perspective warp in dataset (done in augmentation)
        # Just resize to input size
        output = F.interpolate(
            watermarked.unsqueeze(0),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Apply additional transforms
        if self.transform is not None:
            output = self.transform(output)

        return {
            "image": output,
            "is_watermark": torch.tensor(1.0),
            "corners": corners.flatten(),
            "has_corners": torch.tensor(1.0),
        }

    def _generate_negative(self, image: Tensor) -> dict[str, Tensor]:
        """Generate clean (non-watermarked) image."""
        # Resize to input size
        output = F.interpolate(
            image.unsqueeze(0),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Apply additional transforms
        if self.transform is not None:
            output = self.transform(output)

        return {
            "image": output,
            "is_watermark": torch.tensor(0.0),
            "corners": torch.zeros(8),
            "has_corners": torch.tensor(0.0),
        }

    def _random_perspective_corners(self) -> Tensor:
        """Generate random quadrilateral corners (normalized [0, 1])."""
        # Start with a rectangle covering most of the image
        cx, cy = random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)
        size_x = random.uniform(0.5, 0.8)
        size_y = random.uniform(0.5, 0.8)

        # Base rectangle corners
        corners = torch.tensor(
            [
                [cx - size_x / 2, cy - size_y / 2],  # TL
                [cx + size_x / 2, cy - size_y / 2],  # TR
                [cx + size_x / 2, cy + size_y / 2],  # BR
                [cx - size_x / 2, cy + size_y / 2],  # BL
            ]
        )

        # Add perspective distortion
        strength = random.uniform(*self.perspective_strength)
        for i in range(4):
            corners[i, 0] += random.uniform(-strength, strength)
            corners[i, 1] += random.uniform(-strength, strength)

        # Clamp to valid range
        corners = corners.clamp(0.02, 0.98)

        return corners
