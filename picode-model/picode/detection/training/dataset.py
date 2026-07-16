# picode/detection/training/dataset.py
"""Synthetic dataset for FastDetector training."""

from __future__ import annotations

import math
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
        encoder_input_size: Input size expected by encoder (default 400)
        perspective_strength: Range of perspective distortion (min, max)
        transform: Optional additional transforms
        sobel_mask_sigma: Gaussian sigma for Sobel mask smoothing (None=disabled)
        sobel_mask_floor: Minimum mask value in smooth regions (default 0.85)
        strength_values: List of residual strengths to sample from (None=full strength)
        hard_negative_p: Probability of applying hard negative transform to negatives

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
        encoder_input_size: int = 400,
        perspective_strength: tuple[float, float] = (0.0, 0.15),
        transform: Callable[[Tensor], Tensor] | None = None,
        sobel_mask_sigma: float | None = None,
        sobel_mask_floor: float = 0.85,
        strength_values: list[float] | None = None,
        hard_negative_p: float = 0.0,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.encoder = encoder
        self.num_bits = num_bits
        self.positive_ratio = positive_ratio
        self.input_size = input_size
        self.perspective_strength = perspective_strength
        self.transform = transform
        self.sobel_mask_sigma = sobel_mask_sigma
        self.sobel_mask_floor = sobel_mask_floor
        self.strength_values = strength_values
        self._hard_negative_p = hard_negative_p

        # Instantiate hard negative transform if needed
        if hard_negative_p > 0:
            from picode.detection.training.hard_negative import HardNegativeTransform

            self._hard_negative: HardNegativeTransform | None = HardNegativeTransform()
        else:
            self._hard_negative = None

        # Collect image paths
        self.image_paths = list(self.image_dir.glob("*.jpg")) + list(
            self.image_dir.glob("*.png")
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        # Image loading transform
        self._load_transform = transforms.Compose(
            [
                transforms.Resize((encoder_input_size, encoder_input_size)),
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
        # Get encoder device (could be CUDA or CPU)
        encoder_device = next(self.encoder.parameters()).device

        # Encode watermark - move tensors to encoder's device
        message = torch.randint(0, 2, (1, self.num_bits)).float().to(encoder_device)
        image_on_device = image.unsqueeze(0).to(encoder_device)

        with torch.no_grad():
            # Encoder returns unclamped image (allows gradients during training)
            # We clamp to [0, 1] for detector training dataset
            output = self.encoder(image_on_device, message)
            if isinstance(output, dict):
                raw_encoded = output["encoded"]
            else:
                raw_encoded = output

            # Apply Sobel mask + strength scaling (matches production inference)
            if self.sobel_mask_sigma is not None or self.strength_values is not None:
                residual = raw_encoded - image_on_device

                if self.sobel_mask_sigma is not None:
                    mask = self._sobel_texture_mask(image_on_device)
                    residual = residual * mask

                if self.strength_values is not None:
                    strength = random.choice(self.strength_values)
                    residual = residual * strength

                watermarked = (image_on_device + residual).clamp(0, 1)
            else:
                watermarked = torch.clamp(raw_encoded, 0, 1)

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

        # Move back to CPU for DataLoader compatibility
        output = output.cpu()

        # Build sample dict
        sample = {
            "image": output,
            "is_watermark": torch.tensor(1.0),
            "corners": corners.flatten(),
            "has_corners": torch.tensor(1.0),
        }

        # Apply additional transforms (expects dict, returns dict)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _generate_negative(self, image: Tensor) -> dict[str, Tensor]:
        """Generate clean (non-watermarked) image."""
        # Resize to input size
        output = F.interpolate(
            image.unsqueeze(0),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Apply hard negative transform (JPEG artifacts, resize, filters)
        if self._hard_negative is not None and random.random() < self._hard_negative_p:
            output = self._hard_negative(output)

        # Build sample dict
        sample = {
            "image": output,
            "is_watermark": torch.tensor(0.0),
            "corners": torch.zeros(8),
            "has_corners": torch.tensor(0.0),
        }

        # Apply additional transforms (expects dict, returns dict)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _sobel_texture_mask(self, image: Tensor) -> Tensor:
        """Compute Sobel gradient texture mask for an image.

        Args:
            image: (1, 3, H, W) tensor in [0, 1].

        Returns:
            Mask (1, 1, H, W) in [floor, 1.0] — high in textured regions.
        """
        sigma = self.sobel_mask_sigma
        assert sigma is not None
        floor = self.sobel_mask_floor

        gray = image.mean(dim=1, keepdim=True)  # (1, 1, H, W)
        sx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=image.device,
        ).view(1, 1, 3, 3)
        sy = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device=image.device,
        ).view(1, 1, 3, 3)
        gx = F.conv2d(gray, sx, padding=1)
        gy = F.conv2d(gray, sy, padding=1)
        grad_mag = (gx ** 2 + gy ** 2).sqrt()

        # Smooth with Gaussian
        k = 2 * math.ceil(3 * sigma) + 1
        ax = torch.arange(k, dtype=torch.float32, device=image.device) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        gk = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        gk = (gk / gk.sum()).view(1, 1, k, k)
        grad_smooth = F.conv2d(grad_mag, gk, padding=k // 2)

        # Normalize per image and apply floor
        grad_max = grad_smooth.amax(dim=(-2, -1), keepdim=True) + 1e-8
        mask = floor + (1.0 - floor) * (grad_smooth / grad_max)
        return mask

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
