# picode/detection/training/augmentation.py
"""Augmentation pipeline for FastDetector training."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import Tensor


class PhotometricAugmentation:
    """Photometric augmentations that don't affect corner positions.

    Includes:
        - Brightness adjustment
        - Contrast adjustment
        - Saturation adjustment
        - Gaussian noise
        - Gaussian blur
        - JPEG compression simulation

    Args:
        brightness_range: (min, max) brightness adjustment factor
        contrast_range: (min, max) contrast adjustment factor
        saturation_range: (min, max) saturation adjustment factor
        noise_std: Standard deviation of Gaussian noise
        blur_kernel_range: (min, max) kernel size for blur
        jpeg_quality_range: (min, max) JPEG quality
        p: Probability of applying any augmentation
    """

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        contrast_range: tuple[float, float] = (0.8, 1.2),
        saturation_range: tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.02,
        blur_kernel_range: tuple[int, int] = (3, 7),
        jpeg_quality_range: tuple[int, int] = (50, 95),
        p: float = 0.5,
    ) -> None:
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.noise_std = noise_std
        self.blur_kernel_range = blur_kernel_range
        self.jpeg_quality_range = jpeg_quality_range
        self.p = p

    def __call__(self, image: Tensor) -> Tensor:
        """Apply photometric augmentations.

        Args:
            image: (C, H, W) tensor in [0, 1]

        Returns:
            Augmented image (C, H, W) in [0, 1]
        """
        if random.random() > self.p:
            return image

        # Brightness
        if random.random() < 0.5:
            factor = random.uniform(*self.brightness_range)
            image = image * factor

        # Contrast
        if random.random() < 0.5:
            factor = random.uniform(*self.contrast_range)
            mean = image.mean()
            image = (image - mean) * factor + mean

        # Saturation
        if random.random() < 0.5:
            factor = random.uniform(*self.saturation_range)
            gray = image.mean(dim=0, keepdim=True)
            image = image * factor + gray * (1 - factor)

        # Gaussian noise
        if random.random() < 0.3:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise

        # Gaussian blur
        if random.random() < 0.3:
            kernel_size = random.choice(
                range(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2)
            )
            image = self._gaussian_blur(image, kernel_size)

        # Clamp to valid range
        return image.clamp(0.0, 1.0)

    def _gaussian_blur(self, image: Tensor, kernel_size: int) -> Tensor:
        """Apply Gaussian blur."""
        # Create Gaussian kernel
        sigma = kernel_size / 6.0
        x = torch.arange(kernel_size).float() - kernel_size // 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.outer(kernel_1d)
        kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size)

        # Apply convolution
        padding = kernel_size // 2
        image = image.unsqueeze(0)  # Add batch dim
        blurred = F.conv2d(
            image,
            kernel_2d.to(image.device),
            padding=padding,
            groups=3,
        )
        return blurred.squeeze(0)


class GeometricAugmentation:
    """Geometric augmentations that require corner adjustment.

    Includes:
        - Perspective transform
        - Rotation

    Args:
        perspective_strength: (min, max) perspective distortion strength
        rotation_degrees: (min, max) rotation angle in degrees
        p: Probability of applying augmentation
    """

    def __init__(
        self,
        perspective_strength: tuple[float, float] = (0.0, 0.1),
        rotation_degrees: tuple[float, float] = (-15, 15),
        p: float = 0.5,
    ) -> None:
        self.perspective_strength = perspective_strength
        self.rotation_degrees = rotation_degrees
        self.p = p

    def __call__(
        self, image: Tensor, corners: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply geometric augmentations.

        Args:
            image: (C, H, W) tensor in [0, 1]
            corners: (8,) tensor of normalized [0, 1] corner coordinates

        Returns:
            Tuple of (augmented image, adjusted corners)
        """
        if random.random() > self.p:
            return image, corners

        # Apply rotation
        if random.random() < 0.5:
            angle = random.uniform(*self.rotation_degrees)
            image, corners = self._rotate(image, corners, angle)

        # Apply perspective
        if random.random() < 0.5:
            strength = random.uniform(*self.perspective_strength)
            corners = self._perturb_corners(corners, strength)

        # Ensure corners remain valid
        corners = corners.clamp(0.0, 1.0)

        return image, corners

    def _rotate(
        self, image: Tensor, corners: Tensor, angle: float
    ) -> tuple[Tensor, Tensor]:
        """Rotate image and corners around center."""
        import math

        # Convert angle to radians
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        # Rotate corners around center (0.5, 0.5)
        corners_2d = corners.reshape(4, 2)
        centered = corners_2d - 0.5

        rotation_matrix = torch.tensor([
            [cos_a, -sin_a],
            [sin_a, cos_a],
        ])
        rotated = centered @ rotation_matrix.T
        corners_new = (rotated + 0.5).flatten()

        # Rotate image using affine grid
        c, h, w = image.shape
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
        ]).unsqueeze(0).float()

        grid = F.affine_grid(theta, (1, c, h, w), align_corners=False)
        image_rotated = F.grid_sample(
            image.unsqueeze(0), grid, align_corners=False, padding_mode="border"
        ).squeeze(0)

        return image_rotated, corners_new

    def _perturb_corners(self, corners: Tensor, strength: float) -> Tensor:
        """Add random perspective distortion to corners."""
        perturbation = (torch.rand(8) - 0.5) * 2 * strength
        return corners + perturbation


class DetectionAugmentation:
    """Combined augmentation pipeline for detection training.

    Applies photometric and geometric augmentations appropriately:
    - Photometric: Applied to all images
    - Geometric: Applied only to positive samples (with corners)

    Args:
        photometric_p: Probability of photometric augmentation
        geometric_p: Probability of geometric augmentation
    """

    def __init__(
        self,
        photometric_p: float = 0.5,
        geometric_p: float = 0.5,
        perspective_strength: tuple[float, float] = (0.0, 0.1),
        rotation_degrees: tuple[float, float] = (-15, 15),
    ) -> None:
        self.photometric = PhotometricAugmentation(p=photometric_p)
        self.geometric = GeometricAugmentation(
            perspective_strength=perspective_strength,
            rotation_degrees=rotation_degrees,
            p=geometric_p,
        )

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply augmentations to a training sample.

        Args:
            sample: Dict with keys:
                - image: (C, H, W) tensor
                - is_watermark: scalar tensor (0 or 1)
                - corners: (8,) tensor of corner coordinates
                - has_corners: scalar tensor (0 or 1)

        Returns:
            Augmented sample with same keys
        """
        image = sample["image"]
        corners = sample["corners"]
        is_positive = sample["has_corners"].item() > 0.5

        # Apply photometric augmentation to all images
        image = self.photometric(image)

        # Apply geometric augmentation only to positive samples
        if is_positive:
            image, corners = self.geometric(image, corners)

        return {
            "image": image,
            "is_watermark": sample["is_watermark"],
            "corners": corners,
            "has_corners": sample["has_corners"],
        }
