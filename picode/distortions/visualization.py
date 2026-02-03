"""Visualization helpers for distortion analysis."""

import math

import torch
from torch import Tensor

from picode.distortions.base import Distortion


def create_diff(
    original: Tensor,
    distorted: Tensor,
    amplify: float = 5.0,
) -> Tensor:
    """Create amplified difference image for visualization.

    Args:
        original: Original image tensor (B, C, H, W).
        distorted: Distorted image tensor (B, C, H, W).
        amplify: Amplification factor for visibility.

    Returns:
        Difference image clamped to [0, 1].
    """
    diff = (distorted - original).abs() * amplify
    return torch.clamp(diff, 0.0, 1.0)


def create_comparison(
    original: Tensor,
    distorted: Tensor,
    amplify: float = 5.0,
) -> Tensor:
    """Create side-by-side comparison: original | distorted | diff.

    Args:
        original: Original image tensor (B, C, H, W).
        distorted: Distorted image tensor (B, C, H, W).
        amplify: Amplification factor for diff visibility.

    Returns:
        Concatenated comparison image (B, C, H, W*3).
    """
    diff = create_diff(original, distorted, amplify)
    return torch.cat([original, distorted, diff], dim=-1)


def create_intensity_grid(
    image: Tensor,
    distortion: Distortion,
    levels: int = 9,
) -> Tensor:
    """Create grid showing distortion at different intensity levels.

    Args:
        image: Input image tensor (B, C, H, W).
        distortion: Distortion to apply.
        levels: Number of intensity levels (should be a perfect square).

    Returns:
        Grid image with sqrt(levels) rows and columns.
    """
    # Determine grid dimensions
    grid_size = int(math.ceil(math.sqrt(levels)))
    actual_levels = grid_size * grid_size

    intensities = torch.linspace(0.0, 1.0, actual_levels)

    rows = []
    idx = 0
    for _ in range(grid_size):
        row_images = []
        for _ in range(grid_size):
            if idx < levels:
                # Apply distortion at this intensity
                original_intensity = distortion.intensity
                distortion.intensity = intensities[idx].item()

                # Use deterministic seed for reproducibility
                torch.manual_seed(42)
                distorted = distortion(image)

                distortion.intensity = original_intensity
            else:
                # Pad with zeros if we don't have enough levels
                distorted = torch.zeros_like(image)

            row_images.append(distorted)
            idx += 1

        row = torch.cat(row_images, dim=-1)
        rows.append(row)

    grid = torch.cat(rows, dim=-2)
    return grid
