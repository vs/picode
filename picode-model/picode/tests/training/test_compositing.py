"""Tests for compositing module."""

from __future__ import annotations

import torch

from picode.training.compositing import composite_into_background, extract_with_jitter


class TestCompositeIntoBackground:
    def test_output_shape(self) -> None:
        encoded = torch.rand(2, 3, 512, 512)
        background = torch.rand(2, 3, 512, 512)
        result = composite_into_background(
            encoded, background, scale_min=0.5, scale_max=0.9,
            perspective_strength=0.05,
        )
        assert result["composited"].shape == (2, 3, 512, 512)
        assert result["mask"].shape == (2, 1, 512, 512)
        assert result["corners"].shape == (2, 4, 2)

    def test_output_range(self) -> None:
        encoded = torch.rand(2, 3, 512, 512)
        background = torch.rand(2, 3, 512, 512)
        result = composite_into_background(
            encoded, background, scale_min=0.7, scale_max=0.9,
            perspective_strength=0.0,
        )
        assert result["composited"].min() >= 0.0
        assert result["composited"].max() <= 1.0
        assert result["mask"].min() >= 0.0
        assert result["mask"].max() <= 1.0

    def test_mask_covers_encoded_region(self) -> None:
        encoded = torch.ones(1, 3, 512, 512)
        background = torch.zeros(1, 3, 512, 512)
        result = composite_into_background(
            encoded, background, scale_min=0.5, scale_max=0.5,
            perspective_strength=0.0,
        )
        mask = result["mask"].expand_as(result["composited"])
        masked_pixels = result["composited"][mask > 0.5]
        assert masked_pixels.mean() > 0.9


class TestExtractWithJitter:
    def test_output_shape(self) -> None:
        composited = torch.rand(2, 3, 512, 512)
        corners = torch.tensor([
            [[64, 64], [448, 64], [448, 448], [64, 448]],
            [[100, 100], [400, 100], [400, 400], [100, 400]],
        ]).float()
        result = extract_with_jitter(
            composited, corners, output_size=512, jitter=0.0,
        )
        assert result.shape == (2, 3, 512, 512)

    def test_jitter_changes_output(self) -> None:
        torch.manual_seed(42)
        composited = torch.rand(1, 3, 512, 512)
        corners = torch.tensor([[[64, 64], [448, 64], [448, 448], [64, 448]]]).float()
        r1 = extract_with_jitter(composited, corners, output_size=512, jitter=0.1)
        r2 = extract_with_jitter(composited, corners, output_size=512, jitter=0.1)
        assert not torch.allclose(r1, r2)

    def test_no_jitter_is_deterministic(self) -> None:
        composited = torch.rand(1, 3, 512, 512)
        corners = torch.tensor([[[64, 64], [448, 64], [448, 448], [64, 448]]]).float()
        r1 = extract_with_jitter(composited, corners, output_size=512, jitter=0.0)
        r2 = extract_with_jitter(composited, corners, output_size=512, jitter=0.0)
        assert torch.allclose(r1, r2)
