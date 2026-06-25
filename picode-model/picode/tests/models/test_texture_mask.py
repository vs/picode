"""Tests for texture-based residual masking."""

import torch


class TestComputeTextureMask:
    """Tests for compute_texture_mask."""

    def test_output_shape_matches_input(self):
        """Mask spatial dims match input image."""
        from picode.models.picotrust.texture_mask import compute_texture_mask

        image = torch.rand(1, 3, 64, 64)
        mask = compute_texture_mask(image, floor=0.3)
        assert mask.shape == (1, 1, 64, 64)

    def test_output_range(self):
        """All mask values in [floor, 1.0]."""
        from picode.models.picotrust.texture_mask import compute_texture_mask

        image = torch.rand(1, 3, 128, 128)
        floor = 0.3
        mask = compute_texture_mask(image, floor=floor)
        assert mask.min() >= floor - 1e-6
        assert mask.max() <= 1.0 + 1e-6

    def test_constant_image_returns_floor(self):
        """A constant image (zero variance everywhere) should produce mask ≈ floor."""
        from picode.models.picotrust.texture_mask import compute_texture_mask

        image = torch.full((1, 3, 64, 64), 0.5)
        mask = compute_texture_mask(image, floor=0.3)
        assert torch.allclose(mask, torch.full_like(mask, 0.3), atol=1e-4)

    def test_noisy_image_near_one(self):
        """A uniformly noisy image should produce mask values near 1.0."""
        from picode.models.picotrust.texture_mask import compute_texture_mask

        torch.manual_seed(42)
        image = torch.rand(1, 3, 64, 64)
        mask = compute_texture_mask(image, floor=0.3)
        # Most values should be well above 0.5
        assert mask.mean() > 0.7

    def test_smooth_region_lower_than_textured(self):
        """Smooth region gets lower mask value than textured region."""
        from picode.models.picotrust.texture_mask import compute_texture_mask

        image = torch.zeros(1, 3, 64, 64)
        # Left half: smooth (constant)
        image[:, :, :, :32] = 0.5
        # Right half: noisy/textured
        torch.manual_seed(42)
        image[:, :, :, 32:] = torch.rand(1, 3, 64, 32)

        mask = compute_texture_mask(image, floor=0.2)
        smooth_mean = mask[:, :, :, :16].mean()  # Interior of smooth half
        textured_mean = mask[:, :, :, 48:].mean()  # Interior of textured half
        assert textured_mean > smooth_mean

    def test_batch_dimension(self):
        """Works with batch size > 1."""
        from picode.models.picotrust.texture_mask import compute_texture_mask

        image = torch.rand(4, 3, 64, 64)
        mask = compute_texture_mask(image, floor=0.3)
        assert mask.shape == (4, 1, 64, 64)

    def test_floor_zero(self):
        """Floor=0 allows mask values down to 0."""
        from picode.models.picotrust.texture_mask import compute_texture_mask

        image = torch.full((1, 3, 64, 64), 0.5)
        mask = compute_texture_mask(image, floor=0.0)
        assert mask.min() >= -1e-6

    def test_small_image_no_error(self):
        """Images smaller than default kernel size don't crash."""
        from picode.models.picotrust.texture_mask import compute_texture_mask

        image = torch.rand(1, 3, 8, 8)
        mask = compute_texture_mask(image, floor=0.3)
        assert mask.shape == (1, 1, 8, 8)
