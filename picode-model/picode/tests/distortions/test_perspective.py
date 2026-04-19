"""Tests for perspective transform utilities.

These tests verify the pre-encode warp / post-encode unwarp infrastructure
that enables StegaStamp's canonical encoding space.
"""

import pytest
import torch
from torch import Tensor

from picode.distortions.native.perspective import (
    get_identity_transform,
    get_rand_transform_matrix,
    perspective_transform,
)


class TestGetRandTransformMatrix:
    """Tests for get_rand_transform_matrix function."""

    def test_output_shapes(self) -> None:
        """Verify correct output shapes for forward and inverse matrices."""
        batch_size = 4
        image_size = 64
        max_translation = 0.1
        device = torch.device("cpu")

        M_forward, M_inverse = get_rand_transform_matrix(
            batch_size, image_size, max_translation, device
        )

        assert M_forward.shape == (batch_size, 3, 3)
        assert M_inverse.shape == (batch_size, 3, 3)

    def test_device_placement(self) -> None:
        """Verify matrices are placed on correct device."""
        batch_size = 2
        image_size = 32
        max_translation = 0.1

        # Test CPU
        M_forward, M_inverse = get_rand_transform_matrix(
            batch_size, image_size, max_translation, torch.device("cpu")
        )
        assert M_forward.device.type == "cpu"
        assert M_inverse.device.type == "cpu"

    def test_different_batches_different_transforms(self) -> None:
        """Each batch item should have a different transform."""
        batch_size = 4
        image_size = 64
        max_translation = 0.1

        torch.manual_seed(42)
        M_forward, _ = get_rand_transform_matrix(
            batch_size, image_size, max_translation, torch.device("cpu")
        )

        # Check that transforms differ between batch items
        # Compare first and second transform
        assert not torch.allclose(M_forward[0], M_forward[1], atol=1e-4)

    def test_normalized_homography(self) -> None:
        """H[2,2] should be approximately 1 (normalized)."""
        batch_size = 2
        image_size = 64
        max_translation = 0.1

        torch.manual_seed(42)
        M_forward, M_inverse = get_rand_transform_matrix(
            batch_size, image_size, max_translation, torch.device("cpu")
        )

        for b in range(batch_size):
            assert abs(M_forward[b, 2, 2].item() - 1.0) < 0.01
            assert abs(M_inverse[b, 2, 2].item() - 1.0) < 0.01


class TestGetIdentityTransform:
    """Tests for get_identity_transform function."""

    def test_output_shape(self) -> None:
        """Verify correct output shape."""
        batch_size = 4
        M = get_identity_transform(batch_size, torch.device("cpu"))
        assert M.shape == (batch_size, 3, 3)

    def test_is_identity(self) -> None:
        """Verify matrices are identity."""
        batch_size = 3
        M = get_identity_transform(batch_size, torch.device("cpu"))

        identity = torch.eye(3)
        for b in range(batch_size):
            torch.testing.assert_close(M[b], identity)


class TestPerspectiveTransform:
    """Tests for perspective_transform function."""

    @pytest.fixture
    def sample_image(self) -> Tensor:
        """Create a sample image tensor (2, 3, 64, 64)."""
        torch.manual_seed(42)
        return torch.rand(2, 3, 64, 64)

    def test_identity_transform_preserves_image(self, sample_image: Tensor) -> None:
        """Identity transform should not change the image."""
        B = sample_image.shape[0]
        M = get_identity_transform(B, sample_image.device)

        output = perspective_transform(sample_image, M)

        # Should be very close to input
        torch.testing.assert_close(output, sample_image, rtol=1e-4, atol=1e-4)

    def test_output_shape_preserved(self, sample_image: Tensor) -> None:
        """Output shape should match input shape."""
        B, C, H, W = sample_image.shape
        M_forward, M_inverse = get_rand_transform_matrix(
            B, H, 0.1, sample_image.device
        )

        output = perspective_transform(sample_image, M_inverse)

        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, sample_image: Tensor) -> None:
        """Output values should be in valid range."""
        B, C, H, W = sample_image.shape
        M_forward, M_inverse = get_rand_transform_matrix(
            B, H, 0.1, sample_image.device
        )

        output = perspective_transform(sample_image, M_inverse, padding_mode="border")

        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_forward_inverse_roundtrip_center_region(self, sample_image: Tensor) -> None:
        """Forward-inverse roundtrip should approximately recover center region.

        Due to the nature of perspective transforms and bilinear interpolation,
        we expect the center of the image to be better preserved than edges.
        """
        B, C, H, W = sample_image.shape

        torch.manual_seed(42)
        M_forward, M_inverse = get_rand_transform_matrix(
            B, H, 0.05, sample_image.device  # Small translation for better recovery
        )

        # Forward transform (warp)
        warped = perspective_transform(sample_image, M_inverse, padding_mode="border")

        # Inverse transform (unwarp)
        unwarped = perspective_transform(warped, M_forward, padding_mode="border")

        # Check center region (exclude border pixels that may have artifacts)
        margin = H // 4
        center_original = sample_image[:, :, margin:-margin, margin:-margin]
        center_recovered = unwarped[:, :, margin:-margin, margin:-margin]

        # Center should be reasonably close (allowing for interpolation artifacts)
        # Note: bilinear interpolation introduces some error even with small transforms
        error = (center_original - center_recovered).abs().mean()
        assert error < 0.2, f"Center region error too high: {error}"

    def test_gradient_flow(self, sample_image: Tensor) -> None:
        """Gradients should flow through the transform."""
        B, C, H, W = sample_image.shape
        image = sample_image.clone().requires_grad_(True)
        M = get_identity_transform(B, image.device)

        output = perspective_transform(image, M)
        loss = output.mean()
        loss.backward()

        assert image.grad is not None
        assert image.grad.shape == image.shape
        # Gradient should not be zero (identity transform has gradient = 1)
        assert image.grad.abs().mean() > 0

    def test_different_padding_modes(self, sample_image: Tensor) -> None:
        """Different padding modes should produce valid outputs."""
        B, C, H, W = sample_image.shape

        torch.manual_seed(42)
        M_forward, M_inverse = get_rand_transform_matrix(
            B, H, 0.2, sample_image.device  # Larger translation to see padding effect
        )

        for padding_mode in ["zeros", "border", "reflection"]:
            output = perspective_transform(sample_image, M_inverse, padding_mode=padding_mode)
            assert output.shape == sample_image.shape
            # All outputs should be finite
            assert torch.isfinite(output).all()


class TestWarpUnwarpFlow:
    """Integration tests for the warp-encode-unwarp flow."""

    @pytest.fixture
    def sample_image(self) -> Tensor:
        """Create a sample image tensor (2, 3, 64, 64)."""
        torch.manual_seed(42)
        return torch.rand(2, 3, 64, 64)

    def test_warp_unwarp_residual_flow(self, sample_image: Tensor) -> None:
        """Test the complete warp-encode-unwarp residual flow.

        This simulates what happens in training:
        1. Warp input image
        2. Create residual in warped space
        3. Unwarp residual
        4. Add to original image
        """
        B, C, H, W = sample_image.shape

        torch.manual_seed(42)
        M_forward, M_inverse = get_rand_transform_matrix(
            B, H, 0.1, sample_image.device
        )

        # 1. Warp input
        warped = perspective_transform(sample_image, M_inverse, padding_mode="border")

        # 2. Create a small residual in warped space (simulating encoder output)
        residual_warped = torch.randn_like(warped) * 0.01

        # 3. Unwarp residual
        residual_unwarped = perspective_transform(
            residual_warped, M_forward, padding_mode="zeros"
        )

        # 4. Add to original
        encoded = (sample_image + residual_unwarped).clamp(0.0, 1.0)

        # Output should be valid
        assert encoded.shape == sample_image.shape
        assert encoded.min() >= 0.0
        assert encoded.max() <= 1.0

    def test_zero_translation_is_identity(self, sample_image: Tensor) -> None:
        """Zero translation should produce identity-like transforms."""
        B, C, H, W = sample_image.shape

        torch.manual_seed(42)
        M_forward, M_inverse = get_rand_transform_matrix(
            B, H, 0.0, sample_image.device  # Zero translation
        )

        # Both should be close to identity
        identity = get_identity_transform(B, sample_image.device)
        torch.testing.assert_close(M_forward, identity, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(M_inverse, identity, rtol=1e-4, atol=1e-4)

        # Transform should preserve image
        # Note: grid_sample may introduce small numerical differences even with identity
        output = perspective_transform(sample_image, M_inverse)
        torch.testing.assert_close(output, sample_image, rtol=1e-3, atol=1e-3)

    def test_gradient_through_warp_unwarp_chain(self, sample_image: Tensor) -> None:
        """Gradients should flow through the entire warp-unwarp chain."""
        B, C, H, W = sample_image.shape
        image = sample_image.clone().requires_grad_(True)

        torch.manual_seed(42)
        M_forward, M_inverse = get_rand_transform_matrix(
            B, H, 0.1, image.device
        )

        # Warp
        warped = perspective_transform(image, M_inverse, padding_mode="border")

        # Simulate encoding (just a linear transform for testing)
        residual_warped = warped * 0.1

        # Unwarp
        residual_unwarped = perspective_transform(
            residual_warped, M_forward, padding_mode="zeros"
        )

        # Final output
        encoded = image + residual_unwarped

        # Backprop
        loss = encoded.mean()
        loss.backward()

        # Gradients should exist and be non-zero
        assert image.grad is not None
        assert image.grad.abs().mean() > 0
