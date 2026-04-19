"""Tests for compression distortions."""

import pytest
import torch


class TestJPEGCompression:
    """Tests for JPEGCompression distortion."""

    def test_preserves_shape(self, distortion_module, sample_image):
        distortion = distortion_module.JPEGCompression()
        output = distortion(sample_image)
        assert output.shape == sample_image.shape

    def test_output_in_valid_range(self, distortion_module, sample_image):
        distortion = distortion_module.JPEGCompression(intensity=1.0)
        output = distortion(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_zero_intensity_unchanged(self, distortion_module, sample_image):
        distortion = distortion_module.JPEGCompression(intensity=0.0)
        output = distortion(sample_image)
        torch.testing.assert_close(output, sample_image)

    def test_gradient_flow(self, distortion_module, sample_image):
        distortion = distortion_module.JPEGCompression()
        image = sample_image.clone().requires_grad_(True)
        output = distortion(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None

    def test_low_quality_more_artifacts(self, distortion_module, sample_image):
        """Lower quality should produce more deviation from original."""
        high_q = distortion_module.JPEGCompression(intensity=1.0, quality=90)
        low_q = distortion_module.JPEGCompression(intensity=1.0, quality=10)

        out_high = high_q(sample_image)
        out_low = low_q(sample_image)

        diff_high = (out_high - sample_image).abs().mean()
        diff_low = (out_low - sample_image).abs().mean()

        assert diff_low > diff_high

    def test_batch_processing(self, distortion_module, batch_images):
        """Should handle batched inputs."""
        distortion = distortion_module.JPEGCompression()
        output = distortion(batch_images)
        assert output.shape == batch_images.shape


class TestJPEGCompressionNativeOnly:
    """Tests for native JPEGCompression features (Y/C tables, 4:2:0 subsampling)."""

    @pytest.fixture
    def native_module(self):
        """Get the native distortion module."""
        from picode.distortions import native
        return native

    def test_separate_y_c_tables(self, native_module):
        """Verify Y and C quantization tables are different."""
        distortion = native_module.JPEGCompression(quality=50)

        # Y and C tables should be different
        assert not torch.allclose(distortion.y_q_matrix, distortion.c_q_matrix)

        # C table should generally have higher values (more quantization)
        # The C table has many 99 values (coarser quantization for chroma)
        assert distortion.c_q_matrix.mean() > distortion.y_q_matrix.mean()

    def test_downsample_c_true_default(self, native_module):
        """Verify downsample_c=True is the default."""
        distortion = native_module.JPEGCompression()
        assert distortion.downsample_c is True

    def test_downsample_c_false(self, native_module, sample_image):
        """Test JPEG without chroma subsampling."""
        distortion = native_module.JPEGCompression(
            intensity=1.0, quality=50, downsample_c=False
        )
        output = distortion(sample_image)
        assert output.shape == sample_image.shape
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_chroma_subsampling_produces_different_output(self, native_module):
        """4:2:0 subsampling should produce different results than no subsampling."""
        torch.manual_seed(42)
        image = torch.rand(1, 3, 64, 64)

        with_subsampling = native_module.JPEGCompression(
            intensity=1.0, quality=50, downsample_c=True
        )
        without_subsampling = native_module.JPEGCompression(
            intensity=1.0, quality=50, downsample_c=False
        )

        out_with = with_subsampling(image)
        out_without = without_subsampling(image)

        # Outputs should be different
        assert not torch.allclose(out_with, out_without)

    def test_round_only_at_0_default(self, native_module):
        """Verify round_only_at_0 is the default rounding mode."""
        distortion = native_module.JPEGCompression()
        assert distortion.rounding == "round_only_at_0"

    def test_rounding_mode_straight_through(self, native_module, sample_image):
        """Test straight-through rounding mode."""
        distortion = native_module.JPEGCompression(
            intensity=1.0, quality=50, rounding="straight_through"
        )
        output = distortion(sample_image)
        assert output.shape == sample_image.shape

    def test_invalid_rounding_mode_raises(self, native_module):
        """Invalid rounding mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown rounding mode"):
            native_module.JPEGCompression(rounding="invalid")

    def test_gradient_flow_with_round_only_at_0(self, native_module):
        """Gradient should flow through round_only_at_0 rounding."""
        torch.manual_seed(42)
        image = torch.rand(1, 3, 64, 64, requires_grad=True)
        distortion = native_module.JPEGCompression(
            intensity=1.0, quality=50, rounding="round_only_at_0"
        )
        output = distortion(image)
        loss = output.mean()
        loss.backward()
        assert image.grad is not None
        # Gradient should have meaningful values (not all zeros)
        assert image.grad.abs().mean() > 0

    def test_non_multiple_of_16_size_with_subsampling(self, native_module):
        """Test handling of image sizes that aren't multiples of 16."""
        torch.manual_seed(42)
        # 50x50 is not a multiple of 16
        image = torch.rand(1, 3, 50, 50)
        distortion = native_module.JPEGCompression(
            intensity=1.0, quality=50, downsample_c=True
        )
        output = distortion(image)
        assert output.shape == image.shape


class TestRoundOnlyAt0:
    """Tests for the round_only_at_0 function."""

    def test_small_values_become_cubic(self):
        """Values with |x| < 0.5 should use x^3."""
        from picode.distortions.native.compression import round_only_at_0

        x = torch.tensor([0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4])
        result = round_only_at_0(x)
        expected = x**3  # All |x| < 0.5, so all should use x^3
        torch.testing.assert_close(result, expected)

    def test_large_values_unchanged(self):
        """Values with |x| >= 0.5 should remain unchanged."""
        from picode.distortions.native.compression import round_only_at_0

        x = torch.tensor([0.5, 0.7, 1.0, 2.0, -0.5, -0.7, -1.0, -2.0])
        result = round_only_at_0(x)
        torch.testing.assert_close(result, x)

    def test_boundary_at_0_5(self):
        """Test behavior exactly at the 0.5 boundary."""
        from picode.distortions.native.compression import round_only_at_0

        # At exactly 0.5, condition is False (>=0.5), so value is unchanged
        x = torch.tensor([0.5, -0.5])
        result = round_only_at_0(x)
        torch.testing.assert_close(result, x)

        # Just below 0.5, condition is True, so value uses x^3
        x_below = torch.tensor([0.49, -0.49])
        result_below = round_only_at_0(x_below)
        expected_below = x_below**3
        torch.testing.assert_close(result_below, expected_below)

    def test_gradient_flow(self):
        """Gradients should flow through round_only_at_0."""
        from picode.distortions.native.compression import round_only_at_0

        x = torch.tensor([0.1, 0.6, -0.2, -0.8], requires_grad=True)
        result = round_only_at_0(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None


class TestDownsample420:
    """Tests for 4:2:0 chroma subsampling functions."""

    def test_downsample_halves_chroma_dimensions(self):
        """Downsampling should halve Cb/Cr dimensions."""
        from picode.distortions.native.compression import downsample_420

        y = torch.rand(2, 64, 64)
        cb = torch.rand(2, 64, 64)
        cr = torch.rand(2, 64, 64)

        y_out, cb_out, cr_out = downsample_420(y, cb, cr)

        # Y should be unchanged
        torch.testing.assert_close(y_out, y)

        # Cb/Cr should be halved
        assert cb_out.shape == (2, 32, 32)
        assert cr_out.shape == (2, 32, 32)

    def test_upsample_doubles_chroma_dimensions(self):
        """Upsampling should double Cb/Cr dimensions."""
        from picode.distortions.native.compression import upsample_420

        y = torch.rand(2, 64, 64)
        cb = torch.rand(2, 32, 32)
        cr = torch.rand(2, 32, 32)

        y_out, cb_out, cr_out = upsample_420(y, cb, cr)

        # Y should be unchanged
        torch.testing.assert_close(y_out, y)

        # Cb/Cr should be doubled
        assert cb_out.shape == (2, 64, 64)
        assert cr_out.shape == (2, 64, 64)

    def test_downsample_upsample_roundtrip_shape(self):
        """Downsample then upsample should restore original shape."""
        from picode.distortions.native.compression import downsample_420, upsample_420

        y = torch.rand(2, 64, 64)
        cb = torch.rand(2, 64, 64)
        cr = torch.rand(2, 64, 64)

        y_down, cb_down, cr_down = downsample_420(y, cb, cr)
        y_up, cb_up, cr_up = upsample_420(y_down, cb_down, cr_down)

        assert y_up.shape == y.shape
        assert cb_up.shape == cb.shape
        assert cr_up.shape == cr.shape

    def test_upsample_nearest_neighbor_pattern(self):
        """Upsampling should use nearest neighbor (each pixel repeated 2x2)."""
        from picode.distortions.native.compression import upsample_420

        y = torch.zeros(1, 4, 4)
        # Simple 2x2 chroma with distinct values
        cb = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        cr = torch.zeros(1, 2, 2)

        _, cb_up, _ = upsample_420(y, cb, cr)

        # Each value should be repeated in a 2x2 block
        expected = torch.tensor([
            [[1.0, 1.0, 2.0, 2.0],
             [1.0, 1.0, 2.0, 2.0],
             [3.0, 3.0, 4.0, 4.0],
             [3.0, 3.0, 4.0, 4.0]]
        ])
        torch.testing.assert_close(cb_up, expected)
