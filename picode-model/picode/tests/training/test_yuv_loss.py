"""Tests for YUV-weighted L2 loss."""

import pytest
import torch

from picode.training.trainer import compute_yuv_l2_loss


class TestYUVL2Loss:
    """Tests for YUV-weighted L2 loss computation."""

    def test_yuv_loss_shape(self) -> None:
        """YUV loss returns scalar."""
        original = torch.rand(2, 3, 64, 64)
        encoded = torch.rand(2, 3, 64, 64)
        yuv_weights = [1.0, 100.0, 100.0]

        loss = compute_yuv_l2_loss(original, encoded, yuv_weights)

        assert loss.shape == ()
        assert loss.dtype == torch.float32

    def test_yuv_loss_zero_for_identical(self) -> None:
        """YUV loss is zero when images are identical."""
        image = torch.rand(2, 3, 64, 64)
        yuv_weights = [1.0, 100.0, 100.0]

        loss = compute_yuv_l2_loss(image, image, yuv_weights)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_yuv_loss_weights_chrominance(self) -> None:
        """YUV loss penalizes chrominance (U/V) more than luma (Y)."""
        original = torch.zeros(1, 3, 64, 64)

        # Pure luma change (grayscale shift)
        luma_change = original.clone()
        luma_change[:, :, :, :] = 0.1  # Equal RGB = pure Y change

        # Pure chrominance change (color shift, same brightness)
        chroma_change = original.clone()
        chroma_change[:, 0, :, :] = 0.1  # Red only = U/V change
        chroma_change[:, 1, :, :] = 0.0
        chroma_change[:, 2, :, :] = 0.0

        yuv_weights = [1.0, 100.0, 100.0]

        loss_luma = compute_yuv_l2_loss(original, luma_change, yuv_weights)
        loss_chroma = compute_yuv_l2_loss(original, chroma_change, yuv_weights)

        # Chrominance change should be penalized more heavily
        assert loss_chroma > loss_luma

    def test_yuv_loss_gradient_flows(self) -> None:
        """Gradients flow through YUV loss."""
        original = torch.rand(2, 3, 64, 64)
        encoded = torch.rand(2, 3, 64, 64, requires_grad=True)
        yuv_weights = [1.0, 100.0, 100.0]

        loss = compute_yuv_l2_loss(original, encoded, yuv_weights)
        loss.backward()

        assert encoded.grad is not None
        assert not torch.all(encoded.grad == 0)
