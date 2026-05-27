"""Tests for PicodeFrame loss functions."""

import torch

from picode.models.picodeframe import loss as frame_loss


class TestMessageLoss:
    """Test message loss functions."""

    def test_bce_loss_shape(self) -> None:
        logits = torch.randn(2, 127)
        targets = torch.randint(0, 2, (2, 127)).float()
        loss = frame_loss.message_loss(logits, targets)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_mse_loss_shape(self) -> None:
        logits = torch.randn(2, 127)
        targets = torch.randint(0, 2, (2, 127)).float()
        loss = frame_loss.message_loss_mse(logits, targets)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_mse_loss_perfect_prediction(self) -> None:
        targets = torch.ones(2, 127)
        logits = torch.full((2, 127), 10.0)
        loss = frame_loss.message_loss_mse(logits, targets)
        assert loss.item() < 0.001

    def test_mse_loss_worst_prediction(self) -> None:
        targets = torch.ones(2, 127)
        logits = torch.full((2, 127), -10.0)
        loss = frame_loss.message_loss_mse(logits, targets)
        assert loss.item() > 0.9

    def test_mse_gradient_at_half(self) -> None:
        """MSE should have non-zero gradient at 0.5 (unlike BCE trivial equilibrium)."""
        logits = torch.zeros(2, 127, requires_grad=True)
        targets = torch.ones(2, 127)
        loss = frame_loss.message_loss_mse(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().mean().item() > 0.0005


class TestFrameL2Loss:
    """Test frame L2 loss."""

    def test_zero_when_identical(self) -> None:
        image = torch.rand(2, 3, 64, 64)
        mask = torch.zeros(2, 1, 64, 64)
        mask[:, :, 8:56, 8:56] = 1.0
        loss = frame_loss.frame_l2_loss(image, image, mask)
        assert loss.item() == 0.0

    def test_positive_when_different(self) -> None:
        image1 = torch.rand(2, 3, 64, 64)
        image2 = torch.rand(2, 3, 64, 64)
        mask = torch.zeros(2, 1, 64, 64)
        mask[:, :, 8:56, 8:56] = 1.0
        loss = frame_loss.frame_l2_loss(image1, image2, mask)
        assert loss.item() > 0
