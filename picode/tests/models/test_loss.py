"""Tests for loss functions."""

import torch

from picode.models.stegastamp.loss import compute_loss, image_loss, message_loss


class TestMessageLoss:
    """Test message (BCE) loss."""

    def test_perfect_prediction(self) -> None:
        """Perfect prediction gives near-zero loss."""
        message = torch.tensor([[1.0, 0.0, 1.0]])
        decoded = torch.tensor([[0.99, 0.01, 0.99]])
        loss = message_loss(decoded, message)
        assert loss < 0.1

    def test_wrong_prediction(self) -> None:
        """Wrong prediction gives high loss."""
        message = torch.tensor([[1.0, 0.0, 1.0]])
        decoded = torch.tensor([[0.01, 0.99, 0.01]])
        loss = message_loss(decoded, message)
        assert loss > 2.0

    def test_gradient_flow(self) -> None:
        """Gradients flow through loss."""
        message = torch.tensor([[1.0, 0.0]])
        decoded = torch.tensor([[0.5, 0.5]], requires_grad=True)
        loss = message_loss(decoded, message)
        loss.backward()  # type: ignore[no-untyped-call]
        assert decoded.grad is not None


class TestImageLoss:
    """Test image (L2) loss."""

    def test_identical_images(self) -> None:
        """Identical images give zero loss."""
        image = torch.rand(1, 3, 64, 64)
        loss = image_loss(image, image)
        assert loss == 0.0

    def test_different_images(self) -> None:
        """Different images give positive loss."""
        original = torch.zeros(1, 3, 64, 64)
        encoded = torch.ones(1, 3, 64, 64)
        loss = image_loss(encoded, original)
        assert loss == 1.0


class TestComputeLoss:
    """Test combined loss computation."""

    def test_returns_dict(self) -> None:
        """Returns dictionary with all loss components."""
        original = torch.rand(1, 3, 64, 64)
        encoded = torch.rand(1, 3, 64, 64)
        message = torch.randint(0, 2, (1, 10)).float()
        decoded = torch.rand(1, 10)

        losses = compute_loss(original, encoded, message, decoded, use_lpips=False)

        assert "loss" in losses
        assert "loss_msg" in losses
        assert "loss_l2" in losses

    def test_gradient_flow(self) -> None:
        """Gradients flow through combined loss."""
        original = torch.rand(1, 3, 64, 64)
        encoded = torch.rand(1, 3, 64, 64, requires_grad=True)
        message = torch.randint(0, 2, (1, 10)).float()
        decoded = torch.rand(1, 10, requires_grad=True)

        losses = compute_loss(original, encoded, message, decoded, use_lpips=False)
        losses["loss"].backward()  # type: ignore[no-untyped-call]

        assert encoded.grad is not None
        assert decoded.grad is not None
