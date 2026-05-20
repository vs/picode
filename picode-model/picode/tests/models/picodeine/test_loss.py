"""Tests for Picodeine loss functions."""

import torch

from picode.models.picodeine.decoder import Decoder
from picode.models.picodeine.loss import compute_picodeine_loss, message_loss


class TestMessageLoss:
    """Test message loss function."""

    def test_perfect_prediction(self) -> None:
        """Loss should be near zero for perfect prediction."""
        message = torch.ones(2, 127)
        logits = torch.ones(2, 127) * 10.0  # Strong positive = sigmoid(10) ≈ 1
        loss = message_loss(logits, message)
        assert loss.item() < 0.01

    def test_random_prediction(self) -> None:
        """Loss should be ~0.693 for random (logits=0 → sigmoid=0.5)."""
        message = torch.randint(0, 2, (2, 127)).float()
        logits = torch.zeros(2, 127)
        loss = message_loss(logits, message)
        assert abs(loss.item() - 0.693) < 0.01

    def test_gradient_flow(self) -> None:
        logits = torch.randn(2, 127, requires_grad=True)
        message = torch.randint(0, 2, (2, 127)).float()
        loss = message_loss(logits, message)
        loss.backward()
        assert logits.grad is not None


class TestComputeLoss:
    """Test combined loss computation."""

    def test_returns_all_components(self) -> None:
        decoder = Decoder(num_bits=127)
        losses = compute_picodeine_loss(
            original=torch.rand(2, 3, 512, 512),
            encoded=torch.rand(2, 3, 512, 512),
            messages=torch.randint(0, 2, (2, 127)).float(),
            decoded_logits=torch.randn(2, 127),
            decoder=decoder,
            lpips_fn=None,
            use_lpips=False,
        )
        assert "loss_msg" in losses
        assert "loss_l2" in losses
        assert "loss_stn_reg" in losses
        assert "loss" in losses

    def test_total_loss_is_scalar(self) -> None:
        decoder = Decoder(num_bits=127)
        losses = compute_picodeine_loss(
            original=torch.rand(2, 3, 512, 512),
            encoded=torch.rand(2, 3, 512, 512),
            messages=torch.randint(0, 2, (2, 127)).float(),
            decoded_logits=torch.randn(2, 127),
            decoder=decoder,
            lpips_fn=None,
            use_lpips=False,
        )
        assert losses["loss"].dim() == 0

    def test_gradient_flow_through_total(self) -> None:
        decoder = Decoder(num_bits=127)
        encoded = torch.rand(2, 3, 512, 512, requires_grad=True)
        logits = torch.randn(2, 127, requires_grad=True)
        losses = compute_picodeine_loss(
            original=torch.rand(2, 3, 512, 512),
            encoded=encoded,
            messages=torch.randint(0, 2, (2, 127)).float(),
            decoded_logits=logits,
            decoder=decoder,
            lpips_fn=None,
            use_lpips=False,
        )
        losses["loss"].backward()
        assert encoded.grad is not None
        assert logits.grad is not None
