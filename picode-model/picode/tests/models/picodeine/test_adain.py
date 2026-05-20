"""Tests for AdaIN module and MappingNetwork."""

import torch
import torch.nn as nn

from picode.models.picodeine.adain import AdaIN, MappingNetwork


class TestMappingNetwork:
    """Tests for the message-to-latent mapping network."""

    def test_output_shape(self) -> None:
        net = MappingNetwork(num_bits=127, mapping_dim=256)
        message = torch.rand(2, 127)
        w = net(message)
        assert w.shape == (2, 256)

    def test_different_messages_different_outputs(self) -> None:
        net = MappingNetwork(num_bits=127, mapping_dim=256)
        m1 = torch.zeros(1, 127)
        m2 = torch.ones(1, 127)
        w1 = net(m1)
        w2 = net(m2)
        assert not torch.allclose(w1, w2)

    def test_gradient_flow(self) -> None:
        net = MappingNetwork(num_bits=127, mapping_dim=256)
        message = torch.rand(2, 127, requires_grad=True)
        w = net(message)
        w.sum().backward()
        assert message.grad is not None
        assert message.grad.abs().sum() > 0


class TestAdaIN:
    """Tests for the Adaptive Instance Normalization module."""

    def test_output_shape(self) -> None:
        adain = AdaIN(mapping_dim=256, num_features=128)
        x = torch.randn(2, 128, 64, 64)
        w = torch.randn(2, 256)
        out = adain(x, w)
        assert out.shape == (2, 128, 64, 64)

    def test_identity_at_init(self) -> None:
        """At initialization (gamma=0, beta=0 from zero-init projection),
        AdaIN should approximate identity (output = instance-normalized input)."""
        adain = AdaIN(mapping_dim=256, num_features=64)
        x = torch.randn(2, 64, 32, 32)
        w = torch.zeros(2, 256)
        out = adain(x, w)
        # With gamma=0, beta=0: output = (1+0)*norm(x) + 0 = norm(x)
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-8
        expected = (x - mean) / std
        assert torch.allclose(out, expected, atol=1e-5)

    def test_identity_for_any_w_at_init(self) -> None:
        """Zero-init weight means any w produces identity (not just w=0)."""
        adain = AdaIN(mapping_dim=256, num_features=64)
        x = torch.randn(2, 64, 32, 32)
        w_random = torch.randn(2, 256)
        out = adain(x, w_random)
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-8
        expected = (x - mean) / std
        assert torch.allclose(out, expected, atol=1e-5)

    def test_modulation_changes_output(self) -> None:
        """After training (non-zero weights), different w produces different output."""
        adain = AdaIN(mapping_dim=256, num_features=64)
        # Simulate trained state with non-zero projection weights
        nn.init.normal_(adain.projection.weight)
        x = torch.randn(2, 64, 32, 32)
        w_zero = torch.zeros(2, 256)
        w_nonzero = torch.randn(2, 256)
        out_zero = adain(x, w_zero)
        out_nonzero = adain(x, w_nonzero)
        assert not torch.allclose(out_zero, out_nonzero)

    def test_gradient_flow(self) -> None:
        adain = AdaIN(mapping_dim=256, num_features=128)
        x = torch.randn(2, 128, 64, 64, requires_grad=True)
        w = torch.randn(2, 256, requires_grad=True)
        out = adain(x, w)
        out.sum().backward()
        assert x.grad is not None
        assert w.grad is not None
