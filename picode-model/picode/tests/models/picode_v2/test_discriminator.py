"""Tests for picode_v2 PatchDiscriminator."""

import torch

from picode.models.picode_v2.discriminator import PatchDiscriminator


class TestPatchDiscriminator:
    """Tests for PatchDiscriminator."""

    def test_output_shape(self) -> None:
        """Output is a grid of predictions."""
        disc = PatchDiscriminator()
        x = torch.rand(2, 3, 400, 400)
        out = disc(x)
        # PatchGAN outputs a spatial grid, not a single value
        assert out.dim() == 4
        assert out.shape[0] == 2
        assert out.shape[1] == 1  # Single channel output

    def test_gradient_flow(self) -> None:
        """Gradients flow through discriminator."""
        disc = PatchDiscriminator()
        x = torch.rand(2, 3, 400, 400, requires_grad=True)
        out = disc(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().mean() > 0

    def test_different_inputs_different_outputs(self) -> None:
        """Different inputs produce different outputs."""
        disc = PatchDiscriminator()
        x1 = torch.rand(1, 3, 400, 400)
        x2 = torch.rand(1, 3, 400, 400)
        out1 = disc(x1)
        out2 = disc(x2)
        assert not torch.allclose(out1, out2, atol=1e-3)

    def test_works_with_smaller_images(self) -> None:
        """Works with images smaller than 400x400."""
        disc = PatchDiscriminator()
        x = torch.rand(1, 3, 256, 256)
        out = disc(x)
        assert out.dim() == 4
