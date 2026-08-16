"""Tests for PicoTrust v2 encoder with amplitude control and spatial mask."""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from picode.models.picotrust.encoder import Encoder


class TestResidualAmplitudeControl:

    def test_encoder_accepts_strength_param(self):
        enc = Encoder(num_bits=100, image_size=256, strength=0.03)
        assert enc.strength == 0.03

    def test_default_strength_is_none(self):
        enc = Encoder(num_bits=100, image_size=256)
        assert enc.strength is None

    def test_residual_bounded_by_strength(self, sample_image_256, sample_message):
        enc = Encoder(num_bits=100, image_size=256, strength=0.03)
        result = enc(sample_image_256, sample_message)
        residual = result["encoded"] - sample_image_256
        assert residual.abs().max().item() < 0.03 + 1e-6

    def test_unconstrained_residual_without_strength(self, sample_image_256, sample_message):
        enc = Encoder(num_bits=100, image_size=256)
        result = enc(sample_image_256, sample_message)
        # Should return a Tensor, not dict
        assert isinstance(result, torch.Tensor)


class TestLearnedSpatialMask:

    def test_encoder_with_mask_returns_mask(self, sample_image_256, sample_message):
        enc = Encoder(num_bits=100, image_size=256, strength=0.03, use_mask=True)
        result = enc(sample_image_256, sample_message)
        assert "mask" in result
        assert result["mask"].shape == (2, 1, 256, 256)

    def test_mask_is_in_zero_one(self, sample_image_256, sample_message):
        enc = Encoder(num_bits=100, image_size=256, strength=0.03, use_mask=True)
        result = enc(sample_image_256, sample_message)
        mask = result["mask"]
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_mask_has_gradient(self, sample_image_256, sample_message):
        enc = Encoder(num_bits=100, image_size=256, strength=0.03, use_mask=True)
        result = enc(sample_image_256, sample_message)
        result["encoded"].sum().backward()
        assert enc.mask_head is not None
        for p in enc.mask_head.parameters():
            assert p.grad is not None

    def test_no_mask_without_use_mask(self, sample_image_256, sample_message):
        enc = Encoder(num_bits=100, image_size=256, strength=0.03, use_mask=False)
        result = enc(sample_image_256, sample_message)
        assert "mask" not in result


class TestEncoderAt512:

    def test_512_output_shape(self, sample_image, sample_message):
        enc = Encoder(num_bits=100, image_size=512, strength=0.05, use_mask=True)
        result = enc(sample_image, sample_message)
        assert result["encoded"].shape == (2, 3, 512, 512)
        assert result["mask"].shape == (2, 1, 512, 512)

    def test_512_residual_bounded(self, sample_image, sample_message):
        enc = Encoder(num_bits=100, image_size=512, strength=0.05, use_mask=True)
        result = enc(sample_image, sample_message)
        residual = result["encoded"] - sample_image
        assert residual.abs().max().item() < 0.05 + 1e-6


class TestEncoderV10:

    def test_bilinear_upsample_smooth(self, sample_image_256, sample_message):
        """Bilinear upsampling produces smoother message features than nearest."""
        enc = Encoder(num_bits=100, image_size=256, strength=0.03)
        msg_norm = sample_message - 0.5
        msg_spatial = enc.prepare_message(msg_norm)
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3).expand(3, -1, -1, -1)
        laplacian = F.conv2d(msg_spatial, kernel, padding=1, groups=3)
        hf_energy = laplacian.abs().mean().item()
        assert hf_energy < 0.5, f"HF energy {hf_energy} too high — upsample may not be bilinear"

    def test_e_post_has_dilated_conv(self):
        """E_post should contain a dilated convolution for larger receptive field."""
        enc = Encoder(num_bits=100, image_size=256, strength=0.03)
        dilated_layers = [
            m for m in enc.e_post.modules()
            if isinstance(m, torch.nn.Conv2d) and m.dilation != (1, 1)
        ]
        assert len(dilated_layers) >= 1, "E_post should have at least one dilated conv"
        assert dilated_layers[0].dilation == (2, 2)
        assert dilated_layers[0].padding == (2, 2), (
            "Dilated conv needs padding=2 for same-size output"
        )

    def test_e_post_output_still_grayscale(self, sample_image_256, sample_message):
        """E_post still outputs 1-channel grayscale residual."""
        enc = Encoder(num_bits=100, image_size=256, strength=0.03)
        result = enc(sample_image_256, sample_message)
        residual = result["encoded"] - sample_image_256
        assert torch.allclose(residual[:, 0], residual[:, 1], atol=1e-6)
        assert torch.allclose(residual[:, 1], residual[:, 2], atol=1e-6)


class TestResidualBlur:

    def test_encoder_accepts_blur_sigma(self):
        enc = Encoder(num_bits=64, image_size=256, strength=0.03, residual_blur_sigma=2.0)
        assert enc.residual_blur_sigma == 2.0

    def test_default_blur_sigma_is_zero(self):
        enc = Encoder(num_bits=64, image_size=256, strength=0.03)
        assert enc.residual_blur_sigma == 0.0

    def test_blur_reduces_high_frequency(self, sample_image_256, sample_message):
        """Blurred residual should have less high-frequency energy than unblurred."""
        enc_no_blur = Encoder(num_bits=100, image_size=256, strength=0.05)
        enc_blur = Encoder(num_bits=100, image_size=256, strength=0.05, residual_blur_sigma=2.0)
        # Use same weights
        enc_blur.load_state_dict(enc_no_blur.state_dict(), strict=False)

        # E_post last layer is zero-initialized, so give it non-zero weights
        # to produce a non-trivial residual for this test
        with torch.no_grad():
            torch.manual_seed(99)
            src = cast(nn.Conv2d, enc_no_blur.e_post[-1])
            nn.init.kaiming_normal_(src.weight)
            cast(nn.Conv2d, enc_blur.e_post[-1]).weight.copy_(src.weight)

        res_no_blur = enc_no_blur(sample_image_256, sample_message)["encoded"] - sample_image_256
        res_blur = enc_blur(sample_image_256, sample_message)["encoded"] - sample_image_256

        # Measure HF via Laplacian
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3)
        hf_no_blur = F.conv2d(res_no_blur[:, :1], kernel, padding=1).abs().mean()
        hf_blur = F.conv2d(res_blur[:, :1], kernel, padding=1).abs().mean()
        assert hf_blur < hf_no_blur

    def test_blur_residual_still_bounded(self, sample_image_256, sample_message):
        """Blur should not increase residual beyond strength bound."""
        enc = Encoder(num_bits=100, image_size=256, strength=0.03, residual_blur_sigma=2.0)
        result = enc(sample_image_256, sample_message)
        residual = result["encoded"] - sample_image_256
        assert residual.abs().max().item() < 0.03 + 1e-6

    def test_blur_residual_still_grayscale(self, sample_image_256, sample_message):
        """Blurred residual should still be R=G=B."""
        enc = Encoder(num_bits=100, image_size=256, strength=0.03, residual_blur_sigma=2.0)
        result = enc(sample_image_256, sample_message)
        residual = result["encoded"] - sample_image_256
        assert torch.allclose(residual[:, 0], residual[:, 1], atol=1e-6)
        assert torch.allclose(residual[:, 1], residual[:, 2], atol=1e-6)

    def test_blur_has_gradient(self, sample_image_256, sample_message):
        """Blur is differentiable — gradients flow through."""
        enc = Encoder(num_bits=100, image_size=256, strength=0.05, residual_blur_sigma=2.0)
        result = enc(sample_image_256, sample_message)
        result["encoded"].sum().backward()
        assert enc.e_post[-1].weight.grad is not None

    def test_no_blur_when_sigma_zero(self, sample_image_256, sample_message):
        """sigma=0 should produce identical output to no blur param."""
        enc_default = Encoder(num_bits=100, image_size=256, strength=0.05)
        enc_zero = Encoder(num_bits=100, image_size=256, strength=0.05, residual_blur_sigma=0.0)
        enc_zero.load_state_dict(enc_default.state_dict(), strict=False)
        torch.manual_seed(42)
        r1 = enc_default(sample_image_256, sample_message)["encoded"]
        torch.manual_seed(42)
        r2 = enc_zero(sample_image_256, sample_message)["encoded"]
        assert torch.allclose(r1, r2, atol=1e-6)
