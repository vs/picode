"""Tests for PicoGrain encoder with noise modulation and luminance mask."""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picode.models.picograin.encoder import Encoder


class TestEncoderInit:

    def test_accepts_num_bits_127(self):
        enc = Encoder(num_bits=127, image_size=512, strength=0.10)
        assert enc.num_bits == 127

    def test_default_strength(self):
        enc = Encoder(num_bits=127, image_size=256)
        assert enc.strength is None

    def test_no_batchnorm(self):
        enc = Encoder(num_bits=127, image_size=256, strength=0.10)
        for module in enc.modules():
            assert not isinstance(
                module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
            )


class TestNoiseModulation:

    def test_output_shape(self, sample_image: Tensor, sample_message: Tensor):
        enc = Encoder(num_bits=127, image_size=512, strength=0.10)
        result = enc(sample_image, sample_message)
        assert result["encoded"].shape == (2, 3, 512, 512)

    def test_residual_is_grayscale(self, sample_image: Tensor, sample_message: Tensor):
        """All 3 RGB channels of residual should be identical."""
        enc = Encoder(num_bits=127, image_size=512, strength=0.10)
        result = enc(sample_image, sample_message)
        residual = result["encoded"] - sample_image
        assert torch.allclose(residual[:, 0], residual[:, 1], atol=1e-6)
        assert torch.allclose(residual[:, 1], residual[:, 2], atol=1e-6)

    def test_residual_bounded_by_strength(
        self, sample_image: Tensor, sample_message: Tensor,
    ):
        enc = Encoder(num_bits=127, image_size=512, strength=0.10)
        result = enc(sample_image, sample_message)
        residual = result["encoded"] - sample_image
        assert residual.abs().max().item() < 0.10 + 1e-6

    def test_residual_has_high_frequency(
        self, sample_image: Tensor, sample_message: Tensor,
    ):
        """Noise modulation should produce high-frequency residual (grain-like)."""
        enc = Encoder(num_bits=127, image_size=512, strength=0.10)
        with torch.no_grad():
            torch.manual_seed(42)
            nn.init.kaiming_normal_(cast(nn.Conv2d, enc.e_post[-1]).weight)
        result = enc(sample_image, sample_message)
        residual = result["encoded"] - sample_image
        kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32,
        ).view(1, 1, 3, 3)
        laplacian = F.conv2d(residual[:, :1], kernel, padding=1)
        hf_energy = laplacian.abs().mean().item()
        assert hf_energy > 0.0001, f"HF energy {hf_energy} too low — not grain-like"

    def test_different_noise_per_call(
        self, sample_image: Tensor, sample_message: Tensor,
    ):
        """Each forward pass should produce different grain (fresh noise)."""
        enc = Encoder(num_bits=127, image_size=512, strength=0.10)
        with torch.no_grad():
            nn.init.kaiming_normal_(cast(nn.Conv2d, enc.e_post[-1]).weight)
        r1 = enc(sample_image, sample_message)["encoded"]
        r2 = enc(sample_image, sample_message)["encoded"]
        assert not torch.allclose(r1, r2, atol=1e-4)

    def test_gradient_flow(self, sample_image: Tensor, sample_message: Tensor):
        enc = Encoder(num_bits=127, image_size=512, strength=0.10)
        sample_image.requires_grad_(True)
        result = enc(sample_image, sample_message)
        result["encoded"].sum().backward()
        assert sample_image.grad is not None
        assert not torch.all(sample_image.grad == 0)


class TestLuminanceMask:

    def test_returns_luminance_mask(self, sample_image: Tensor, sample_message: Tensor):
        enc = Encoder(num_bits=127, image_size=512, strength=0.10)
        result = enc(sample_image, sample_message)
        assert "lum_mask" in result
        assert result["lum_mask"].shape == (2, 1, 512, 512)

    def test_mask_in_range(self, sample_image: Tensor, sample_message: Tensor):
        enc = Encoder(num_bits=127, image_size=512, strength=0.10)
        result = enc(sample_image, sample_message)
        mask = result["lum_mask"]
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_bright_areas_have_more_grain(self):
        """Bright image should produce higher mask values than dark image."""
        enc = Encoder(num_bits=127, image_size=256, strength=0.10)
        bright = torch.ones(1, 3, 256, 256) * 0.9
        dark = torch.ones(1, 3, 256, 256) * 0.1
        msg = torch.randint(0, 2, (1, 127)).float()
        mask_bright = enc(bright, msg)["lum_mask"].mean()
        mask_dark = enc(dark, msg)["lum_mask"].mean()
        assert mask_bright > mask_dark

    def test_mask_floor_prevents_zero(self):
        """Even pure-black images should have non-zero mask (floor)."""
        enc = Encoder(num_bits=127, image_size=256, strength=0.10)
        black = torch.zeros(1, 3, 256, 256)
        msg = torch.randint(0, 2, (1, 127)).float()
        mask = enc(black, msg)["lum_mask"]
        assert mask.min() >= 0.09  # floor=0.1
