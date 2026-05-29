"""Tests for PicoTrust v2 encoder with amplitude control and spatial mask."""

import torch

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
