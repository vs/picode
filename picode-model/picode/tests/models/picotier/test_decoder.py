"""Tests for PicoTier decoder with tier classification and conditioned output."""

import torch

from picode.models.picotier.decoder import Decoder
from picode.models.picotier.tiers import MAX_BITS, NUM_TIERS, TIERS


class TestDecoderConstruction:

    def test_decoder_creates_256(self):
        dec = Decoder(image_size=256)
        assert dec.image_size == 256

    def test_decoder_creates_512(self):
        dec = Decoder(image_size=512)
        assert dec.image_size == 512


class TestDecoderForward:

    def test_output_shapes_256(self, sample_image_256):
        dec = Decoder(image_size=256)
        tier_3 = torch.tensor([3, 3])
        logits, tier_logits = dec(sample_image_256, tier=tier_3)
        assert logits.shape == (2, MAX_BITS)
        assert tier_logits.shape == (2, NUM_TIERS)

    def test_output_shapes_512(self, sample_image):
        dec = Decoder(image_size=512)
        tier_3 = torch.tensor([3, 3])
        logits, tier_logits = dec(sample_image, tier=tier_3)
        assert logits.shape == (2, MAX_BITS)
        assert tier_logits.shape == (2, NUM_TIERS)

    def test_forward_without_tier_uses_classifier(self, sample_image_256):
        """When tier is not provided, decoder auto-detects via classifier."""
        dec = Decoder(image_size=256)
        logits, tier_logits = dec(sample_image_256)
        assert logits.shape == (2, MAX_BITS)
        assert tier_logits.shape == (2, NUM_TIERS)


class TestDecode:

    def test_decode_returns_tier_and_bits(self, sample_image_256):
        dec = Decoder(image_size=256)
        tier_idx, bits = dec.decode(sample_image_256)
        assert tier_idx.shape == (2,)
        assert len(bits) == 2
        # Each sample should have correct number of bits for its detected tier
        for i in range(2):
            t = int(tier_idx[i].item())
            expected_bits = int(TIERS[t]["bits"])
            assert bits[i].shape[0] == expected_bits
            # Bits should be 0 or 1
            assert ((bits[i] == 0) | (bits[i] == 1)).all()


class TestGradientFlow:

    def test_gradients_flow(self, sample_image_256):
        dec = Decoder(image_size=256)
        tier = torch.tensor([3, 3])
        logits, tier_logits = dec(sample_image_256, tier=tier)
        (logits.sum() + tier_logits.sum()).backward()
        assert dec.tier_classifier.weight.grad is not None
        assert dec.tier_embedding.weight.grad is not None
