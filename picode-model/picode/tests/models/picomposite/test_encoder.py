"""Tests for PicoMposite encoder with tier conditioning."""

import torch

from picode.models.picomposite.encoder import Encoder
from picode.models.picomposite.tiers import MAX_BITS, TIERS


class TestEncoderConstruction:

    def test_encoder_creates(self):
        enc = Encoder(image_size=256)
        assert enc.image_size == 256

    def test_encoder_512(self):
        enc = Encoder(image_size=512)
        assert enc.image_size == 512


class TestEncoderForward:

    def test_output_shape_256(self, sample_image_256, sample_message_96, tier_3):
        enc = Encoder(image_size=256)
        result = enc(sample_image_256, sample_message_96, tier_3)
        assert result["encoded"].shape == (2, 3, 256, 256)

    def test_output_shape_512(self, sample_image, sample_message_96, tier_3):
        enc = Encoder(image_size=512)
        result = enc(sample_image, sample_message_96, tier_3)
        assert result["encoded"].shape == (2, 3, 512, 512)

    def test_output_range(self, sample_image_256, sample_message_96, tier_0):
        enc = Encoder(image_size=256)
        result = enc(sample_image_256, sample_message_96, tier_0)
        encoded = result["encoded"]
        # Residual is bounded, but encoded can slightly exceed [0,1] before clamping
        assert encoded.min() >= -0.02
        assert encoded.max() <= 1.02


class TestTierConditioning:

    def test_per_tier_strength_bounds_residual(self, sample_image_256, sample_message_96):
        enc = Encoder(image_size=256)
        for tier_idx in range(4):
            tier = torch.tensor([tier_idx, tier_idx])
            result = enc(sample_image_256, sample_message_96, tier)
            residual = result["encoded"] - sample_image_256
            strength = TIERS[tier_idx]["strength"]
            assert residual.abs().max().item() < strength + 1e-6, (
                f"Tier {tier_idx}: residual {residual.abs().max():.6f} > strength {strength}"
            )

    def test_different_tiers_produce_different_outputs(
        self, sample_image_256, sample_message_96
    ):
        enc = Encoder(image_size=256)
        tier_0 = torch.tensor([0, 0])
        tier_3 = torch.tensor([3, 3])
        out_0 = enc(sample_image_256, sample_message_96, tier_0)["encoded"]
        out_3 = enc(sample_image_256, sample_message_96, tier_3)["encoded"]
        assert not torch.allclose(out_0, out_3)

    def test_mixed_tiers_in_batch(self, sample_image_256, sample_message_96, mixed_tiers):
        enc = Encoder(image_size=256)
        result = enc(sample_image_256, sample_message_96, mixed_tiers)
        residual = result["encoded"] - sample_image_256
        # Sample 0 is tier 0 (strength 0.010)
        assert residual[0].abs().max().item() < TIERS[0]["strength"] + 1e-6
        # Sample 1 is tier 3 (strength 0.016)
        assert residual[1].abs().max().item() < TIERS[3]["strength"] + 1e-6


class TestGradientFlow:

    def test_gradients_flow(self, sample_image_256, sample_message_96, tier_3):
        enc = Encoder(image_size=256)
        result = enc(sample_image_256, sample_message_96, tier_3)
        result["encoded"].sum().backward()
        # Check some key parameters have gradients
        assert enc.secret_dense.weight.grad is not None
        assert enc.tier_embedding.weight.grad is not None
        assert enc.bottleneck_proj.weight.grad is not None
