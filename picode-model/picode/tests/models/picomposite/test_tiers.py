"""Tests for PicoMposite tier definitions."""

from picode.models.picomposite.tiers import (
    EMBED_DIM,
    MAX_BITS,
    NUM_TIERS,
    TIER_NAMES,
    TIERS,
)


class TestTierDefinitions:

    def test_num_tiers_matches_dict(self):
        assert NUM_TIERS == len(TIERS)

    def test_max_bits_is_largest_tier(self):
        assert MAX_BITS == max(t["bits"] for t in TIERS.values())

    def test_tiers_have_required_keys(self):
        for idx, tier in TIERS.items():
            assert "bits" in tier, f"Tier {idx} missing 'bits'"
            assert "strength" in tier, f"Tier {idx} missing 'strength'"
            assert "decoder_blur_sigma" in tier, f"Tier {idx} missing 'decoder_blur_sigma'"
            assert "perceptual_blur_sigma" in tier, f"Tier {idx} missing 'perceptual_blur_sigma'"

    def test_tier_bits_are_ordered(self):
        bits = [TIERS[i]["bits"] for i in range(NUM_TIERS)]
        assert bits == sorted(bits)

    def test_tier_strengths_are_ordered(self):
        strengths = [TIERS[i]["strength"] for i in range(NUM_TIERS)]
        assert strengths == sorted(strengths)

    def test_tier_perceptual_blur_decreasing(self):
        """Higher tiers (more bits) should have lower or equal perceptual blur sigma."""
        sigmas = [TIERS[i]["perceptual_blur_sigma"] for i in range(NUM_TIERS)]
        assert sigmas == sorted(sigmas, reverse=True)

    def test_tier_decoder_blur_decreasing(self):
        """Higher tiers (more bits) should have lower or equal decoder blur sigma."""
        sigmas = [TIERS[i]["decoder_blur_sigma"] for i in range(NUM_TIERS)]
        assert sigmas == sorted(sigmas, reverse=True)

    def test_tier_values(self):
        expected = {
            0: {"bits": 30, "strength": 0.008,
                "decoder_blur_sigma": 1.0, "perceptual_blur_sigma": 1.0},
            1: {"bits": 48, "strength": 0.010,
                "decoder_blur_sigma": 0.7, "perceptual_blur_sigma": 0.7},
            2: {"bits": 72, "strength": 0.012,
                "decoder_blur_sigma": 0.5, "perceptual_blur_sigma": 0.5},
            3: {"bits": 96, "strength": 0.014,
                "decoder_blur_sigma": 0.3, "perceptual_blur_sigma": 0.5},
        }
        for t_idx, values in expected.items():
            assert TIERS[t_idx] == values, f"Tier {t_idx} mismatch"

    def test_tier_names(self):
        assert TIER_NAMES == {0: "UHQ", 1: "HQ", 2: "MQ", 3: "LQ"}

    def test_embed_dim(self):
        assert EMBED_DIM == 32
