"""Tests for PicoMposite tier definitions."""

from picode.models.picomposite.tiers import (
    EMBED_DIM, MAX_BITS, NUM_TIERS, TIER_NAMES, TIERS,
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
            assert "blur_sigma" in tier, f"Tier {idx} missing 'blur_sigma'"

    def test_tier_bits_are_ordered(self):
        bits = [TIERS[i]["bits"] for i in range(NUM_TIERS)]
        assert bits == sorted(bits)

    def test_tier_strengths_are_ordered(self):
        strengths = [TIERS[i]["strength"] for i in range(NUM_TIERS)]
        assert strengths == sorted(strengths)

    def test_tier_blur_sigma_decreasing(self):
        """Higher tiers (more bits) should have lower blur sigma."""
        sigmas = [TIERS[i]["blur_sigma"] for i in range(NUM_TIERS)]
        assert sigmas == sorted(sigmas, reverse=True)

    def test_tier_values(self):
        assert TIERS[0] == {"bits": 30, "strength": 0.010, "blur_sigma": 1.4}
        assert TIERS[1] == {"bits": 48, "strength": 0.012, "blur_sigma": 1.0}
        assert TIERS[2] == {"bits": 72, "strength": 0.016, "blur_sigma": 0.8}
        assert TIERS[3] == {"bits": 96, "strength": 0.020, "blur_sigma": 0.6}

    def test_tier_names(self):
        assert TIER_NAMES == {0: "UHQ", 1: "HQ", 2: "MQ", 3: "LQ"}

    def test_embed_dim(self):
        assert EMBED_DIM == 32
