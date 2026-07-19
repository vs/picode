"""Tests for PicoTier tier definitions."""

from picode.models.picotier.tiers import (
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
            assert "sobel_mask_floor" in tier, f"Tier {idx} missing 'sobel_mask_floor'"

    def test_tier_bits_are_ordered(self):
        bits = [TIERS[i]["bits"] for i in range(NUM_TIERS)]
        assert bits == sorted(bits)

    def test_tier_strengths_are_ordered(self):
        strengths = [TIERS[i]["strength"] for i in range(NUM_TIERS)]
        assert strengths == sorted(strengths)

    def test_tier_sobel_floor_increasing(self):
        """Higher tiers (more bits) should have higher or equal Sobel mask floor."""
        floors = [TIERS[i]["sobel_mask_floor"] for i in range(NUM_TIERS)]
        assert floors == sorted(floors)

    def test_tier_values(self):
        expected = {
            0: {"bits": 30, "strength": 0.008, "sobel_mask_floor": 0.75},
            1: {"bits": 48, "strength": 0.010, "sobel_mask_floor": 0.80},
            2: {"bits": 72, "strength": 0.012, "sobel_mask_floor": 0.85},
            3: {"bits": 96, "strength": 0.014, "sobel_mask_floor": 0.90},
        }
        for t_idx, values in expected.items():
            assert TIERS[t_idx] == values, f"Tier {t_idx} mismatch"

    def test_tier_names(self):
        assert TIER_NAMES == {0: "UHQ", 1: "HQ", 2: "MQ", 3: "LQ"}

    def test_embed_dim(self):
        assert EMBED_DIM == 32
