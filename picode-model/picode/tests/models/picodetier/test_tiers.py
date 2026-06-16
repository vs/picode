"""Tests for PicodeTier tier definitions."""

from picode.models.picodetier.tiers import TIERS, NUM_TIERS, MAX_BITS, EMBED_DIM


class TestTierDefinitions:

    def test_num_tiers_matches_dict(self):
        assert NUM_TIERS == len(TIERS)

    def test_max_bits_is_largest_tier(self):
        assert MAX_BITS == max(t["bits"] for t in TIERS.values())

    def test_tiers_have_required_keys(self):
        for idx, tier in TIERS.items():
            assert "bits" in tier, f"Tier {idx} missing 'bits'"
            assert "strength" in tier, f"Tier {idx} missing 'strength'"

    def test_tier_bits_are_ordered(self):
        bits = [TIERS[i]["bits"] for i in range(NUM_TIERS)]
        assert bits == sorted(bits)

    def test_tier_strengths_are_ordered(self):
        strengths = [TIERS[i]["strength"] for i in range(NUM_TIERS)]
        assert strengths == sorted(strengths)

    def test_tier_values(self):
        assert TIERS[0] == {"bits": 16, "strength": 0.010}
        assert TIERS[1] == {"bits": 32, "strength": 0.012}
        assert TIERS[2] == {"bits": 64, "strength": 0.014}
        assert TIERS[3] == {"bits": 96, "strength": 0.016}

    def test_embed_dim(self):
        assert EMBED_DIM == 32
