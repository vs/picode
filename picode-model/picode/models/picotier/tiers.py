"""PicoTier tier definitions.

Single source of truth for tier bit counts, residual strengths, Sobel mask
floors, and embedding config. Each tier targets a different quality/capacity
tradeoff with LDPC ECC applied at the application layer.

Tier summary:
    UHQ: 30 channel bits → LDPC(30,17) → 17 data bits (131K IDs), 6-char link
    HQ:  48 channel bits → LDPC(48,26) → 26 data bits (67M IDs), 9-char link
    MQ:  72 channel bits → LDPC(72,38) → 38 data bits (274B IDs), 13-char link
    LQ:  96 channel bits → LDPC(96,50) → 50 data bits (1.1Q IDs), 17-char link

Sobel mask floor design (based on b72s20m85 results):
    - sobel_mask_floor: per-tier, controls residual attenuation in smooth regions
    - Lower floor = more aggressive masking = better invisibility (UHQ)
    - Higher floor = less masking = more signal for high-capacity tiers (LQ)
    - MQ floor (0.85) matches proven b72s20m85 production model
"""

TIERS: dict[int, dict[str, float | int]] = {
    # UHQ — fewest bits, most aggressive masking for maximum invisibility
    0: {"bits": 30, "strength": 0.008, "sobel_mask_floor": 0.75},
    # HQ — balanced
    1: {"bits": 48, "strength": 0.010, "sobel_mask_floor": 0.80},
    # MQ — matches b72s20m85 production model
    2: {"bits": 72, "strength": 0.012, "sobel_mask_floor": 0.85},
    # LQ — most bits, gentlest masking to preserve signal everywhere
    3: {"bits": 96, "strength": 0.014, "sobel_mask_floor": 0.90},
}

TIER_NAMES: dict[int, str] = {0: "UHQ", 1: "HQ", 2: "MQ", 3: "LQ"}

NUM_TIERS = len(TIERS)
MAX_BITS = max(int(t["bits"]) for t in TIERS.values())
EMBED_DIM = 32
