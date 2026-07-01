"""PicoMposite tier definitions.

Single source of truth for tier bit counts, residual strengths, decoder blur,
and embedding config. Each tier targets a different quality/capacity tradeoff
with LDPC ECC applied at the application layer.

Tier summary:
    UHQ: 30 channel bits → LDPC(30,17) → 17 data bits (131K IDs), 6-char link
    HQ:  48 channel bits → LDPC(48,26) → 26 data bits (67M IDs), 9-char link
    MQ:  72 channel bits → LDPC(72,38) → 38 data bits (274B IDs), 13-char link
    LQ:  96 channel bits → LDPC(96,50) → 50 data bits (1.1Q IDs), 17-char link
"""

TIERS: dict[int, dict[str, float | int]] = {
    0: {"bits": 30, "strength": 0.010, "blur_sigma": 1.4},   # UHQ
    1: {"bits": 48, "strength": 0.012, "blur_sigma": 1.0},   # HQ
    2: {"bits": 72, "strength": 0.016, "blur_sigma": 0.8},   # MQ
    3: {"bits": 96, "strength": 0.018, "blur_sigma": 0.6},   # LQ
}

TIER_NAMES: dict[int, str] = {0: "UHQ", 1: "HQ", 2: "MQ", 3: "LQ"}

NUM_TIERS = len(TIERS)
MAX_BITS = max(int(t["bits"]) for t in TIERS.values())
EMBED_DIM = 32
