"""PicoComposite tier definitions.

Single source of truth for tier bit counts, residual strengths, blur sigmas,
and embedding config. Each tier targets a different quality/capacity tradeoff
with LDPC ECC applied at the application layer.

Tier summary:
    UHQ: 30 channel bits → LDPC(30,17) → 17 data bits (131K IDs), 6-char link
    HQ:  48 channel bits → LDPC(48,26) → 26 data bits (67M IDs), 9-char link
    MQ:  72 channel bits → LDPC(72,38) → 38 data bits (274B IDs), 13-char link
    LQ:  96 channel bits → LDPC(96,50) → 50 data bits (1.1Q IDs), 17-char link

Blur sigma design (based on v14-v20 experiments):
    - decoder_blur_sigma: shared across all tiers (0.5) — one decoder for mobile
    - perceptual_blur_sigma: per-tier, controls content-adaptivity during training
    - Higher perceptual σ = stronger LPIPS guidance = cleaner smooth regions
    - UHQ uses highest perceptual σ for maximum adaptivity (fewest bits, most headroom)
    - σ > 1.0 for LPIPS creates diagonal artifacts (v18/v19 lesson) — cap at 1.0
"""

TIERS: dict[int, dict[str, float | int]] = {
    # UHQ — fewest bits, highest perceptual σ for maximum content-adaptivity
    0: {"bits": 30, "strength": 0.008, "decoder_blur_sigma": 0.5, "perceptual_blur_sigma": 1.0},
    # HQ — balanced
    1: {"bits": 48, "strength": 0.010, "decoder_blur_sigma": 0.5, "perceptual_blur_sigma": 0.7},
    # MQ — matches PicoTrust v15 (72 bits)
    2: {"bits": 72, "strength": 0.012, "decoder_blur_sigma": 0.5, "perceptual_blur_sigma": 0.5},
    # LQ — most bits, lowest perceptual σ
    3: {"bits": 96, "strength": 0.014, "decoder_blur_sigma": 0.5, "perceptual_blur_sigma": 0.5},
}

TIER_NAMES: dict[int, str] = {0: "UHQ", 1: "HQ", 2: "MQ", 3: "LQ"}

NUM_TIERS = len(TIERS)
MAX_BITS = max(int(t["bits"]) for t in TIERS.values())
EMBED_DIM = 32
