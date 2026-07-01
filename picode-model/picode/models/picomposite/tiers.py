"""PicoMposite tier definitions.

Single source of truth for tier bit counts, residual strengths, and embedding config.
"""

TIERS: dict[int, dict[str, float | int]] = {
    0: {"bits": 16, "strength": 0.010},
    1: {"bits": 32, "strength": 0.012},
    2: {"bits": 64, "strength": 0.014},
    3: {"bits": 96, "strength": 0.016},
}

NUM_TIERS = len(TIERS)
MAX_BITS = max(int(t["bits"]) for t in TIERS.values())
EMBED_DIM = 32
