"""Confidence scoring for decoder outputs."""

import torch
from torch import Tensor


def compute_confidence(logits: Tensor) -> float:
    """Compute confidence score from decoder logits.

    Measures how "certain" the decoder is about each bit.
    On random image regions, decoder outputs uncertain bits (~0.5 probability).
    On encoded regions, bits are confident (near 0 or 1).

    Args:
        logits: (num_bits,) raw decoder output (unbounded).

    Returns:
        Confidence score in [0, 0.5] where:
        - 0.0 = maximally uncertain (all bits at 0.5 probability)
        - 0.5 = maximally confident (all bits at 0 or 1 probability)
    """
    probs = torch.sigmoid(logits)
    # Distance from uncertainty: |p - 0.5|
    confidence = torch.abs(probs - 0.5).mean().item()
    return confidence
