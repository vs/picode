"""Pair validation using SIFT feature matching."""

from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of pair validation.

    Attributes:
        valid: Whether the pair passed validation.
        reason: Reason for rejection (if invalid).
        corners: Detected corner coordinates [[x, y], ...].
        corner_confidence: Confidence in corner detection (0-1).
        coverage: Fraction of reference image covered by detected region.
        similarity: Visual similarity score (0-1).
        quality_score: Overall quality score combining all metrics.
    """

    valid: bool
    reason: str | None = None
    corners: list[list[float]] = field(default_factory=list)
    corner_confidence: float = 0.0
    coverage: float = 0.0
    similarity: float = 0.0
    quality_score: float = 0.0
