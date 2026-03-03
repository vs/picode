"""Tests for confidence scoring."""

import torch

from picode.detection.confidence import compute_confidence


class TestComputeConfidence:
    """Tests for compute_confidence function."""

    def test_uncertain_logits_low_confidence(self) -> None:
        """Logits near zero (prob ~0.5) should have low confidence."""
        # Logits near 0 -> sigmoid -> ~0.5 -> uncertain
        logits = torch.zeros(100)
        confidence = compute_confidence(logits)
        assert confidence < 0.1

    def test_confident_logits_high_confidence(self) -> None:
        """Logits far from zero should have high confidence."""
        # Large positive/negative logits -> sigmoid -> near 0 or 1 -> confident
        logits = torch.randn(100) * 5  # Strong signals
        confidence = compute_confidence(logits)
        assert confidence > 0.3

    def test_mixed_logits_medium_confidence(self) -> None:
        """Mix of confident and uncertain logits."""
        logits = torch.cat([
            torch.zeros(50),        # Uncertain
            torch.ones(50) * 5,     # Confident
        ])
        confidence = compute_confidence(logits)
        assert 0.1 < confidence < 0.4

    def test_output_range(self) -> None:
        """Confidence should be in [0, 0.5]."""
        for _ in range(10):
            logits = torch.randn(100) * 3
            confidence = compute_confidence(logits)
            assert 0 <= confidence <= 0.5
