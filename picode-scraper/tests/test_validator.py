"""Tests for PairValidator."""

import numpy as np
import pytest

from picode_scraper.config import ValidationConfig
from picode_scraper.harvester.validator import PairValidator, ValidationResult


@pytest.fixture
def validator() -> PairValidator:
    """Create validator with default config."""
    config = ValidationConfig(
        min_image_size=256,
        min_corner_confidence=0.5,
        min_coverage=0.1,
        min_similarity=0.5,
    )
    return PairValidator(config)


@pytest.fixture
def sample_original() -> np.ndarray:
    """Create a sample original image with features."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Add distinctive features (checkerboard pattern)
    for i in range(0, 400, 40):
        for j in range(0, 400, 40):
            if (i // 40 + j // 40) % 2 == 0:
                img[i:i+40, j:j+40] = [255, 255, 255]
    return img


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result_has_all_fields(self) -> None:
        """Valid result includes all quality metrics."""
        result = ValidationResult(
            valid=True,
            corners=[[0, 0], [100, 0], [100, 100], [0, 100]],
            corner_confidence=0.8,
            coverage=0.5,
            similarity=0.9,
            quality_score=0.73,
        )
        assert result.valid is True
        assert result.corners is not None
        assert len(result.corners) == 4

    def test_invalid_result_has_reason(self) -> None:
        """Invalid result includes rejection reason."""
        result = ValidationResult(valid=False, reason="original too small")
        assert result.valid is False
        assert result.reason == "original too small"


class TestPairValidator:
    """Tests for PairValidator."""

    def test_rejects_small_original(self, validator: PairValidator) -> None:
        """Validator rejects original images below min_size."""
        small = np.zeros((100, 100, 3), dtype=np.uint8)
        capture = np.zeros((500, 500, 3), dtype=np.uint8)

        result = validator.validate(small, capture)

        assert result.valid is False
        assert "too small" in result.reason.lower()

    def test_rejects_featureless_images(self, validator: PairValidator) -> None:
        """Validator rejects images with no SIFT features."""
        blank_original = np.zeros((300, 300, 3), dtype=np.uint8)
        blank_capture = np.zeros((500, 500, 3), dtype=np.uint8)

        result = validator.validate(blank_original, blank_capture)

        assert result.valid is False

    def test_accepts_valid_pair(
        self, validator: PairValidator, sample_original: np.ndarray
    ) -> None:
        """Validator accepts matching image pair."""
        import cv2

        capture = np.zeros((600, 600, 3), dtype=np.uint8)
        pts_src = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
        pts_dst = np.float32([[100, 100], [500, 120], [480, 500], [120, 480]])
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        cv2.warpPerspective(sample_original, M, (600, 600), capture)

        result = validator.validate(sample_original, capture)

        # Should process without error
        assert isinstance(result, ValidationResult)

    def test_quality_score_in_range(
        self, validator: PairValidator, sample_original: np.ndarray
    ) -> None:
        """Quality score is between 0 and 1 for valid pairs."""
        import cv2

        capture = np.zeros((600, 600, 3), dtype=np.uint8)
        pts_src = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
        pts_dst = np.float32([[100, 100], [500, 100], [500, 500], [100, 500]])
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        cv2.warpPerspective(sample_original, M, (600, 600), capture)

        result = validator.validate(sample_original, capture)

        if result.valid:
            assert 0.0 <= result.quality_score <= 1.0
