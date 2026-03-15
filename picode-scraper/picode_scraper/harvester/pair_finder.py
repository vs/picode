"""Pair finding via validate-first approach."""

from itertools import combinations

import numpy as np

from picode_scraper.config import ValidationConfig
from picode_scraper.harvester.validator import PairValidator, ValidationResult
from picode_scraper.sources.base import CandidateImage


class PairFinder:
    """Find valid image pairs by validating all combinations.

    Uses SIFT feature matching to determine which images are
    original/capture pairs, rather than relying on HTML adjacency.
    """

    def __init__(
        self,
        config: ValidationConfig,
        max_combinations: int = 50,
    ) -> None:
        """Initialize pair finder.

        Args:
            config: Validation configuration
            max_combinations: Maximum combinations to check (performance limit)
        """
        self.validator = PairValidator(config)
        self.max_combinations = max_combinations

    def find_pairs(
        self,
        images: list[tuple[CandidateImage, bytes, np.ndarray]],
    ) -> list[tuple[tuple, tuple, ValidationResult]]:
        """Find all valid pairs among candidate images.

        Args:
            images: List of (candidate, raw_bytes, decoded_array) tuples

        Returns:
            List of (original_tuple, capture_tuple, validation_result) for valid pairs
        """
        if len(images) < 2:
            return []

        valid_pairs: list[tuple[tuple, tuple, ValidationResult]] = []

        # Generate combinations (limit to avoid O(n^2) explosion)
        combos = list(combinations(images, 2))
        if len(combos) > self.max_combinations:
            # Prioritize by position (closer images more likely to be pairs)
            combos.sort(key=lambda x: abs(x[0][0].position - x[1][0].position))
            combos = combos[: self.max_combinations]

        for img1, img2 in combos:
            candidate1, _, array1 = img1
            candidate2, _, array2 = img2

            # Try both orderings (original vs capture)
            result1 = self.validator.validate(array1, array2)
            if result1.valid:
                valid_pairs.append((img1, img2, result1))
                continue

            result2 = self.validator.validate(array2, array1)
            if result2.valid:
                valid_pairs.append((img2, img1, result2))

        return valid_pairs
