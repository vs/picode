"""Tests for harvester utilities."""

import numpy as np

from picode_scraper.harvester.utils import compute_phash


def test_compute_phash_returns_8_bytes() -> None:
    """compute_phash should return 8-byte hash."""
    # Create a simple test image (100x100 RGB)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = [255, 255, 255]  # White square in center

    phash = compute_phash(img)

    assert isinstance(phash, bytes)
    assert len(phash) == 8


def test_compute_phash_similar_images() -> None:
    """Similar images should have similar hashes."""
    # Create two similar images (slight brightness change)
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img1[25:75, 25:75] = [255, 255, 255]

    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2[25:75, 25:75] = [250, 250, 250]  # Slightly darker white

    hash1 = compute_phash(img1)
    hash2 = compute_phash(img2)

    # Compute hamming distance
    distance = bin(int.from_bytes(hash1, "big") ^ int.from_bytes(hash2, "big")).count("1")
    assert distance < 10  # Similar images should be close


def test_compute_phash_different_images() -> None:
    """Different images should have different hashes."""
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img1[25:75, 25:75] = [255, 255, 255]

    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Gray image

    hash1 = compute_phash(img1)
    hash2 = compute_phash(img2)

    # Compute hamming distance
    distance = bin(int.from_bytes(hash1, "big") ^ int.from_bytes(hash2, "big")).count("1")
    assert distance > 5  # Different images should be far apart
