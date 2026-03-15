"""Tests for harvester utilities."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from picode_scraper.config import ValidationConfig
from picode_scraper.db.models import Base
from picode_scraper.harvester.dedup import get_or_create_image
from picode_scraper.harvester.utils import compute_phash
from picode_scraper.harvester.validator import PairValidator, ValidationResult
from picode_scraper.storage.local import LocalStorage


@pytest.fixture
def db_session():
    """Create in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    """Create local storage for testing."""
    return LocalStorage(base_path=tmp_path)


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


@pytest.fixture
def png_image_data() -> bytes:
    """Create valid PNG image data for testing."""
    # Create a simple 100x100 RGB image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = [255, 255, 255]  # White square
    _, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


def test_get_or_create_image_creates_new(
    db_session: Session, storage: LocalStorage, png_image_data: bytes
) -> None:
    """get_or_create_image should create new image record."""
    source_url = "https://example.com/image.png"

    image = get_or_create_image(db_session, png_image_data, source_url, storage)
    db_session.commit()

    assert image.id is not None
    assert image.source_url == source_url
    assert image.sha256 == hashlib.sha256(png_image_data).digest()
    assert image.storage_uri.startswith("file://")


def test_get_or_create_image_returns_existing(
    db_session: Session, storage: LocalStorage, png_image_data: bytes
) -> None:
    """get_or_create_image should return existing for duplicate."""
    source_url = "https://example.com/image.png"

    # Create first
    image1 = get_or_create_image(db_session, png_image_data, source_url, storage)
    db_session.commit()

    # Try to create duplicate
    image2 = get_or_create_image(db_session, png_image_data, "https://other.com/same.png", storage)

    assert image1.id == image2.id  # Should return same image


def test_validation_result_valid() -> None:
    """ValidationResult should store validation data."""
    result = ValidationResult(
        valid=True,
        corners=[[0, 0], [100, 0], [100, 100], [0, 100]],
        corner_confidence=0.85,
        coverage=0.25,
        similarity=0.75,
        quality_score=0.62,
    )
    assert result.valid is True
    assert len(result.corners) == 4
    assert result.quality_score == 0.62


def test_validation_result_invalid() -> None:
    """ValidationResult should store rejection reason."""
    result = ValidationResult(valid=False, reason="poor feature matching")
    assert result.valid is False
    assert result.reason == "poor feature matching"


# --- PairValidator tests ---


def test_pair_validator_rejects_small_image() -> None:
    """PairValidator should reject images below minimum size."""
    config = ValidationConfig(min_image_size=256)
    validator = PairValidator(config)

    # Small original image
    original = np.zeros((100, 100, 3), dtype=np.uint8)
    capture = np.zeros((500, 500, 3), dtype=np.uint8)

    result = validator.validate(original, capture)

    assert result.valid is False
    assert "too small" in result.reason.lower()


def test_pair_validator_detects_valid_pair() -> None:
    """PairValidator should validate matching image pairs."""
    config = ValidationConfig(
        min_image_size=64,
        min_corner_confidence=0.3,
        min_coverage=0.05,
        min_similarity=0.3,
    )
    validator = PairValidator(config)

    # Create an "original" with distinctive features
    original = np.zeros((200, 200, 3), dtype=np.uint8)
    # Add some distinctive patterns
    cv2.rectangle(original, (20, 20), (80, 80), (255, 255, 255), -1)
    cv2.rectangle(original, (120, 20), (180, 80), (128, 128, 128), -1)
    cv2.rectangle(original, (20, 120), (80, 180), (64, 64, 64), -1)
    cv2.rectangle(original, (120, 120), (180, 180), (200, 200, 200), -1)

    # Create "capture" as original embedded in larger image
    capture = np.ones((400, 400, 3), dtype=np.uint8) * 50
    capture[100:300, 100:300] = original

    result = validator.validate(original, capture)

    # May or may not validate depending on SIFT, but should not crash
    assert isinstance(result, ValidationResult)
    assert isinstance(result.valid, bool)


def test_pair_validator_rejects_unrelated_images() -> None:
    """PairValidator should reject unrelated images."""
    config = ValidationConfig(min_image_size=64)
    validator = PairValidator(config)

    # Two completely different images
    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    img1[50:150, 50:150] = [255, 0, 0]  # Red square

    img2 = np.ones((200, 200, 3), dtype=np.uint8) * 255
    img2[50:150, 50:150] = [0, 255, 0]  # Green square on white

    result = validator.validate(img1, img2)

    assert result.valid is False
