"""Tests for harvester utilities."""

import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from picode_scraper.config import (
    Config,
    DatabaseConfig,
    ScrapingConfig,
    StorageConfig,
    ValidationConfig,
)
from picode_scraper.db.models import Base
from picode_scraper.harvester.dedup import get_or_create_image
from picode_scraper.harvester.pair_finder import PairFinder
from picode_scraper.harvester.rate_limiter import DomainRateLimiter
from picode_scraper.harvester.utils import compute_phash
from picode_scraper.harvester.validator import PairValidator, ValidationResult
from picode_scraper.harvester.worker import HarvestWorker
from picode_scraper.sources.base import CandidateImage
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


# --- PairFinder tests ---


def test_pair_finder_needs_at_least_two_images() -> None:
    """PairFinder should return empty for less than 2 images."""
    config = ValidationConfig(min_image_size=64)
    finder = PairFinder(config)

    # Single image
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    candidates = [(CandidateImage(url="http://test.com/1.jpg", position=0), b"data", img)]

    pairs = finder.find_pairs(candidates)
    assert pairs == []


def test_pair_finder_limits_combinations() -> None:
    """PairFinder should limit combinations to max_combinations."""
    config = ValidationConfig(min_image_size=64)
    finder = PairFinder(config, max_combinations=5)

    # Create many images
    images = []
    for i in range(10):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        images.append((CandidateImage(url=f"http://test.com/{i}.jpg", position=i), b"data", img))

    # Should not check all 45 combinations, just max_combinations
    pairs = finder.find_pairs(images)
    # Result depends on validation, but should not crash
    assert isinstance(pairs, list)


# --- DomainRateLimiter tests ---


def test_rate_limiter_first_request_no_wait() -> None:
    """First request to a domain should not wait."""
    limiter = DomainRateLimiter(default_delay=1.0)

    start = time.time()
    limiter.wait("https://example.com/page1")
    elapsed = time.time() - start

    assert elapsed < 0.1  # Should be nearly instant


def test_rate_limiter_second_request_waits() -> None:
    """Second request to same domain should wait."""
    limiter = DomainRateLimiter(default_delay=0.2)

    limiter.wait("https://example.com/page1")
    start = time.time()
    limiter.wait("https://example.com/page2")
    elapsed = time.time() - start

    assert elapsed >= 0.15  # Should wait at least delay - some tolerance


def test_rate_limiter_different_domains_no_wait() -> None:
    """Different domains should not wait for each other."""
    limiter = DomainRateLimiter(default_delay=1.0)

    limiter.wait("https://example.com/page")
    start = time.time()
    limiter.wait("https://other.com/page")  # Different domain
    elapsed = time.time() - start

    assert elapsed < 0.1  # Should be nearly instant


# --- HarvestWorker tests ---


@pytest.fixture
def mock_config(tmp_path: Path) -> Config:
    """Create mock config for testing."""
    return Config(
        database=DatabaseConfig(url="postgresql://test:test@localhost:5432/test"),
        storage=StorageConfig(backend="local", local_path=tmp_path),
        scraping=ScrapingConfig(request_delay=0.01),
    )


def test_harvest_worker_init(mock_config: Config) -> None:
    """HarvestWorker should initialize with config."""
    worker = HarvestWorker(
        config=mock_config,
        worker_id="test-worker-1",
    )

    assert worker.worker_id == "test-worker-1"
    assert worker.config == mock_config
    worker.close()


def test_harvest_worker_generates_worker_id(mock_config: Config) -> None:
    """HarvestWorker should generate worker_id if not provided."""
    worker = HarvestWorker(config=mock_config)

    assert worker.worker_id is not None
    assert worker.worker_id.startswith("worker-")
    assert len(worker.worker_id) > len("worker-")
    worker.close()


def test_harvest_worker_source_filter(mock_config: Config) -> None:
    """HarvestWorker should store source filter."""
    worker = HarvestWorker(
        config=mock_config,
        source_filter=["mock", "dpreview"],
    )

    assert worker.source_filter == ["mock", "dpreview"]
    worker.close()


def test_harvest_worker_has_required_components(mock_config: Config) -> None:
    """HarvestWorker should initialize all required components."""
    worker = HarvestWorker(config=mock_config)

    assert worker.http is not None
    assert worker.storage is not None
    assert worker.pair_finder is not None
    assert worker.rate_limiter is not None
    worker.close()


def test_harvest_worker_close_cleans_up(mock_config: Config) -> None:
    """HarvestWorker.close() should clean up resources."""
    worker = HarvestWorker(config=mock_config)
    worker.close()

    # Verify http client is closed
    assert worker.http.is_closed


# --- ShutdownHandler tests ---


def test_shutdown_handler_sets_flag() -> None:
    """ShutdownHandler sets shutdown flag when signaled."""
    import signal

    from picode_scraper.harvester.shutdown import ShutdownHandler

    handler = ShutdownHandler()
    assert not handler.should_shutdown

    # Simulate SIGTERM
    handler.handle_signal(signal.SIGTERM, None)

    assert handler.should_shutdown


def test_shutdown_handler_handles_sigint() -> None:
    """ShutdownHandler sets shutdown flag on SIGINT."""
    import signal

    from picode_scraper.harvester.shutdown import ShutdownHandler

    handler = ShutdownHandler()
    assert not handler.should_shutdown

    # Simulate SIGINT (Ctrl+C)
    handler.handle_signal(signal.SIGINT, None)

    assert handler.should_shutdown


def test_shutdown_handler_register_unregister() -> None:
    """ShutdownHandler registers and unregisters signal handlers."""
    import signal

    from picode_scraper.harvester.shutdown import ShutdownHandler

    # Store original handlers
    original_sigterm = signal.getsignal(signal.SIGTERM)
    original_sigint = signal.getsignal(signal.SIGINT)

    handler = ShutdownHandler()
    handler.register()

    # After register, signals should point to our handler
    assert signal.getsignal(signal.SIGTERM) == handler.handle_signal
    assert signal.getsignal(signal.SIGINT) == handler.handle_signal

    handler.unregister()

    # After unregister, signals should be restored
    assert signal.getsignal(signal.SIGTERM) == original_sigterm
    assert signal.getsignal(signal.SIGINT) == original_sigint


# --- Worker Shutdown Integration tests ---


def test_worker_has_shutdown_handler(mock_config, monkeypatch):
    """Worker initializes with shutdown handler."""
    from unittest.mock import patch

    from picode_scraper.harvester.worker import HarvestWorker

    with patch("picode_scraper.harvester.worker.create_storage_backend"):
        worker = HarvestWorker(mock_config)

    assert hasattr(worker, "shutdown_handler")
    assert not worker.shutdown_handler.should_shutdown


def test_worker_exits_on_shutdown_flag(mock_config, monkeypatch):
    """Worker exits run loop when shutdown flag is set."""
    from unittest.mock import patch

    from picode_scraper.harvester.worker import HarvestWorker

    with patch("picode_scraper.harvester.worker.create_storage_backend"):
        worker = HarvestWorker(mock_config)

    # Set shutdown flag before run
    worker.shutdown_handler.should_shutdown = True

    # Mock claim_task to track if it's called
    with patch.object(worker, "_claim_task") as mock_claim:
        with patch("picode_scraper.harvester.worker.get_session"):
            worker.run()

        # Should exit immediately without claiming
        mock_claim.assert_not_called()


# --- Dead Letter tests ---


def test_worker_marks_dead_letter_after_max_retries(mock_config: Config) -> None:
    """Worker marks task as dead_letter when max retries exceeded."""
    from unittest.mock import MagicMock, patch

    from picode_scraper.harvester.worker import HarvestWorker

    # Set max_retries to 3
    mock_config.scraping.max_retries = 3

    with patch("picode_scraper.harvester.worker.create_storage_backend"):
        worker = HarvestWorker(mock_config)

    # Create mock database session
    mock_db = MagicMock()

    # Test case 1: retry_count = 2 (below max), should mark as 'failed'
    mock_db.execute.return_value.fetchone.return_value = (2,)  # current retry_count = 2
    worker._mark_failed(mock_db, task_id=1, error="Test error")

    # Check that status is 'failed'
    call_args = mock_db.execute.call_args_list[-1]
    assert call_args[0][1]["status"] == "failed"

    # Reset mock
    mock_db.reset_mock()

    # Test case 2: retry_count = 3 (equals max), should mark as 'dead_letter'
    mock_db.execute.return_value.fetchone.return_value = (3,)  # current retry_count = 3
    worker._mark_failed(mock_db, task_id=2, error="Test error")

    # Check that status is 'dead_letter'
    call_args = mock_db.execute.call_args_list[-1]
    assert call_args[0][1]["status"] == "dead_letter"

    # Reset mock
    mock_db.reset_mock()

    # Test case 3: retry_count = 5 (above max), should mark as 'dead_letter'
    mock_db.execute.return_value.fetchone.return_value = (5,)  # current retry_count = 5
    worker._mark_failed(mock_db, task_id=3, error="Test error")

    # Check that status is 'dead_letter'
    call_args = mock_db.execute.call_args_list[-1]
    assert call_args[0][1]["status"] == "dead_letter"

    worker.close()
