"""Tests for database models."""

import uuid
from typing import Generator

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from picode_scraper.db.models import Base, HarvestTask, Image, Pair, Source


@pytest.fixture
def engine() -> Generator[Engine, None, None]:
    """Create in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Generator[Session, None, None]:
    """Create a session for testing."""
    session = Session(engine)
    yield session
    session.rollback()
    session.close()


def test_source_creation(session: Session) -> None:
    """Source model should store source information."""
    source = Source(
        source_type="dpreview",
        url="https://forums.dpreview.com/thread/123",
        title="Monitor calibration thread",
        priority=5,
        extra_data={"forum": "general"},
    )
    session.add(source)
    session.commit()

    assert source.id is not None
    assert source.source_type == "dpreview"
    assert source.discovered_at is not None


def test_harvest_task_creation(session: Session) -> None:
    """HarvestTask model should track scraping tasks."""
    source = Source(source_type="dpreview", url="https://example.com/source")
    session.add(source)
    session.commit()

    task = HarvestTask(
        source_id=source.id,
        url="https://example.com/thread/1",
        status="pending",
    )
    session.add(task)
    session.commit()

    assert task.id is not None
    assert task.status == "pending"
    assert task.retry_count == 0


def test_image_creation(session: Session) -> None:
    """Image model should store image metadata with hashes."""
    image = Image(
        source_url="https://example.com/image.jpg",
        phash=b"\x00" * 8,
        sha256=b"\x00" * 32,
        storage_uri="file:///data/images/test.jpg",
        format="jpg",
        width=1920,
        height=1080,
    )
    session.add(image)
    session.commit()

    assert image.id is not None
    assert image.format == "jpg"


def test_pair_creation(session: Session) -> None:
    """Pair model should link original and capture images."""
    # Create images
    original = Image(
        source_url="https://example.com/original.png",
        phash=b"\x01" * 8,
        sha256=b"\x01" * 32,
        storage_uri="file:///data/images/orig.png",
        format="png",
        width=1920,
        height=1080,
    )
    capture = Image(
        source_url="https://example.com/capture.jpg",
        phash=b"\x02" * 8,
        sha256=b"\x02" * 32,
        storage_uri="file:///data/images/cap.jpg",
        format="jpg",
        width=4032,
        height=3024,
    )
    session.add_all([original, capture])
    session.commit()

    # Create pair
    pair = Pair(
        original_image_id=original.id,
        capture_image_id=capture.id,
        capture_type="screen",
        quality_score=0.85,
        corners=[[0, 0], [100, 0], [100, 100], [0, 100]],
        extra_data={"source_url": "https://example.com"},
    )
    session.add(pair)
    session.commit()

    assert pair.id is not None
    assert isinstance(pair.id, uuid.UUID)
    assert pair.original_image_id == original.id
    assert pair.capture_image_id == capture.id


def test_pair_unique_constraint(session: Session) -> None:
    """Pair should enforce unique original+capture combination."""
    original = Image(
        source_url="https://example.com/o.png",
        phash=b"\x03" * 8,
        sha256=b"\x03" * 32,
        storage_uri="file:///a.png",
        format="png",
        width=100,
        height=100,
    )
    capture = Image(
        source_url="https://example.com/c.jpg",
        phash=b"\x04" * 8,
        sha256=b"\x04" * 32,
        storage_uri="file:///b.jpg",
        format="jpg",
        width=100,
        height=100,
    )
    session.add_all([original, capture])
    session.commit()

    pair1 = Pair(original_image_id=original.id, capture_image_id=capture.id)
    session.add(pair1)
    session.commit()

    # Duplicate should fail
    pair2 = Pair(original_image_id=original.id, capture_image_id=capture.id)
    session.add(pair2)
    with pytest.raises(Exception):  # IntegrityError
        session.commit()
