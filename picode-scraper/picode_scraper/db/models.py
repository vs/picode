"""SQLAlchemy models for picode-scraper."""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for all models."""

    type_annotation_map = {
        dict[str, Any]: JSON,
        list[list[float]]: JSON,
    }


class Source(Base):
    """A source location to scrape (e.g., a forum, community)."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    title: Mapped[str | None] = mapped_column(Text)
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )
    priority: Mapped[int] = mapped_column(Integer, default=0)
    extra_data: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    # Relationships
    tasks: Mapped[list["HarvestTask"]] = relationship(back_populates="source")


class HarvestTask(Base):
    """A single page/thread to harvest."""

    __tablename__ = "harvest_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int | None] = mapped_column(ForeignKey("sources.id"))
    url: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    claimed_by: Mapped[str | None] = mapped_column(String(100))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    source: Mapped[Source | None] = relationship(back_populates="tasks")
    pairs: Mapped[list["Pair"]] = relationship(back_populates="task")

    __table_args__ = (
        Index("idx_harvest_pending", "status", postgresql_where=(status == "pending")),
    )


class Image(Base):
    """A single image with deduplication hashes."""

    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    phash: Mapped[bytes] = mapped_column(LargeBinary(8), nullable=False)
    sha256: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    format: Mapped[str] = mapped_column(String(10), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("idx_images_phash", "phash"),
        Index("idx_images_sha256", "sha256"),
        Index("idx_images_source_url", "source_url"),
    )


class Pair(Base):
    """A validated original/capture image pair."""

    __tablename__ = "pairs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    task_id: Mapped[int | None] = mapped_column(ForeignKey("harvest_tasks.id"))
    original_image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id"), nullable=False
    )
    capture_image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id"), nullable=False
    )
    capture_type: Mapped[str | None] = mapped_column(String(20))
    quality_score: Mapped[float | None] = mapped_column(Float)
    corners: Mapped[list[list[float]] | None] = mapped_column(JSON)
    extra_data: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )

    # Relationships
    task: Mapped[HarvestTask | None] = relationship(back_populates="pairs")
    original_image: Mapped[Image] = relationship(foreign_keys=[original_image_id])
    capture_image: Mapped[Image] = relationship(foreign_keys=[capture_image_id])

    __table_args__ = (
        UniqueConstraint("original_image_id", "capture_image_id", name="uq_pair_images"),
    )
