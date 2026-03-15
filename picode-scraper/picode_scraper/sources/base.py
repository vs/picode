"""Source plugin base classes and data structures."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiscoveredSource:
    """A source location found during discovery."""

    url: str
    title: str
    source_type: str
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateImage:
    """An image found on a page that might be part of a pair."""

    url: str
    alt_text: str | None = None
    context: str | None = None  # Surrounding text/labels
    position: int = 0  # Order on page


@dataclass
class PageContent:
    """Parsed content from a harvested page."""

    url: str
    candidate_images: list[CandidateImage]
    likely_capture_type: str  # 'screen', 'print', 'unknown'
    metadata: dict[str, Any] = field(default_factory=dict)
