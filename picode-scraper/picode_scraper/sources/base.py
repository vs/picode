"""Source plugin base classes and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


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


class Source(ABC):
    """Abstract base for all source plugins."""

    name: str  # e.g., 'dpreview'
    display_name: str  # e.g., 'DPReview Forums'

    @abstractmethod
    def discover(self, search_terms: list[str]) -> Iterator[DiscoveredSource]:
        """Find relevant threads/pages to harvest."""
        pass

    @abstractmethod
    def extract_images(self, url: str, html: str) -> PageContent:
        """Extract all candidate images from a page.

        NOTE: This does NOT identify pairs. It returns all substantial
        images found. The PairFinder will validate all combinations
        using SIFT feature matching to find actual pairs.
        """
        pass

    @abstractmethod
    def get_pagination(self, url: str) -> list[str]:
        """Return all page URLs for multi-page threads."""
        pass
