"""Source plugins for picode-scraper."""

# Import sources to trigger registration
from picode_scraper.sources import dpreview as _dpreview  # noqa: F401
from picode_scraper.sources import mock as _mock  # noqa: F401
from picode_scraper.sources.base import (
    CandidateImage,
    DiscoveredSource,
    PageContent,
    Source,
)
from picode_scraper.sources.registry import get_source, list_sources, register

__all__ = [
    "CandidateImage",
    "DiscoveredSource",
    "PageContent",
    "Source",
    "get_source",
    "list_sources",
    "register",
]
