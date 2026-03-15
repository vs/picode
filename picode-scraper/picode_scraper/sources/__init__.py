"""Source plugins for picode-scraper."""

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
