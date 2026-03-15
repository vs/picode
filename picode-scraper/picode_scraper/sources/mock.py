"""Mock source for testing."""

from typing import TYPE_CHECKING, Iterator

from bs4 import BeautifulSoup

from picode_scraper.sources.base import (
    CandidateImage,
    DiscoveredSource,
    PageContent,
    Source,
)
from picode_scraper.sources.registry import register

if TYPE_CHECKING:
    from bs4.element import Tag


@register("mock")
class MockSource(Source):
    """Mock source for testing and development."""

    name = "mock"
    display_name = "Mock Source"

    def discover(self, search_terms: list[str]) -> Iterator[DiscoveredSource]:
        """Yield mock discovered sources."""
        for i, term in enumerate(search_terms):
            yield DiscoveredSource(
                url=f"https://mock.example.com/thread/{i}",
                title=f"Mock thread for: {term}",
                source_type="mock",
                priority=i,
                metadata={"search_term": term},
            )

    def extract_images(self, url: str, html: str) -> PageContent:
        """Extract images from HTML."""
        soup = BeautifulSoup(html, "lxml")
        candidates: list[CandidateImage] = []

        for i, img in enumerate(soup.find_all("img", src=True)):
            src = img.get("src")
            if not src or not isinstance(src, str):
                continue

            # Skip small icons/avatars
            if self._is_content_image(img):
                alt = img.get("alt")
                candidates.append(
                    CandidateImage(
                        url=src,
                        alt_text=alt if isinstance(alt, str) else None,
                        context=self._get_surrounding_text(img),
                        position=i,
                    )
                )

        return PageContent(
            url=url,
            candidate_images=candidates,
            likely_capture_type="unknown",
            metadata={"parser": "mock"},
        )

    def get_pagination(self, url: str) -> list[str]:
        """Return empty pagination (single page)."""
        return []

    def _is_content_image(self, img: "Tag") -> bool:
        """Filter avatars, icons, emoticons."""
        src = img.get("src")
        if not src or not isinstance(src, str):
            return False
        src_lower = src.lower()
        if any(x in src_lower for x in ["avatar", "icon", "emoji", "smiley"]):
            return False
        # Check dimensions if available
        try:
            width_attr = img.get("width")
            height_attr = img.get("height")
            w = int(width_attr) if width_attr and isinstance(width_attr, str) else 999
            h = int(height_attr) if height_attr and isinstance(height_attr, str) else 999
            if w < 100 or h < 100:
                return False
        except (ValueError, TypeError):
            pass
        return True

    def _get_surrounding_text(self, img: "Tag") -> str | None:
        """Get text near the image."""
        parent = img.parent
        if parent:
            text = parent.get_text(strip=True)
            return text[:200] if text else None
        return None
