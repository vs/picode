"""DPReview forum source plugin."""

from typing import Iterator

from picode_scraper.sources.base import (
    CandidateImage,
    DiscoveredSource,
    PageContent,
    Source,
)
from picode_scraper.sources.dpreview.parser import parse_thread_page
from picode_scraper.sources.registry import register


@register("dpreview")
class DPReviewSource(Source):
    """Source plugin for DPReview forum archive."""

    name = "dpreview"
    display_name = "DPReview Forums (Archive)"

    BASE_URL = "https://www.dpreview.com"

    SUBFORUMS = [
        "/forums/post/printing-finishing",
        "/forums/post/post-processing",
    ]

    DEFAULT_SEARCH_TERMS = [
        "monitor vs print",
        "screen calibration",
        "soft proofing",
        "print comparison",
    ]

    def discover(self, search_terms: list[str]) -> Iterator[DiscoveredSource]:
        """Find relevant threads to harvest.

        NOTE: This is a placeholder. Real discovery requires HTTP requests
        which should be done by the harvester worker, not in discover().
        For now, return empty - the harvester will need to implement
        the actual search crawling.
        """
        # Discovery would search subforums with search terms
        # For now, yield nothing - actual implementation needs HTTP
        return iter([])

    def extract_images(self, url: str, html: str) -> PageContent:
        """Extract all candidate images from a thread page.

        Args:
            url: URL of the thread page
            html: HTML content of the page

        Returns:
            PageContent with all candidate images found
        """
        posts = parse_thread_page(html)

        # Flatten all images from all posts
        all_images: list[CandidateImage] = []
        position = 0
        for post in posts:
            for img in post.images:
                # Update position to be global across thread
                all_images.append(
                    CandidateImage(
                        url=img.url,
                        alt_text=img.alt_text,
                        context=post.text[:200] if post.text else None,
                        position=position,
                    )
                )
                position += 1

        return PageContent(
            url=url,
            candidate_images=all_images,
            likely_capture_type="unknown",
            metadata={"source": "dpreview", "post_count": len(posts)},
        )

    def get_pagination(self, url: str) -> list[str]:
        """Return pagination URLs for a thread.

        NOTE: This requires the HTML content which we don't have here.
        The harvester should call get_pagination_urls directly with HTML.
        This method returns empty list as a fallback.
        """
        return []
