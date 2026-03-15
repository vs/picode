"""Tests for source plugin infrastructure."""

from picode_scraper.sources.base import (
    CandidateImage,
    DiscoveredSource,
    PageContent,
)


def test_discovered_source_creation() -> None:
    """DiscoveredSource should store source metadata."""
    source = DiscoveredSource(
        url="https://example.com/thread/123",
        title="Test Thread",
        source_type="mock",
        priority=5,
        metadata={"forum": "general"},
    )
    assert source.url == "https://example.com/thread/123"
    assert source.source_type == "mock"
    assert source.priority == 5


def test_candidate_image_creation() -> None:
    """CandidateImage should store image metadata."""
    img = CandidateImage(
        url="https://example.com/image.jpg",
        alt_text="Test image",
        context="Before and after comparison",
        position=0,
    )
    assert img.url == "https://example.com/image.jpg"
    assert img.position == 0


def test_page_content_creation() -> None:
    """PageContent should aggregate page data."""
    candidates = [
        CandidateImage(url="https://example.com/img1.jpg", position=0),
        CandidateImage(url="https://example.com/img2.jpg", position=1),
    ]
    content = PageContent(
        url="https://example.com/page",
        candidate_images=candidates,
        likely_capture_type="screen",
        metadata={"thread_id": "123"},
    )
    assert len(content.candidate_images) == 2
    assert content.likely_capture_type == "screen"
