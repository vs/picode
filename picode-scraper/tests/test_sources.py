"""Tests for source plugin infrastructure."""

from typing import Iterator

from picode_scraper.sources.base import (
    CandidateImage,
    DiscoveredSource,
    PageContent,
    Source,
)
from picode_scraper.sources.registry import get_source, list_sources, register


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


def test_source_abc_requires_implementation() -> None:
    """Source ABC should require discover and extract_images."""
    import pytest

    with pytest.raises(TypeError, match="abstract"):
        Source()  # type: ignore


def test_register_decorator() -> None:
    """Register decorator should add source to registry."""

    @register("test_source")
    class TestSource(Source):
        name = "test_source"
        display_name = "Test Source"

        def discover(self, search_terms: list[str]) -> Iterator[DiscoveredSource]:
            yield DiscoveredSource(
                url="https://test.com",
                title="Test",
                source_type="test_source",
            )

        def extract_images(self, url: str, html: str) -> PageContent:
            return PageContent(url=url, candidate_images=[], likely_capture_type="unknown")

        def get_pagination(self, url: str) -> list[str]:
            return []

    assert "test_source" in list_sources()
    source = get_source("test_source")
    assert source.name == "test_source"


def test_list_sources_returns_registered() -> None:
    """list_sources should return all registered source names."""
    sources = list_sources()
    assert isinstance(sources, list)


def test_get_source_unknown_raises() -> None:
    """get_source should raise KeyError for unknown sources."""
    import pytest

    with pytest.raises(KeyError):
        get_source("nonexistent_source")


def test_mock_source_discover() -> None:
    """MockSource should yield configured sources."""
    # Import to trigger registration
    from picode_scraper.sources.mock import MockSource  # noqa: F401

    source = get_source("mock")
    results = list(source.discover(["test"]))

    assert len(results) > 0
    assert all(isinstance(r, DiscoveredSource) for r in results)


def test_mock_source_extract_images() -> None:
    """MockSource should extract candidate images."""
    from picode_scraper.sources.mock import MockSource  # noqa: F401

    source = get_source("mock")
    html = "<html><body><img src='http://test.com/img1.jpg'></body></html>"
    content = source.extract_images("http://test.com/page", html)

    assert isinstance(content, PageContent)
    assert content.url == "http://test.com/page"


def test_mock_source_filters_icons() -> None:
    """MockSource should filter out small icons and avatars."""
    from picode_scraper.sources.mock import MockSource  # noqa: F401

    source = get_source("mock")
    html = """
    <html><body>
        <img src='http://test.com/avatar.jpg'>
        <img src='http://test.com/icon.png'>
        <img src='http://test.com/emoji.gif'>
        <img src='http://test.com/small.jpg' width='50' height='50'>
        <img src='http://test.com/real_image.jpg'>
    </body></html>
    """
    content = source.extract_images("http://test.com/page", html)

    # Only real_image.jpg should pass filters
    assert len(content.candidate_images) == 1
    assert content.candidate_images[0].url == "http://test.com/real_image.jpg"


def test_mock_source_get_pagination() -> None:
    """MockSource should return empty pagination."""
    from picode_scraper.sources.mock import MockSource  # noqa: F401

    source = get_source("mock")
    pages = source.get_pagination("http://test.com/page")

    assert pages == []
