"""Tests for DPReview source plugin."""

from pathlib import Path

import pytest

from picode_scraper.sources.dpreview.parser import (
    ThreadInfo,
    extract_post_images,
    is_content_image,
    parse_search_results,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "dpreview"


class TestIsContentImage:
    """Tests for is_content_image filter."""

    def test_filters_avatar_by_class(self) -> None:
        """Should filter images with avatar class."""
        from bs4 import BeautifulSoup

        html = '<img class="avatar" src="http://test.com/img.jpg">'
        soup = BeautifulSoup(html, "lxml")
        img = soup.find("img")
        assert img is not None
        assert is_content_image(img) is False

    def test_filters_avatar_by_src(self) -> None:
        """Should filter images with avatar in src."""
        from bs4 import BeautifulSoup

        html = '<img src="http://test.com/avatars/user123.jpg">'
        soup = BeautifulSoup(html, "lxml")
        img = soup.find("img")
        assert img is not None
        assert is_content_image(img) is False

    def test_filters_small_images(self) -> None:
        """Should filter images smaller than 100px."""
        from bs4 import BeautifulSoup

        html = '<img src="http://test.com/icon.jpg" width="50" height="50">'
        soup = BeautifulSoup(html, "lxml")
        img = soup.find("img")
        assert img is not None
        assert is_content_image(img) is False

    def test_filters_ui_elements(self) -> None:
        """Should filter known UI element paths."""
        from bs4 import BeautifulSoup

        html = '<img src="http://test.com/images/buttons/reply.gif">'
        soup = BeautifulSoup(html, "lxml")
        img = soup.find("img")
        assert img is not None
        assert is_content_image(img) is False

    def test_keeps_large_content_images(self) -> None:
        """Should keep large images in post body."""
        from bs4 import BeautifulSoup

        html = '<img src="http://test.com/uploads/photo.jpg" width="800" height="600">'
        soup = BeautifulSoup(html, "lxml")
        img = soup.find("img")
        assert img is not None
        assert is_content_image(img) is True

    def test_keeps_images_without_dimensions(self) -> None:
        """Should keep images without explicit dimensions (assume content)."""
        from bs4 import BeautifulSoup

        html = '<img src="http://test.com/uploads/photo.jpg">'
        soup = BeautifulSoup(html, "lxml")
        img = soup.find("img")
        assert img is not None
        assert is_content_image(img) is True


class TestExtractPostImages:
    """Tests for extract_post_images function."""

    def test_extracts_content_images(self) -> None:
        """Should extract content images from post HTML."""
        html = """
        <div class="post-content">
            <img src="http://test.com/photo1.jpg" width="800" height="600">
            <img src="http://test.com/photo2.jpg" width="640" height="480">
        </div>
        """
        images = extract_post_images(html)
        assert len(images) == 2
        assert images[0].url == "http://test.com/photo1.jpg"
        assert images[1].url == "http://test.com/photo2.jpg"

    def test_filters_avatars_and_icons(self) -> None:
        """Should filter out avatars and icons."""
        html = """
        <div class="post-content">
            <img class="avatar" src="http://test.com/avatar.jpg">
            <img src="http://test.com/icons/quote.gif" width="16" height="16">
            <img src="http://test.com/photo.jpg" width="800" height="600">
        </div>
        """
        images = extract_post_images(html)
        assert len(images) == 1
        assert images[0].url == "http://test.com/photo.jpg"

    def test_captures_alt_text(self) -> None:
        """Should capture alt text from images."""
        html = '<img src="http://test.com/photo.jpg" alt="Before calibration">'
        images = extract_post_images(html)
        assert len(images) == 1
        assert images[0].alt_text == "Before calibration"

    def test_assigns_position(self) -> None:
        """Should assign position based on order found."""
        html = """
        <img src="http://test.com/photo1.jpg">
        <img src="http://test.com/photo2.jpg">
        """
        images = extract_post_images(html)
        assert images[0].position == 0
        assert images[1].position == 1

    def test_returns_empty_for_no_images(self) -> None:
        """Should return empty list when no images found."""
        html = "<div class='post-content'>No images here</div>"
        images = extract_post_images(html)
        assert images == []


class TestParseSearchResults:
    """Tests for parse_search_results function."""

    def test_extracts_threads_from_search(self) -> None:
        """Should extract thread URLs from search results."""
        html = (FIXTURES_DIR / "search_results.html").read_text()
        threads = parse_search_results(html)

        assert len(threads) == 3
        assert threads[0].url == "/forums/thread/12345"
        assert threads[0].title == "Monitor vs Print color comparison"

    def test_captures_thread_metadata(self) -> None:
        """Should capture forum name and post count."""
        html = (FIXTURES_DIR / "search_results.html").read_text()
        threads = parse_search_results(html)

        assert threads[0].forum == "Printing & Finishing"
        assert threads[0].post_count == 15

    def test_handles_missing_metadata(self) -> None:
        """Should handle threads with missing metadata gracefully."""
        html = """
        <div class="search-result">
            <a href="/forums/thread/999" class="thread-title">Minimal thread</a>
        </div>
        """
        threads = parse_search_results(html)

        assert len(threads) == 1
        assert threads[0].url == "/forums/thread/999"
        assert threads[0].forum is None
        assert threads[0].post_count is None

    def test_returns_empty_for_no_results(self) -> None:
        """Should return empty list when no search results."""
        html = "<div class='search-results'>No results found</div>"
        threads = parse_search_results(html)

        assert threads == []
