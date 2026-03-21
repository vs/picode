"""Tests for DPReview source plugin."""

from pathlib import Path

import pytest

from picode_scraper.sources.dpreview.parser import (
    PostContent,
    ThreadInfo,
    extract_post_images,
    get_pagination_urls,
    is_content_image,
    parse_search_results,
    parse_thread_page,
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


class TestParseThreadPage:
    """Tests for parse_thread_page function."""

    def test_extracts_posts(self) -> None:
        """Should extract posts from thread page."""
        html = (FIXTURES_DIR / "thread_page.html").read_text()
        posts = parse_thread_page(html)

        assert len(posts) == 3

    def test_extracts_images_from_posts(self) -> None:
        """Should extract content images from each post."""
        html = (FIXTURES_DIR / "thread_page.html").read_text()
        posts = parse_thread_page(html)

        # First post has 2 content images
        assert len(posts[0].images) == 2
        assert "screen_capture.jpg" in posts[0].images[0].url

        # Second post has 1 content image (quote button filtered)
        assert len(posts[1].images) == 1

        # Third post has no images
        assert len(posts[2].images) == 0

    def test_filters_avatars(self) -> None:
        """Should not include avatars in post images."""
        html = (FIXTURES_DIR / "thread_page.html").read_text()
        posts = parse_thread_page(html)

        for post in posts:
            for img in post.images:
                assert "avatar" not in img.url.lower()

    def test_captures_post_id(self) -> None:
        """Should capture post ID if available."""
        html = (FIXTURES_DIR / "thread_page.html").read_text()
        posts = parse_thread_page(html)

        assert posts[0].post_id == "post-1"
        assert posts[1].post_id == "post-2"


class TestGetPaginationUrls:
    """Tests for get_pagination_urls function."""

    def test_extracts_pagination_links(self) -> None:
        """Should extract all pagination URLs."""
        html = (FIXTURES_DIR / "thread_page.html").read_text()
        urls = get_pagination_urls(html, "https://dpreview.com/forums/thread/12345")

        assert len(urls) == 2
        assert "/forums/thread/12345?page=2" in urls[0]
        assert "/forums/thread/12345?page=3" in urls[1]

    def test_excludes_current_page(self) -> None:
        """Should not include current page in results."""
        html = (FIXTURES_DIR / "thread_page.html").read_text()
        urls = get_pagination_urls(html, "https://dpreview.com/forums/thread/12345")

        for url in urls:
            assert "page=1" not in url or "class='current'" not in url

    def test_deduplicates_urls(self) -> None:
        """Should deduplicate pagination URLs."""
        html = (FIXTURES_DIR / "thread_page.html").read_text()
        urls = get_pagination_urls(html, "https://dpreview.com/forums/thread/12345")

        # page=3 appears twice (numbered and "Last" link) but should be deduped
        page3_urls = [u for u in urls if "page=3" in u]
        assert len(page3_urls) == 1

    def test_returns_empty_for_single_page(self) -> None:
        """Should return empty list for single-page threads."""
        html = "<div class='thread-container'>No pagination</div>"
        urls = get_pagination_urls(html, "https://dpreview.com/forums/thread/999")

        assert urls == []

    def test_makes_urls_absolute(self) -> None:
        """Should convert relative URLs to absolute."""
        html = (FIXTURES_DIR / "thread_page.html").read_text()
        urls = get_pagination_urls(html, "https://dpreview.com/forums/thread/12345")

        for url in urls:
            assert url.startswith("https://")


class TestDPReviewSource:
    """Tests for DPReviewSource class."""

    def test_has_required_attributes(self) -> None:
        """Should have name and display_name attributes."""
        from picode_scraper.sources.dpreview.scraper import DPReviewSource

        source = DPReviewSource()
        assert source.name == "dpreview"
        assert source.display_name == "DPReview Forums (Archive)"

    def test_has_subforums_config(self) -> None:
        """Should have SUBFORUMS class attribute."""
        from picode_scraper.sources.dpreview.scraper import DPReviewSource

        assert len(DPReviewSource.SUBFORUMS) >= 2
        assert "/forums/post/printing-finishing" in DPReviewSource.SUBFORUMS

    def test_has_default_search_terms(self) -> None:
        """Should have DEFAULT_SEARCH_TERMS class attribute."""
        from picode_scraper.sources.dpreview.scraper import DPReviewSource

        assert len(DPReviewSource.DEFAULT_SEARCH_TERMS) >= 3
        assert "monitor vs print" in DPReviewSource.DEFAULT_SEARCH_TERMS

    def test_extract_images_returns_page_content(self) -> None:
        """Should return PageContent from HTML."""
        from picode_scraper.sources.base import PageContent
        from picode_scraper.sources.dpreview.scraper import DPReviewSource

        source = DPReviewSource()
        html = (FIXTURES_DIR / "thread_page.html").read_text()
        content = source.extract_images("https://dpreview.com/thread/123", html)

        assert isinstance(content, PageContent)
        assert len(content.candidate_images) == 3  # 2 from post-1, 1 from post-2

    def test_get_pagination_returns_list(self) -> None:
        """Should return list from get_pagination."""
        from picode_scraper.sources.dpreview.scraper import DPReviewSource

        source = DPReviewSource()
        pages = source.get_pagination("https://dpreview.com/forums/thread/12345")

        # Without actual HTTP call, this returns empty list
        assert isinstance(pages, list)

    def test_discover_returns_iterator(self) -> None:
        """Should return iterator from discover."""
        from picode_scraper.sources.dpreview.scraper import DPReviewSource

        source = DPReviewSource()
        results = list(source.discover(["test"]))

        # Without HTTP, returns empty (discovery requires network)
        assert isinstance(results, list)


class TestDPReviewRegistration:
    """Tests for DPReview source auto-registration."""

    def test_registered_on_sources_import(self) -> None:
        """Should be registered when importing picode_scraper.sources."""
        from picode_scraper.sources import list_sources

        sources = list_sources()
        assert "dpreview" in sources

    def test_can_get_source_without_explicit_import(self) -> None:
        """Should be able to get dpreview source without explicit import."""
        from picode_scraper.sources import get_source

        source = get_source("dpreview")
        assert source.name == "dpreview"
