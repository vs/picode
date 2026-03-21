"""Tests for DPReview source plugin."""

import pytest

from picode_scraper.sources.dpreview.parser import is_content_image


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
