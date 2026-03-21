"""HTML parsing utilities for DPReview forum pages."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from picode_scraper.sources.base import CandidateImage

if TYPE_CHECKING:
    from bs4.element import Tag


@dataclass
class ThreadInfo:
    """Information about a discovered thread."""

    url: str
    title: str
    forum: str | None = None
    post_count: int | None = None


@dataclass
class PostContent:
    """Content extracted from a single forum post."""

    post_id: str | None
    images: list[CandidateImage]
    text: str | None = None

# Known UI element patterns to filter
UI_PATTERNS = (
    "avatar",
    "icon",
    "emoji",
    "smiley",
    "button",
    "/images/ui/",
    "/images/buttons/",
    "/themes/",
)

MIN_IMAGE_DIMENSION = 100


def is_content_image(img: "Tag") -> bool:
    """Determine if an image tag is actual content vs UI element.

    Args:
        img: BeautifulSoup img tag

    Returns:
        True if image appears to be content, False if avatar/icon/UI
    """
    # Check class attribute for avatar
    class_attr = img.get("class")
    classes: str | list[str] = class_attr if class_attr else []
    if isinstance(classes, list):
        class_str = " ".join(classes).lower()
    else:
        class_str = str(classes).lower()

    if "avatar" in class_str:
        return False

    # Check src for known UI patterns
    src = img.get("src")
    if not src or not isinstance(src, str):
        return False

    src_lower = src.lower()
    for pattern in UI_PATTERNS:
        if pattern in src_lower:
            return False

    # Check dimensions if available
    try:
        width_attr = img.get("width")
        height_attr = img.get("height")
        if isinstance(width_attr, str) and isinstance(height_attr, str):
            w = int(width_attr)
            h = int(height_attr)
            if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
                return False
    except (ValueError, TypeError):
        pass

    return True


def extract_post_images(html: str) -> list[CandidateImage]:
    """Extract content images from post HTML.

    Args:
        html: HTML string of a single post

    Returns:
        List of CandidateImage objects for content images
    """
    soup = BeautifulSoup(html, "lxml")
    candidates: list[CandidateImage] = []
    position = 0

    for img in soup.find_all("img", src=True):
        if not is_content_image(img):
            continue

        src = img.get("src")
        if not src or not isinstance(src, str):
            continue

        alt = img.get("alt")
        candidates.append(
            CandidateImage(
                url=src,
                alt_text=alt if isinstance(alt, str) else None,
                position=position,
            )
        )
        position += 1

    return candidates


def parse_search_results(html: str) -> list[ThreadInfo]:
    """Parse search results page to extract thread information.

    Args:
        html: HTML string of search results page

    Returns:
        List of ThreadInfo objects
    """
    soup = BeautifulSoup(html, "lxml")
    threads: list[ThreadInfo] = []

    for result in soup.select(".search-result"):
        link = result.select_one("a.thread-title")
        if not link:
            continue

        url = link.get("href")
        title = link.get_text(strip=True)
        if not url or not isinstance(url, str):
            continue

        # Extract optional metadata
        forum_elem = result.select_one(".forum-name")
        forum = forum_elem.get_text(strip=True) if forum_elem else None

        post_count_elem = result.select_one(".post-count")
        post_count: int | None = None
        if post_count_elem:
            text = post_count_elem.get_text(strip=True)
            match = re.search(r"(\d+)", text)
            if match:
                post_count = int(match.group(1))

        threads.append(
            ThreadInfo(
                url=url,
                title=title,
                forum=forum,
                post_count=post_count,
            )
        )

    return threads


def parse_thread_page(html: str) -> list[PostContent]:
    """Parse thread page to extract posts with their images.

    Args:
        html: HTML string of thread page

    Returns:
        List of PostContent objects
    """
    soup = BeautifulSoup(html, "lxml")
    posts: list[PostContent] = []

    for post_elem in soup.select(".post"):
        post_id = post_elem.get("id")

        # Get post content HTML and extract images
        content_elem = post_elem.select_one(".post-content")
        if content_elem:
            content_html = str(content_elem)
            images = extract_post_images(content_html)
            text = content_elem.get_text(strip=True)
        else:
            images = []
            text = None

        posts.append(
            PostContent(
                post_id=post_id if isinstance(post_id, str) else None,
                images=images,
                text=text[:500] if text else None,
            )
        )

    return posts


def get_pagination_urls(html: str, base_url: str) -> list[str]:
    """Extract pagination URLs from a thread page.

    Args:
        html: HTML string of thread page
        base_url: Base URL for making relative URLs absolute

    Returns:
        List of absolute URLs for other pages (excluding current)
    """
    soup = BeautifulSoup(html, "lxml")
    urls: set[str] = set()

    pagination = soup.select_one(".pagination")
    if not pagination:
        return []

    for link in pagination.find_all("a", href=True):
        # Skip current page
        class_attr = link.get("class")
        classes: str | list[str] = class_attr if class_attr else []
        if isinstance(classes, list):
            class_str = " ".join(classes).lower()
        else:
            class_str = str(classes).lower()

        if "current" in class_str:
            continue

        href = link.get("href")
        if not href or not isinstance(href, str):
            continue

        # Make URL absolute
        absolute_url = urljoin(base_url, href)
        urls.add(absolute_url)

    return sorted(urls)
