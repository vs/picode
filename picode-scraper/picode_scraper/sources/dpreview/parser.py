"""HTML parsing utilities for DPReview forum pages."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bs4.element import Tag

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
    classes = img.get("class", [])
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
        if width_attr and height_attr:
            w = int(width_attr)
            h = int(height_attr)
            if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
                return False
    except (ValueError, TypeError):
        pass

    return True
