"""Storage backend protocol and utilities."""

from typing import Protocol


class StorageBackend(Protocol):
    """Protocol for image storage backends."""

    def save(self, data: bytes, format: str) -> str:
        """Save image data and return URI."""
        ...

    def load(self, uri: str) -> bytes:
        """Load image data from URI."""
        ...

    def delete(self, uri: str) -> None:
        """Delete image at URI."""
        ...


def detect_format(data: bytes) -> str:
    """Detect image format from magic bytes.

    Args:
        data: Raw image bytes

    Returns:
        Format string: 'png', 'jpg', 'webp', or 'jpg' as default
    """
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if len(data) >= 2 and data[:2] == b"\xff\xd8":
        return "jpg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if len(data) >= 4 and data[:4] in (b"\x00\x00\x00\x0c", b"\x00\x00\x00\x18"):
        return "heic"
    return "jpg"  # Default fallback
