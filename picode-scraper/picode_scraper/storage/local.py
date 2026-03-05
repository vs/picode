"""Local filesystem storage backend."""

import uuid
from pathlib import Path


class LocalStorage:
    """Local filesystem storage backend."""

    def __init__(self, base_path: str | Path) -> None:
        """Initialize local storage.

        Args:
            base_path: Directory to store images
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, data: bytes, format: str) -> str:
        """Save image with UUID filename, preserving original format.

        Args:
            data: Raw image bytes
            format: Image format extension (e.g., 'png', 'jpg')

        Returns:
            file:// URI pointing to saved image
        """
        filename = f"{uuid.uuid4()}.{format}"
        path = self.base_path / filename
        path.write_bytes(data)
        return f"file://{path.absolute()}"

    def load(self, uri: str) -> bytes:
        """Load image from file:// URI.

        Args:
            uri: file:// URI to load

        Returns:
            Raw image bytes
        """
        path = Path(uri.replace("file://", ""))
        return path.read_bytes()

    def delete(self, uri: str) -> None:
        """Delete image at file:// URI.

        Args:
            uri: file:// URI to delete
        """
        path = Path(uri.replace("file://", ""))
        path.unlink(missing_ok=True)
