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

    def _validate_path(self, uri: str) -> Path:
        """Validate and extract path from URI, ensuring it's within base_path.

        Args:
            uri: file:// URI to validate

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path is outside storage directory
        """
        path = Path(uri.replace("file://", ""))
        # Resolve to absolute path and check if within base_path
        resolved = path.resolve()
        try:
            resolved.relative_to(self.base_path.resolve())
        except ValueError:
            raise ValueError(f"Path {uri} is outside storage directory")
        return resolved

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

        Raises:
            ValueError: If path is outside storage directory
        """
        path = self._validate_path(uri)
        return path.read_bytes()

    def delete(self, uri: str) -> None:
        """Delete image at file:// URI.

        Args:
            uri: file:// URI to delete
        """
        try:
            path = self._validate_path(uri)
            path.unlink(missing_ok=True)
        except ValueError:
            pass  # Silently ignore paths outside storage (for safety)
