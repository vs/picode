"""Tests for storage backends."""

from pathlib import Path

import pytest

from picode_scraper.storage import create_storage_backend, detect_format
from picode_scraper.storage.local import LocalStorage
from picode_scraper.config import StorageConfig


class TestDetectFormat:
    """Tests for format detection."""

    def test_detect_png(self):
        """Should detect PNG format."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        assert detect_format(png_header) == "png"

    def test_detect_jpeg(self):
        """Should detect JPEG format."""
        jpeg_header = b"\xff\xd8\xff" + b"\x00" * 100
        assert detect_format(jpeg_header) == "jpg"

    def test_detect_webp(self):
        """Should detect WebP format."""
        webp_header = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 100
        assert detect_format(webp_header) == "webp"

    def test_unknown_defaults_to_jpg(self):
        """Unknown format should default to jpg."""
        unknown = b"\x00\x00\x00\x00" + b"\x00" * 100
        assert detect_format(unknown) == "jpg"


class TestLocalStorage:
    """Tests for local filesystem storage."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        """Create a LocalStorage instance with temp directory."""
        return LocalStorage(base_path=tmp_path)

    def test_save_returns_file_uri(self, storage: LocalStorage):
        """Save should return file:// URI."""
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        uri = storage.save(data, "png")

        assert uri.startswith("file://")
        assert uri.endswith(".png")

    def test_save_creates_file(self, storage: LocalStorage, tmp_path: Path):
        """Save should create actual file."""
        data = b"test image data"
        uri = storage.save(data, "jpg")

        path = Path(uri.replace("file://", ""))
        assert path.exists()
        assert path.read_bytes() == data

    def test_load_retrieves_data(self, storage: LocalStorage):
        """Load should retrieve saved data."""
        original_data = b"test data for load"
        uri = storage.save(original_data, "png")

        loaded_data = storage.load(uri)
        assert loaded_data == original_data

    def test_delete_removes_file(self, storage: LocalStorage):
        """Delete should remove file."""
        data = b"test data"
        uri = storage.save(data, "jpg")

        storage.delete(uri)

        path = Path(uri.replace("file://", ""))
        assert not path.exists()

    def test_delete_missing_file_no_error(self, storage: LocalStorage):
        """Delete should not error on missing file."""
        storage.delete("file:///nonexistent/path.jpg")  # Should not raise


class TestStorageFactory:
    """Tests for storage backend factory."""

    def test_creates_local_storage(self, tmp_path: Path):
        """Factory should create LocalStorage for 'local' backend."""
        config = StorageConfig(backend="local", local_path=tmp_path)
        storage = create_storage_backend(config)

        assert isinstance(storage, LocalStorage)

    def test_local_storage_uses_config_path(self, tmp_path: Path):
        """LocalStorage should use path from config."""
        config = StorageConfig(backend="local", local_path=tmp_path)
        storage = create_storage_backend(config)

        uri = storage.save(b"test", "jpg")
        assert str(tmp_path) in uri
