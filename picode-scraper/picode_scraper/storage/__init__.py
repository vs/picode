"""Storage backends for picode-scraper."""

from picode_scraper.config import StorageConfig
from picode_scraper.storage.base import StorageBackend, detect_format
from picode_scraper.storage.local import LocalStorage

__all__ = [
    "StorageBackend",
    "LocalStorage",
    "create_storage_backend",
    "detect_format",
]


def create_storage_backend(config: StorageConfig) -> StorageBackend:
    """Factory to create appropriate storage backend.

    Args:
        config: Storage configuration

    Returns:
        StorageBackend implementation

    Raises:
        ValueError: If backend type is not supported
    """
    if config.backend == "local":
        return LocalStorage(base_path=config.local_path)
    elif config.backend == "s3":
        # S3 storage will be implemented in a later phase
        raise NotImplementedError("S3 storage not yet implemented")
    else:
        raise ValueError(f"Unknown storage backend: {config.backend}")
