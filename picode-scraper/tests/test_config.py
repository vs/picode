"""Tests for configuration loading."""

import tempfile
from pathlib import Path

import pytest
import yaml

from picode_scraper.config import StorageConfig, ValidationConfig, load_config


def test_load_config_from_yaml() -> None:
    """Config should load from YAML file."""
    config_data = {
        "database": {
            "url": "postgresql://user:pass@localhost:5432/test",
            "pool_size": 3,
        },
        "storage": {
            "backend": "local",
            "local_path": "/tmp/test-images",
        },
        "scraping": {
            "user_agent": "TestBot/1.0",
            "request_delay": 2.0,
        },
        "validation": {
            "min_image_size": 128,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)

    try:
        config = load_config(config_path)
        assert config.database.pool_size == 3
        assert config.storage.backend == "local"
        assert config.scraping.request_delay == 2.0
        assert config.validation.min_image_size == 128
    finally:
        config_path.unlink()


def test_storage_config_validates_backend() -> None:
    """StorageConfig should reject invalid backends."""
    with pytest.raises(ValueError, match="backend must be"):
        StorageConfig(backend="invalid")


def test_storage_config_defaults() -> None:
    """StorageConfig should have sensible defaults."""
    config = StorageConfig()
    assert config.backend == "local"
    assert config.local_path == Path("/data/images")


def test_validation_config_defaults() -> None:
    """ValidationConfig should have sensible defaults."""
    config = ValidationConfig()
    assert config.min_image_size == 256
    assert config.min_corner_confidence == 0.7
    assert config.proxy_resolution == 800
