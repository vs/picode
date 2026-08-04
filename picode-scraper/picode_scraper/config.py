"""Configuration loading and validation."""

import os
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import AfterValidator, field_validator
from pydantic_settings import BaseSettings


def _validate_database_url(v: str) -> str:
    """Validate database URL supports PostgreSQL or SQLite."""
    if not v.startswith(("postgresql://", "sqlite://")):
        raise ValueError("Database URL must be postgresql:// or sqlite://")
    return v


DatabaseUrl = Annotated[str, AfterValidator(_validate_database_url)]


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""

    url: DatabaseUrl
    pool_size: int = 5


class StorageConfig(BaseSettings):
    """Storage backend configuration."""

    backend: str = "local"
    local_path: Path = Path("/data/images")
    s3_bucket: str | None = None
    s3_prefix: str = "images"
    s3_endpoint_url: str | None = None

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        if v not in ("local", "s3"):
            raise ValueError("backend must be 'local' or 's3'")
        return v


class ScrapingConfig(BaseSettings):
    """Web scraping configuration."""

    user_agent: str = "PicodeDatasetCollector/1.0 (research)"
    request_delay: float = 1.0
    max_retries: int = 3
    timeout: int = 30
    proxy_url: str | None = None
    proxy_rotation: bool = False


class ValidationConfig(BaseSettings):
    """Image pair validation configuration."""

    min_image_size: int = 256
    min_corner_confidence: float = 0.7
    min_coverage: float = 0.1
    min_similarity: float = 0.6
    proxy_resolution: int = 800


class Config(BaseSettings):
    """Main application configuration."""

    database: DatabaseConfig
    storage: StorageConfig = StorageConfig()
    scraping: ScrapingConfig = ScrapingConfig()
    validation: ValidationConfig = ValidationConfig()
    default_search_terms: list[str] = []

    model_config = {"env_prefix": "PICODE_", "env_nested_delimiter": "__"}


ENV_PREFIX = "PICODE_"
ENV_NESTED_DELIMITER = "__"


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``PICODE_<SECTION>__<KEY>`` environment variables onto YAML data.

    pydantic-settings gives init kwargs priority over the environment, so values
    passed to ``Config(**data)`` would otherwise shadow env vars.

    Args:
        data: Parsed YAML configuration.

    Returns:
        The same dict with environment overrides applied.
    """
    for name, value in os.environ.items():
        if not name.startswith(ENV_PREFIX) or name == f"{ENV_PREFIX}CONFIG":
            continue
        keys = name[len(ENV_PREFIX) :].lower().split(ENV_NESTED_DELIMITER)
        node = data
        for key in keys[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                child = {}
                node[key] = child
            node = child
        node[keys[-1]] = value
    return data


def load_config(path: str | Path) -> Config:
    """Load config from YAML file with environment variable overrides.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Config object with loaded settings.

    Raises:
        FileNotFoundError: If config file does not exist.

    IMPORTANT: Database URL should be set via environment variable
    PICODE_DATABASE__URL to avoid committing secrets.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path.absolute()}\n"
            "Make sure the file exists and the path is correct."
        )
    try:
        with open(config_path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file {config_path.absolute()}: {e}") from e
    return Config(**_apply_env_overrides(data))
