"""Source plugin registry."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from picode_scraper.sources.base import Source

_SOURCES: dict[str, type["Source"]] = {}


def register(name: str):
    """Decorator to register a source plugin.

    Usage:
        @register("dpreview")
        class DPReviewSource(Source):
            ...
    """

    def decorator(cls: type["Source"]) -> type["Source"]:
        _SOURCES[name] = cls
        return cls

    return decorator


def get_source(name: str) -> "Source":
    """Get an instance of a registered source.

    Args:
        name: Registered source name

    Returns:
        Source instance

    Raises:
        KeyError: If source name is not registered
    """
    if name not in _SOURCES:
        raise KeyError(f"Unknown source: {name}. Available: {list(_SOURCES.keys())}")
    return _SOURCES[name]()


def list_sources() -> list[str]:
    """List all registered source names."""
    return list(_SOURCES.keys())
