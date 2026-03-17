"""Tests for export functionality."""

from pathlib import Path

import pytest


def test_export_config_defaults():
    """ExportConfig has sensible defaults."""
    from picode_scraper.export import ExportConfig

    config = ExportConfig(output_dir="/tmp/dataset")

    assert config.output_dir == Path("/tmp/dataset")
    assert config.train_ratio == 0.8
    assert config.val_ratio == 0.1
    assert config.test_ratio == 0.1
    assert config.min_quality_score == 0.0
    assert config.create_symlinks is True


def test_export_pair_data_creation():
    """ExportPairData holds pair information."""
    from picode_scraper.export import ExportPairData

    pair = ExportPairData(
        pair_id="00001",
        original_uri="file:///data/images/abc.png",
        capture_uri="file:///data/images/def.jpg",
        capture_type="screen",
        quality_score=0.85,
        corners=[[0, 0], [100, 0], [100, 100], [0, 100]],
        source_url="https://example.com/post/123",
        metadata={"author": "test"},
    )

    assert pair.pair_id == "00001"
    assert pair.capture_type == "screen"
    assert len(pair.corners) == 4


def test_export_config_validates_ratios():
    """ExportConfig validates that ratios sum to 1.0."""
    from picode_scraper.export import ExportConfig

    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        ExportConfig(output_dir="/tmp", train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
