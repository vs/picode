"""Tests for export functionality."""

import csv
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from picode_scraper.export import ExportConfig, ExportPairData, ExportService


def test_export_config_defaults():
    """ExportConfig has sensible defaults."""
    config = ExportConfig(output_dir="/tmp/dataset")

    assert config.output_dir == Path("/tmp/dataset")
    assert config.train_ratio == 0.8
    assert config.val_ratio == 0.1
    assert config.test_ratio == 0.1
    assert config.min_quality_score == 0.0
    assert config.create_symlinks is True


def test_export_pair_data_creation():
    """ExportPairData holds pair information."""
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
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        ExportConfig(output_dir="/tmp", train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


class TestExportService:
    """Tests for ExportService class."""

    @pytest.fixture
    def tmp_output_dir(self, tmp_path: Path) -> Path:
        """Create temporary output directory."""
        output = tmp_path / "dataset"
        output.mkdir()
        return output

    @pytest.fixture
    def sample_pairs(self) -> list[ExportPairData]:
        """Create sample pair data for testing."""
        return [
            ExportPairData(
                pair_id="abcd1234",
                original_uri="file:///data/images/orig1.png",
                capture_uri="file:///data/images/cap1.jpg",
                capture_type="screen",
                quality_score=0.9,
                corners=[[0, 0], [100, 0], [100, 100], [0, 100]],
                source_url="https://example.com/1",
                metadata={"key": "value1"},
            ),
            ExportPairData(
                pair_id="efgh5678",
                original_uri="file:///data/images/orig2.png",
                capture_uri="file:///data/images/cap2.jpg",
                capture_type="screen",
                quality_score=0.85,
                corners=[[10, 10], [110, 10], [110, 110], [10, 110]],
                source_url="https://example.com/2",
                metadata={"key": "value2"},
            ),
            ExportPairData(
                pair_id="ijkl9012",
                original_uri="file:///data/images/orig3.png",
                capture_uri="file:///data/images/cap3.jpg",
                capture_type="photo",
                quality_score=0.75,
                corners=[[20, 20], [120, 20], [120, 120], [20, 120]],
                source_url="https://example.com/3",
                metadata={"key": "value3"},
            ),
        ]

    def test_init(self, tmp_output_dir: Path):
        """ExportService initializes with config."""
        config = ExportConfig(output_dir=tmp_output_dir)
        service = ExportService(config)

        assert service.config == config
        assert service._storage is None

    @patch("picode_scraper.export.service.get_session")
    def test_query_pairs_returns_export_pair_data(self, mock_get_session, tmp_output_dir: Path):
        """query_pairs() converts database Pairs to ExportPairData."""
        # Create mock Pair objects
        mock_pair = MagicMock()
        mock_pair.id = uuid4()
        mock_pair.capture_type = "screen"
        mock_pair.quality_score = 0.9
        mock_pair.corners = [[0, 0], [100, 0], [100, 100], [0, 100]]
        mock_pair.extra_data = {"source_url": "https://example.com"}
        mock_pair.original_image.storage_uri = "file:///data/images/orig.png"
        mock_pair.capture_image.storage_uri = "file:///data/images/cap.jpg"

        # Setup mock session
        mock_session = MagicMock()
        mock_session.query.return_value.all.return_value = [mock_pair]
        mock_get_session.return_value.__enter__.return_value = mock_session

        config = ExportConfig(output_dir=tmp_output_dir)
        service = ExportService(config)
        pairs = service.query_pairs()

        assert len(pairs) == 1
        assert pairs[0].pair_id == str(mock_pair.id)[:8]
        assert pairs[0].capture_type == "screen"
        assert pairs[0].quality_score == 0.9

    @patch("picode_scraper.export.service.get_session")
    def test_query_pairs_with_quality_filter(self, mock_get_session, tmp_output_dir: Path):
        """query_pairs() applies quality score filter."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []
        mock_get_session.return_value.__enter__.return_value = mock_session

        config = ExportConfig(output_dir=tmp_output_dir, min_quality_score=0.5)
        service = ExportService(config)
        service.query_pairs()

        # Verify filter was called since min_quality_score > 0
        mock_query.filter.assert_called_once()

    @patch("picode_scraper.export.service.get_session")
    def test_query_pairs_handles_missing_data(self, mock_get_session, tmp_output_dir: Path):
        """query_pairs() handles None values gracefully."""
        mock_pair = MagicMock()
        mock_pair.id = uuid4()
        mock_pair.capture_type = None
        mock_pair.quality_score = None
        mock_pair.corners = None
        mock_pair.extra_data = None
        mock_pair.original_image.storage_uri = "file:///data/images/orig.png"
        mock_pair.capture_image.storage_uri = "file:///data/images/cap.jpg"

        mock_session = MagicMock()
        mock_session.query.return_value.all.return_value = [mock_pair]
        mock_get_session.return_value.__enter__.return_value = mock_session

        config = ExportConfig(output_dir=tmp_output_dir)
        service = ExportService(config)
        pairs = service.query_pairs()

        assert len(pairs) == 1
        assert pairs[0].capture_type == "unknown"
        assert pairs[0].quality_score == 0.0
        assert pairs[0].corners == []
        assert pairs[0].source_url == ""
        assert pairs[0].metadata == {}

    def test_write_pair_directory_creates_structure(
        self, tmp_output_dir: Path, sample_pairs: list[ExportPairData]
    ):
        """write_pair_directory() creates expected directory structure."""
        config = ExportConfig(output_dir=tmp_output_dir)
        service = ExportService(config)

        # Mock storage backend
        mock_storage = MagicMock()
        mock_storage.load.return_value = b"fake image data"
        service._storage = mock_storage

        pair = sample_pairs[0]
        result = service.write_pair_directory(pair)

        # Verify directory created
        assert result.exists()
        assert result == tmp_output_dir / "pairs" / pair.pair_id

        # Verify files created
        assert (result / "original.png").exists()
        assert (result / "capture.jpg").exists()
        assert (result / "corners.json").exists()
        assert (result / "metadata.json").exists()

        # Verify corners.json content
        corners_data = json.loads((result / "corners.json").read_text())
        assert corners_data == pair.corners

        # Verify metadata.json content
        metadata = json.loads((result / "metadata.json").read_text())
        assert metadata["pair_id"] == pair.pair_id
        assert metadata["capture_type"] == pair.capture_type
        assert metadata["quality_score"] == pair.quality_score
        assert metadata["source_url"] == pair.source_url
        assert metadata["key"] == "value1"

    def test_generate_splits_stratifies_by_type(
        self, tmp_output_dir: Path, sample_pairs: list[ExportPairData]
    ):
        """generate_splits() stratifies by capture_type."""
        config = ExportConfig(
            output_dir=tmp_output_dir, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25
        )
        service = ExportService(config)

        # Create more pairs to test stratification
        pairs = [
            ExportPairData(
                pair_id=f"pair{i:04d}",
                original_uri="file:///o.png",
                capture_uri="file:///c.jpg",
                capture_type=t,
                quality_score=0.8,
                corners=[],
                source_url="",
            )
            for i, t in enumerate(["screen"] * 10 + ["photo"] * 10)
        ]

        train, val, test = service.generate_splits(pairs, seed=42)

        # Verify totals
        assert len(train) + len(val) + len(test) == len(pairs)

        # Verify each split has both types
        train_types = {p.capture_type for p in train}
        val_types = {p.capture_type for p in val}
        test_types = {p.capture_type for p in test}

        # With enough samples, each split should have both types
        assert "screen" in train_types
        assert "photo" in train_types

    def test_generate_splits_reproducible_with_seed(
        self, tmp_output_dir: Path, sample_pairs: list[ExportPairData]
    ):
        """generate_splits() produces same results with same seed."""
        config = ExportConfig(output_dir=tmp_output_dir)
        service = ExportService(config)

        # Create more pairs
        pairs = [
            ExportPairData(
                pair_id=f"pair{i:04d}",
                original_uri="file:///o.png",
                capture_uri="file:///c.jpg",
                capture_type="screen",
                quality_score=0.8,
                corners=[],
                source_url="",
            )
            for i in range(20)
        ]

        train1, val1, test1 = service.generate_splits(pairs, seed=42)
        train2, val2, test2 = service.generate_splits(pairs, seed=42)

        assert [p.pair_id for p in train1] == [p.pair_id for p in train2]
        assert [p.pair_id for p in val1] == [p.pair_id for p in val2]
        assert [p.pair_id for p in test1] == [p.pair_id for p in test2]

    def test_write_splits_creates_files(
        self, tmp_output_dir: Path, sample_pairs: list[ExportPairData]
    ):
        """write_splits() creates train.txt, val.txt, test.txt."""
        config = ExportConfig(output_dir=tmp_output_dir)
        service = ExportService(config)

        train = sample_pairs[:2]
        val = [sample_pairs[2]]
        test: list[ExportPairData] = []

        service.write_splits(train, val, test)

        splits_dir = tmp_output_dir / "splits"
        assert splits_dir.exists()
        assert (splits_dir / "train.txt").exists()
        assert (splits_dir / "val.txt").exists()
        assert (splits_dir / "test.txt").exists()

        # Verify content
        train_ids = (splits_dir / "train.txt").read_text().strip().split("\n")
        assert train_ids == ["abcd1234", "efgh5678"]

        val_ids = (splits_dir / "val.txt").read_text().strip().split("\n")
        assert val_ids == ["ijkl9012"]

        test_ids = (splits_dir / "test.txt").read_text()
        assert test_ids == ""

    def test_write_index_creates_csv(
        self, tmp_output_dir: Path, sample_pairs: list[ExportPairData]
    ):
        """write_index() creates index.csv with correct format."""
        config = ExportConfig(output_dir=tmp_output_dir)
        service = ExportService(config)

        service.write_index(sample_pairs)

        index_path = tmp_output_dir / "index.csv"
        assert index_path.exists()

        # Read and verify CSV
        with open(index_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["pair_id"] == "abcd1234"
        assert rows[0]["capture_type"] == "screen"
        assert rows[0]["quality_score"] == "0.9"
        assert rows[0]["source_url"] == "https://example.com/1"
        assert rows[0]["original_path"] == "pairs/abcd1234/original.png"
        assert rows[0]["capture_path"] == "pairs/abcd1234/capture.jpg"

    def test_create_type_symlinks_creates_links(
        self, tmp_output_dir: Path, sample_pairs: list[ExportPairData]
    ):
        """create_type_symlinks() creates symlinks organized by type."""
        config = ExportConfig(output_dir=tmp_output_dir, create_symlinks=True)
        service = ExportService(config)

        # Create pairs directory first (symlinks need targets)
        pairs_dir = tmp_output_dir / "pairs"
        for pair in sample_pairs:
            (pairs_dir / pair.pair_id).mkdir(parents=True)

        service.create_type_symlinks(sample_pairs)

        by_type_dir = tmp_output_dir / "by_type"
        assert by_type_dir.exists()
        assert (by_type_dir / "screen").exists()
        assert (by_type_dir / "photo").exists()

        # Verify symlinks
        screen_link = by_type_dir / "screen" / "abcd1234"
        assert screen_link.is_symlink()
        assert os.readlink(screen_link) == "../../pairs/abcd1234"

        photo_link = by_type_dir / "photo" / "ijkl9012"
        assert photo_link.is_symlink()

    def test_create_type_symlinks_respects_config(
        self, tmp_output_dir: Path, sample_pairs: list[ExportPairData]
    ):
        """create_type_symlinks() does nothing when create_symlinks=False."""
        config = ExportConfig(output_dir=tmp_output_dir, create_symlinks=False)
        service = ExportService(config)

        service.create_type_symlinks(sample_pairs)

        by_type_dir = tmp_output_dir / "by_type"
        assert not by_type_dir.exists()

    def test_create_type_symlinks_replaces_existing(
        self, tmp_output_dir: Path, sample_pairs: list[ExportPairData]
    ):
        """create_type_symlinks() replaces existing symlinks."""
        config = ExportConfig(output_dir=tmp_output_dir, create_symlinks=True)
        service = ExportService(config)

        # Create initial structure
        pairs_dir = tmp_output_dir / "pairs"
        for pair in sample_pairs:
            (pairs_dir / pair.pair_id).mkdir(parents=True)

        # Create initial symlinks
        service.create_type_symlinks(sample_pairs)

        # Create again (should not raise error)
        service.create_type_symlinks(sample_pairs)

        # Verify link still works
        screen_link = tmp_output_dir / "by_type" / "screen" / "abcd1234"
        assert screen_link.is_symlink()

    @patch("picode_scraper.export.service.get_session")
    def test_export_full_pipeline(self, mock_get_session, tmp_output_dir: Path):
        """export() runs full pipeline and returns statistics."""
        # Setup mock pairs
        mock_pairs = []
        for i in range(5):
            mock_pair = MagicMock()
            mock_pair.id = uuid4()
            mock_pair.capture_type = "screen" if i < 3 else "photo"
            mock_pair.quality_score = 0.8
            mock_pair.corners = [[0, 0], [100, 0], [100, 100], [0, 100]]
            mock_pair.extra_data = {"source_url": f"https://example.com/{i}"}
            mock_pair.original_image.storage_uri = f"file:///data/orig{i}.png"
            mock_pair.capture_image.storage_uri = f"file:///data/cap{i}.jpg"
            mock_pairs.append(mock_pair)

        mock_session = MagicMock()
        mock_session.query.return_value.all.return_value = mock_pairs
        mock_get_session.return_value.__enter__.return_value = mock_session

        config = ExportConfig(output_dir=tmp_output_dir)
        service = ExportService(config)

        # Mock storage
        mock_storage = MagicMock()
        mock_storage.load.return_value = b"fake image data"
        service._storage = mock_storage

        result = service.export(seed=42)

        assert result["total_pairs"] == 5
        assert result["train_count"] + result["val_count"] + result["test_count"] == 5

        # Verify files created
        assert (tmp_output_dir / "pairs").exists()
        assert (tmp_output_dir / "splits" / "train.txt").exists()
        assert (tmp_output_dir / "index.csv").exists()
        assert (tmp_output_dir / "by_type").exists()

    @patch("picode_scraper.export.service.get_session")
    def test_export_returns_zero_for_no_pairs(self, mock_get_session, tmp_output_dir: Path):
        """export() returns zeros when no pairs in database."""
        mock_session = MagicMock()
        mock_session.query.return_value.all.return_value = []
        mock_get_session.return_value.__enter__.return_value = mock_session

        config = ExportConfig(output_dir=tmp_output_dir)
        service = ExportService(config)

        result = service.export()

        assert result == {"total_pairs": 0, "train_count": 0, "val_count": 0, "test_count": 0}
