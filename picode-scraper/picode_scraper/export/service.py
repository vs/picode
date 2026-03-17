"""Export service for generating training-ready datasets."""

import csv
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from picode_scraper.db import Pair, get_session
from picode_scraper.storage import StorageBackend, create_storage_backend


@dataclass
class ExportConfig:
    """Configuration for dataset export."""

    output_dir: str | Path
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    min_quality_score: float = 0.0
    create_symlinks: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class ExportPairData:
    """Data for a single pair to export."""

    pair_id: str
    original_uri: str
    capture_uri: str
    capture_type: str
    quality_score: float
    corners: list[list[float]]
    source_url: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ExportService:
    """Service for exporting collected pairs to training dataset format."""

    _storage: StorageBackend | None

    def __init__(self, config: ExportConfig) -> None:
        """Initialize export service.

        Args:
            config: Export configuration settings.
        """
        self.config = config
        self._storage = None  # Lazy init

    def query_pairs(self, min_quality: float | None = None) -> list[ExportPairData]:
        """Query validated pairs from database.

        Args:
            min_quality: Minimum quality score filter. If None, uses config default.

        Returns:
            List of ExportPairData objects for all matching pairs.
        """
        min_score = min_quality if min_quality is not None else self.config.min_quality_score

        with get_session() as db:
            query = db.query(Pair)
            if min_score > 0:
                query = query.filter(Pair.quality_score >= min_score)

            pairs = query.all()

            return [
                ExportPairData(
                    pair_id=str(pair.id)[:8],  # Use first 8 chars of UUID
                    original_uri=pair.original_image.storage_uri,
                    capture_uri=pair.capture_image.storage_uri,
                    capture_type=pair.capture_type or "unknown",
                    quality_score=pair.quality_score or 0.0,
                    corners=pair.corners or [],
                    source_url=pair.extra_data.get("source_url", "") if pair.extra_data else "",
                    metadata=pair.extra_data or {},
                )
                for pair in pairs
            ]

    def write_pair_directory(self, pair: ExportPairData) -> Path:
        """Write a single pair's directory structure with images and metadata.

        Args:
            pair: The pair data to write.

        Returns:
            Path to the created pair directory.
        """
        pair_dir = Path(self.config.output_dir) / "pairs" / pair.pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Lazy init storage
        if self._storage is None:
            from picode_scraper.config import StorageConfig

            self._storage = create_storage_backend(StorageConfig(backend="local"))

        storage = self._storage  # Type narrowing for mypy

        # Copy images
        original_ext = Path(pair.original_uri).suffix or ".png"
        original_data = storage.load(pair.original_uri)
        (pair_dir / f"original{original_ext}").write_bytes(original_data)

        capture_ext = Path(pair.capture_uri).suffix or ".jpg"
        capture_data = storage.load(pair.capture_uri)
        (pair_dir / f"capture{capture_ext}").write_bytes(capture_data)

        # Write corners.json
        (pair_dir / "corners.json").write_text(json.dumps(pair.corners, indent=2))

        # Write metadata.json
        metadata = {
            "pair_id": pair.pair_id,
            "source_url": pair.source_url,
            "capture_type": pair.capture_type,
            "quality_score": pair.quality_score,
            **pair.metadata,
        }
        (pair_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        return pair_dir

    def generate_splits(
        self,
        pairs: list[ExportPairData],
        seed: int | None = None,
    ) -> tuple[list[ExportPairData], list[ExportPairData], list[ExportPairData]]:
        """Generate stratified train/val/test splits.

        Stratifies by capture_type to ensure each split has representative
        samples from each capture type.

        Args:
            pairs: List of pairs to split.
            seed: Random seed for reproducibility. If None, uses system randomness.

        Returns:
            Tuple of (train, val, test) pair lists.
        """
        if seed is not None:
            random.seed(seed)

        by_type: dict[str, list[ExportPairData]] = {}
        for pair in pairs:
            by_type.setdefault(pair.capture_type, []).append(pair)

        train: list[ExportPairData] = []
        val: list[ExportPairData] = []
        test: list[ExportPairData] = []

        for capture_type, type_pairs in by_type.items():
            shuffled = type_pairs.copy()
            random.shuffle(shuffled)

            n = len(shuffled)
            n_train = int(n * self.config.train_ratio)
            n_val = int(n * self.config.val_ratio)

            train.extend(shuffled[:n_train])
            val.extend(shuffled[n_train : n_train + n_val])
            test.extend(shuffled[n_train + n_val :])

        return train, val, test

    def write_splits(
        self,
        train: list[ExportPairData],
        val: list[ExportPairData],
        test: list[ExportPairData],
    ) -> None:
        """Write train/val/test split files.

        Creates splits/ directory with train.txt, val.txt, test.txt containing
        one pair_id per line.

        Args:
            train: Training set pairs.
            val: Validation set pairs.
            test: Test set pairs.
        """
        splits_dir = Path(self.config.output_dir) / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        for name, pairs in [("train", train), ("val", val), ("test", test)]:
            split_file = splits_dir / f"{name}.txt"
            split_file.write_text("\n".join(p.pair_id for p in pairs))

    def write_index(self, pairs: list[ExportPairData]) -> None:
        """Write global index.csv with all pair metadata.

        Args:
            pairs: List of all exported pairs.
        """
        index_path = Path(self.config.output_dir) / "index.csv"

        fieldnames = [
            "pair_id",
            "capture_type",
            "quality_score",
            "source_url",
            "original_path",
            "capture_path",
        ]

        with open(index_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for pair in pairs:
                writer.writerow(
                    {
                        "pair_id": pair.pair_id,
                        "capture_type": pair.capture_type,
                        "quality_score": str(pair.quality_score),
                        "source_url": pair.source_url,
                        "original_path": f"pairs/{pair.pair_id}/original.png",
                        "capture_path": f"pairs/{pair.pair_id}/capture.jpg",
                    }
                )

    def create_type_symlinks(self, pairs: list[ExportPairData]) -> None:
        """Create by_type/ symlinks for organizing pairs by capture type.

        Creates symlinks from by_type/<capture_type>/<pair_id> to
        ../../pairs/<pair_id> for easy access by type.

        Args:
            pairs: List of all exported pairs.
        """
        if not self.config.create_symlinks:
            return

        by_type_dir = Path(self.config.output_dir) / "by_type"

        for pair in pairs:
            type_dir = by_type_dir / pair.capture_type
            type_dir.mkdir(parents=True, exist_ok=True)

            link_path = type_dir / pair.pair_id
            target = Path("..") / ".." / "pairs" / pair.pair_id

            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()

            os.symlink(target, link_path)

    def export(self, seed: int | None = None) -> dict[str, Any]:
        """Run full export pipeline.

        Queries pairs, generates splits, writes directories, and creates
        all metadata files and symlinks.

        Args:
            seed: Random seed for split generation.

        Returns:
            Dictionary with export statistics including total_pairs,
            train_count, val_count, and test_count.
        """
        pairs = self.query_pairs()

        if not pairs:
            return {"total_pairs": 0, "train_count": 0, "val_count": 0, "test_count": 0}

        train, val, test = self.generate_splits(pairs, seed=seed)

        for pair in pairs:
            self.write_pair_directory(pair)

        self.write_splits(train, val, test)
        self.write_index(pairs)

        if self.config.create_symlinks:
            self.create_type_symlinks(pairs)

        return {
            "total_pairs": len(pairs),
            "train_count": len(train),
            "val_count": len(val),
            "test_count": len(test),
        }
