"""Export service for generating training-ready datasets."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    """Service for exporting training-ready datasets."""

    pass
