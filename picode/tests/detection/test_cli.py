"""Tests for detection CLI."""

import json
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner
from PIL import Image

from picode.detection.cli import main


class TestDetectCLI:
    """Tests for detect CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def temp_image(self, tmp_path: Path) -> Path:
        """Create temporary test image."""
        img = Image.new("RGB", (400, 400), color="red")
        path = tmp_path / "test.png"
        img.save(path)
        return path

    @pytest.fixture
    def temp_checkpoint(self, tmp_path: Path) -> Path:
        """Create temporary checkpoint file."""
        from picode.models.stegastamp import Decoder

        decoder = Decoder(num_bits=100)
        path = tmp_path / "decoder.pt"
        torch.save(decoder.state_dict(), path)
        return path

    def test_help(self, runner: CliRunner) -> None:
        """CLI shows help."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Detect" in result.output or "detect" in result.output

    def test_detect_image(
        self, runner: CliRunner, temp_image: Path, temp_checkpoint: Path
    ) -> None:
        """CLI processes image file."""
        result = runner.invoke(
            main,
            [str(temp_image), "--checkpoint", str(temp_checkpoint), "--threshold", "0.01"],
        )
        # Should complete without error
        assert result.exit_code == 0

    def test_detect_with_output_json(
        self,
        runner: CliRunner,
        temp_image: Path,
        temp_checkpoint: Path,
        tmp_path: Path,
    ) -> None:
        """CLI outputs JSON when requested."""
        output_path = tmp_path / "result.json"
        result = runner.invoke(
            main,
            [
                str(temp_image),
                "--checkpoint",
                str(temp_checkpoint),
                "--output",
                str(output_path),
                "--threshold",
                "0.01",
            ],
        )
        assert result.exit_code == 0
        assert output_path.exists()

        # Validate JSON structure
        with open(output_path) as f:
            data = json.load(f)
        assert "input" in data
        assert "detections" in data

    def test_missing_checkpoint_error(self, runner: CliRunner, temp_image: Path) -> None:
        """CLI errors when checkpoint missing."""
        result = runner.invoke(main, [str(temp_image)])
        assert result.exit_code != 0
