"""Tests for CLI."""

import tempfile
from pathlib import Path

from click.testing import CliRunner
from PIL import Image

from distortions.cli import main


def create_test_image(path: Path) -> None:
    """Create a test PNG image."""
    img = Image.new("RGB", (64, 64), color="red")
    img.save(path)


class TestCLI:
    """Tests for distort CLI."""

    def test_list_command(self):
        """--list should show available distortions."""
        runner = CliRunner()
        result = runner.invoke(main, ["--list"])
        assert result.exit_code == 0
        assert "gaussian-noise" in result.output
        assert "perspective-warp" in result.output

    def test_single_distortion(self):
        """Should apply single distortion and save outputs."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_dir = Path(tmpdir) / "output"
            create_test_image(input_path)

            result = runner.invoke(main, [
                "gaussian-noise",
                str(input_path),
                "-o", str(output_dir),
                "--intensity", "0.5",
            ])

            assert result.exit_code == 0
            assert (output_dir / "distorted.png").exists()
            assert (output_dir / "diff.png").exists()
            assert (output_dir / "comparison.png").exists()
            assert (output_dir / "grid.png").exists()

    def test_all_command(self):
        """Should apply all distortions."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_dir = Path(tmpdir) / "output"
            create_test_image(input_path)

            result = runner.invoke(main, [
                "all",
                str(input_path),
                "-o", str(output_dir),
            ])

            assert result.exit_code == 0
            # Check that subdirectories exist for each distortion
            assert (output_dir / "gaussian-noise").exists()
            assert (output_dir / "motion-blur").exists()

    def test_missing_input(self):
        """Should error on missing input file."""
        runner = CliRunner()
        result = runner.invoke(main, [
            "gaussian-noise",
            "nonexistent.png",
            "-o", "output/",
        ])
        assert result.exit_code != 0
