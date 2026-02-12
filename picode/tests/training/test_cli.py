"""Tests for training CLI."""

from click.testing import CliRunner

from picode.training.cli import main, parse_overrides


class TestParseOverrides:
    def test_simple_override(self) -> None:
        result = parse_overrides(["training.lr=0.001"])
        assert result == {"training": {"lr": 0.001}}

    def test_nested_override(self) -> None:
        result = parse_overrides(["loss.l2.scale=1.5"])
        assert result == {"loss": {"l2": {"scale": 1.5}}}

    def test_multiple_overrides(self) -> None:
        result = parse_overrides(["training.lr=0.001", "data.batch_size=8"])
        assert result == {"training": {"lr": 0.001}, "data": {"batch_size": 8}}

    def test_bool_override(self) -> None:
        result = parse_overrides(["distortion.enable_jpeg=false"])
        assert result == {"distortion": {"enable_jpeg": False}}


class TestCLI:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Train Picode steganography models" in result.output
