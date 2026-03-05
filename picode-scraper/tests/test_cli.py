"""Tests for the CLI module."""

from click.testing import CliRunner

from picode_scraper.cli import cli


def test_cli_help_shows_description() -> None:
    """Test that --help shows the scraper description."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Picode dataset scraper" in result.output


def test_cli_version_shows_version() -> None:
    """Test that --version shows the current version."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_status_command() -> None:
    """Test that status command works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "Status: No database configured" in result.output
