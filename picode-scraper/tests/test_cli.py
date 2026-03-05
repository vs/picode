"""Tests for CLI interface."""

import tempfile
from pathlib import Path

import yaml
from click.testing import CliRunner

from picode_scraper.cli import cli


def test_cli_help():
    """CLI should show help message."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Picode dataset scraper" in result.output


def test_cli_version():
    """CLI should show version."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_status_without_config():
    """Status should work without config (shows warning)."""
    runner = CliRunner()
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "No config" in result.output or "Status" in result.output


def test_cli_with_config():
    """CLI should load config when provided."""
    config_data = {
        "database": {
            "url": "postgresql://user:pass@localhost:5432/test",
        },
        "storage": {"backend": "local", "local_path": "/tmp/test"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    runner = CliRunner()
    result = runner.invoke(cli, ["--config", config_path, "status"])

    Path(config_path).unlink()

    assert result.exit_code == 0


def test_cli_discover_requires_config():
    """Discover command should require config."""
    runner = CliRunner()
    result = runner.invoke(cli, ["discover"])
    assert result.exit_code == 1
    assert "Config file required" in result.output or "Error" in result.output


def test_cli_harvest_requires_config():
    """Harvest command should require config."""
    runner = CliRunner()
    result = runner.invoke(cli, ["harvest"])
    assert result.exit_code == 1
    assert "Config file required" in result.output or "Error" in result.output
