"""Tests for CLI interface."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest
import yaml
from click.testing import CliRunner

from picode_scraper.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def config_file() -> Generator[Path, None, None]:
    """Create a temporary config file."""
    config_data = {
        "database": {
            "url": "postgresql://user:pass@localhost:5432/test",
        },
        "storage": {"backend": "local", "local_path": "/tmp/test"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)

    yield config_path
    config_path.unlink()


def test_cli_help() -> None:
    """CLI should show help message."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Picode dataset scraper" in result.output


def test_cli_version() -> None:
    """CLI should show version."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_status_without_config() -> None:
    """Status should require config and show error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 1
    assert "Config file required" in result.output or "Error" in result.output


def test_cli_with_config(config_file: Path, runner: CliRunner, mock_db: None) -> None:
    """CLI should load config when provided."""
    result = runner.invoke(cli, ["--config", str(config_file), "status"])
    assert result.exit_code == 0


def test_cli_discover_requires_config() -> None:
    """Discover command should require config."""
    runner = CliRunner()
    result = runner.invoke(cli, ["discover"])
    assert result.exit_code == 1
    assert "Config file required" in result.output or "Error" in result.output


def test_cli_harvest_requires_config() -> None:
    """Harvest command should require config."""
    runner = CliRunner()
    result = runner.invoke(cli, ["harvest"])
    assert result.exit_code == 1
    assert "Config file required" in result.output or "Error" in result.output


def test_cli_with_invalid_config() -> None:
    """CLI should show user-friendly error for invalid config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content: [")
        config_path = f.name

    runner = CliRunner()
    result = runner.invoke(cli, ["--config", config_path, "status"])

    Path(config_path).unlink()

    assert result.exit_code != 0
    assert "Error" in result.output


def test_cli_discover_lists_sources() -> None:
    """Discover without sources should list available sources."""
    runner = CliRunner()
    result = runner.invoke(cli, ["discover", "--help"])

    assert result.exit_code == 0
    assert "--source" in result.output or "-s" in result.output


def test_cli_discover_lists_available_sources() -> None:
    """Discover with --list should show available source types."""
    runner = CliRunner()
    result = runner.invoke(cli, ["discover", "--list"])

    assert result.exit_code == 0
    assert "mock" in result.output


def test_cli_harvest_has_worker_options() -> None:
    """Harvest should have worker-id and source options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["harvest", "--help"])

    assert result.exit_code == 0
    assert "--worker-id" in result.output
    assert "--source" in result.output or "-s" in result.output


@pytest.fixture
def mock_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock database connection for status command tests."""
    from contextlib import contextmanager
    from typing import Any
    from unittest.mock import MagicMock

    # Mock init_db to do nothing
    monkeypatch.setattr("picode_scraper.db.init_db", lambda config: None)

    # Create a mock session with query results
    @contextmanager
    def mock_get_session() -> Any:
        mock_session = MagicMock()

        # Mock task stats query result (returns tuples: status, count)
        mock_task_result = MagicMock()
        mock_task_result.fetchall.return_value = [
            ("pending", 10),
            ("completed", 5),
        ]

        # Mock pair count
        mock_pair_result = MagicMock()
        mock_pair_result.scalar.return_value = 42

        # Mock image count
        mock_image_result = MagicMock()
        mock_image_result.scalar.return_value = 84

        # Mock active workers
        mock_workers_result = MagicMock()
        mock_workers_result.scalar.return_value = 2

        # Set up execute to return different mocks based on query
        call_count = [0]

        def execute_side_effect(query: Any) -> Any:
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                return mock_task_result
            elif idx == 1:
                return mock_pair_result
            elif idx == 2:
                return mock_image_result
            else:
                return mock_workers_result

        mock_session.execute.side_effect = execute_side_effect
        yield mock_session

    monkeypatch.setattr("picode_scraper.db.get_session", mock_get_session)


def test_status_shows_task_counts(
    config_file: Path, runner: CliRunner, mock_db: None
) -> None:
    """Status command shows task status counts."""
    result = runner.invoke(cli, ["-c", str(config_file), "status"])
    assert result.exit_code == 0
    # Should show pending/claimed/completed/failed counts
    assert "Tasks:" in result.output


def test_status_shows_pair_counts(
    config_file: Path, runner: CliRunner, mock_db: None
) -> None:
    """Status command shows collected pair count."""
    result = runner.invoke(cli, ["-c", str(config_file), "status"])
    assert result.exit_code == 0
    assert "Pairs collected:" in result.output


def test_status_shows_image_counts(
    config_file: Path, runner: CliRunner, mock_db: None
) -> None:
    """Status command shows stored image count."""
    result = runner.invoke(cli, ["-c", str(config_file), "status"])
    assert result.exit_code == 0
    assert "Images stored:" in result.output
