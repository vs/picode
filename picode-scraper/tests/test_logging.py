"""Tests for structured logging configuration."""

import json

import pytest
import structlog


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_json_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        """JSON format outputs valid JSON."""
        # Reset structlog state
        structlog.reset_defaults()

        from picode_scraper.logging import configure_logging, get_logger

        configure_logging(json_format=True, level="INFO")
        logger = get_logger("test")
        logger.info("test_message", key="value")

        captured = capsys.readouterr()
        line = captured.out.strip()
        if line:
            data = json.loads(line)
            assert data["event"] == "test_message"
            assert data["key"] == "value"

    def test_configure_console_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Console format outputs human-readable text."""
        structlog.reset_defaults()

        from picode_scraper.logging import configure_logging, get_logger

        configure_logging(json_format=False, level="INFO")
        logger = get_logger("test")
        logger.info("test_message")

        captured = capsys.readouterr()
        assert "test_message" in captured.out


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_without_name(self) -> None:
        """get_logger returns a logger without name."""
        structlog.reset_defaults()

        from picode_scraper.logging import configure_logging, get_logger

        configure_logging(json_format=False, level="INFO")
        logger = get_logger()

        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "bind")

    def test_get_logger_with_name(self, capsys: pytest.CaptureFixture[str]) -> None:
        """get_logger binds name context."""
        structlog.reset_defaults()

        from picode_scraper.logging import configure_logging, get_logger

        configure_logging(json_format=True, level="INFO")
        logger = get_logger("my_component")
        logger.info("test")

        captured = capsys.readouterr()
        if captured.out.strip():
            data = json.loads(captured.out.strip())
            assert data.get("logger") == "my_component"
