"""Tests for training loggers."""

import torch

from picode.training.logging.base import Logger
from picode.training.logging.console import ConsoleLogger


class TestConsoleLogger:
    def test_implements_protocol(self) -> None:
        logger: Logger = ConsoleLogger()
        assert hasattr(logger, "log_scalar")
        assert hasattr(logger, "log_scalars")
        assert hasattr(logger, "log_image")
        assert hasattr(logger, "close")

    def test_log_scalar(self, capsys) -> None:
        logger = ConsoleLogger(log_every=1)
        logger.log_scalar("loss", 0.5, step=1)
        captured = capsys.readouterr()
        assert "loss" in captured.out
        assert "0.5" in captured.out

    def test_log_scalars(self, capsys) -> None:
        logger = ConsoleLogger(log_every=1)
        logger.log_scalars({"loss": 0.5, "acc": 0.9}, step=1)
        captured = capsys.readouterr()
        assert "loss" in captured.out
        assert "acc" in captured.out

    def test_buffering(self, capsys) -> None:
        logger = ConsoleLogger(log_every=10)
        # Log at steps 1-9 should not print
        for i in range(1, 10):
            logger.log_scalar("loss", 0.5, step=i)
        captured = capsys.readouterr()
        assert captured.out == ""

        # Log at step 10 should print with average
        logger.log_scalar("loss", 0.5, step=10)
        captured = capsys.readouterr()
        assert "step=10" in captured.out
