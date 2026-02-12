"""Training logging utilities."""

from torch import Tensor

from picode.training.config import LoggingConfig
from picode.training.logging.base import Logger
from picode.training.logging.console import ConsoleLogger
from picode.training.logging.tensorboard import TensorBoardLogger


class CompositeLogger:
    """Broadcast to multiple loggers."""

    def __init__(self, loggers: list[Logger]) -> None:
        self.loggers = loggers

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log to all backends."""
        for logger in self.loggers:
            logger.log_scalar(name, value, step)

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Log to all backends."""
        for logger in self.loggers:
            logger.log_scalars(metrics, step)

    def log_image(self, name: str, image: Tensor, step: int) -> None:
        """Log to all backends."""
        for logger in self.loggers:
            logger.log_image(name, image, step)

    def close(self) -> None:
        """Close all backends."""
        for logger in self.loggers:
            logger.close()


def create_logger(config: LoggingConfig, experiment_name: str) -> CompositeLogger:
    """Factory to create composite logger from config.

    Args:
        config: Logging configuration specifying backends and settings.
        experiment_name: Name of the experiment for log directories.

    Returns:
        CompositeLogger that broadcasts to all configured backends.

    Raises:
        ValueError: If an unknown backend is specified.
    """
    loggers: list[Logger] = []

    for backend in config.backends:
        if backend == "console":
            loggers.append(ConsoleLogger(config.log_every_steps))
        elif backend == "tensorboard":
            loggers.append(TensorBoardLogger(config.tensorboard_dir, experiment_name))
        else:
            raise ValueError(f"Unknown logging backend: {backend}")

    return CompositeLogger(loggers)


__all__ = [
    "CompositeLogger",
    "ConsoleLogger",
    "Logger",
    "TensorBoardLogger",
    "create_logger",
]
