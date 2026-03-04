"""Logger protocol definition."""

from typing import Protocol

from torch import Tensor


class Logger(Protocol):
    """Protocol for training loggers."""

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a single scalar value."""
        ...

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        ...

    def log_image(self, name: str, image: Tensor, step: int) -> None:
        """Log an image tensor."""
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...
