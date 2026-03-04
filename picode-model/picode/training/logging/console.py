"""Console logger implementation."""

from collections import defaultdict

from torch import Tensor


class ConsoleLogger:
    """Log metrics to stdout with buffering."""

    def __init__(self, log_every: int = 100) -> None:
        self.log_every = log_every
        self._buffer: dict[str, list[float]] = defaultdict(list)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a single scalar value."""
        self._buffer[name].append(value)
        if step % self.log_every == 0:
            self._flush(step)

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        for name, value in metrics.items():
            self._buffer[name].append(value)
        if step % self.log_every == 0:
            self._flush(step)

    def _flush(self, step: int) -> None:
        """Print buffered metrics and clear buffer."""
        parts = [f"step={step}"]
        for name, values in sorted(self._buffer.items()):
            avg = sum(values) / len(values)
            parts.append(f"{name}={avg:.4f}")
        print(" | ".join(parts))
        self._buffer.clear()

    def log_image(self, name: str, image: Tensor, step: int) -> None:
        """Console cannot display images, so this is a no-op."""
        pass

    def close(self) -> None:
        """No resources to clean up."""
        pass
