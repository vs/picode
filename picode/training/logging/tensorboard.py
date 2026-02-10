"""TensorBoard logger implementation."""

from typing import Any

from torch import Tensor


class TensorBoardLogger:
    """Log metrics to TensorBoard."""

    writer: Any  # SummaryWriter is untyped

    def __init__(self, log_dir: str, experiment_name: str) -> None:
        # Lazy import to avoid torch.utils.tensorboard import at module level
        from torch.utils.tensorboard import SummaryWriter

        # SummaryWriter is untyped, suppress mypy error
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{experiment_name}")  # type: ignore[no-untyped-call]

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a single scalar value."""
        self.writer.add_scalar(name, value, step)

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def log_image(self, name: str, image: Tensor, step: int) -> None:
        """Log an image tensor (C, H, W)."""
        self.writer.add_image(name, image, step)

    def close(self) -> None:
        """Flush and close the writer."""
        self.writer.close()
