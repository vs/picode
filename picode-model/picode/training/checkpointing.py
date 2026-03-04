"""Checkpoint management for training."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from picode.training.config import CheckpointConfig, Config


@dataclass
class CheckpointState:
    """Everything needed to resume training."""

    step: int
    encoder_state: dict[str, Any]
    decoder_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    best_metric: float
    config: dict[str, Any]


class Checkpointer:
    """Manages saving and loading checkpoints."""

    def __init__(
        self,
        config: CheckpointConfig,
        experiment_name: str,
        metric_name: str = "loss",
        lower_is_better: bool = True,
    ) -> None:
        self.dir = Path(config.dir) / experiment_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_every = config.save_every_steps
        self.keep_last = config.keep_last
        self.metric_name = metric_name
        self.lower_is_better = lower_is_better

        self.best_metric = float("inf") if lower_is_better else float("-inf")
        self._periodic_checkpoints: list[Path] = []

    def should_save(self, step: int) -> bool:
        """Check if we should save a periodic checkpoint."""
        return step > 0 and step % self.save_every == 0

    def is_best(self, metrics: dict[str, float]) -> bool:
        """Check if current metrics beat the best."""
        value = metrics.get(self.metric_name, self.best_metric)
        if self.lower_is_better:
            return value < self.best_metric
        return value > self.best_metric

    def save(
        self,
        step: int,
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        config: Config,
        metrics: dict[str, float],
    ) -> None:
        """Save periodic checkpoint and update best if needed."""
        state = CheckpointState(
            step=step,
            encoder_state=encoder.state_dict(),
            decoder_state=decoder.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler else None,
            best_metric=self.best_metric,
            config=asdict(config),
        )

        # Save periodic checkpoint
        path = self.dir / f"checkpoint_{step:08d}.pt"
        torch.save(asdict(state), path)
        self._periodic_checkpoints.append(path)

        # Prune old periodic checkpoints
        while len(self._periodic_checkpoints) > self.keep_last:
            old = self._periodic_checkpoints.pop(0)
            old.unlink(missing_ok=True)

        # Update best if needed
        if self.is_best(metrics):
            self.best_metric = metrics[self.metric_name]
            best_path = self.dir / "best.pt"
            state.best_metric = self.best_metric
            torch.save(asdict(state), best_path)

    def load(self, path: str | Path | None = None) -> CheckpointState | None:
        """Load checkpoint. If path is None, try to load latest."""
        if path is None:
            path = self._find_latest()
        if path is None:
            return None

        data = torch.load(path, weights_only=False)
        return CheckpointState(**data)

    def _find_latest(self) -> Path | None:
        """Find most recent checkpoint."""
        checkpoints = sorted(self.dir.glob("checkpoint_*.pt"))
        return checkpoints[-1] if checkpoints else None

    def load_best(self) -> CheckpointState | None:
        """Load the best checkpoint."""
        best_path = self.dir / "best.pt"
        if best_path.exists():
            return self.load(best_path)
        return None
