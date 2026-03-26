# picode/detection/training/trainer.py
"""Training loop for FastDetector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


class DetectionTrainer:
    """Trainer for FastDetector model.

    Handles the training loop, validation, checkpointing, and metrics tracking.

    Args:
        model: FastDetectorModel to train
        loss_fn: DetectionLoss instance
        train_loader: Training data loader
        val_loader: Validation data loader
        lr: Learning rate (default 1e-4)
        weight_decay: Weight decay for optimizer (default 1e-4)
        device: Device to train on

    Example:
        >>> trainer = DetectionTrainer(model, loss_fn, train_loader, val_loader)
        >>> history = trainer.fit(num_epochs=50)
        >>> trainer.save_checkpoint("detector.pt")
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epoch = 0

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Will be updated in fit()
            eta_min=lr * 0.01,
        )

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Single training step.

        Args:
            batch: Dict with image, is_watermark, corners, has_corners

        Returns:
            Dict of loss values
        """
        self.model.train()

        # Move to device
        images = batch["image"].to(self.device)
        targets = {
            "is_watermark": batch["is_watermark"].to(self.device),
            "corners": batch["corners"].to(self.device),
            "has_corners": batch["has_corners"].to(self.device),
        }

        # Forward
        self.optimizer.zero_grad()
        outputs = self.model(images)

        # Loss
        losses = self.loss_fn(outputs, targets)

        # Backward
        losses["total"].backward()
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def val_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Single validation step.

        Args:
            batch: Dict with image, is_watermark, corners, has_corners

        Returns:
            Dict of loss values
        """
        self.model.eval()

        # Move to device
        images = batch["image"].to(self.device)
        targets = {
            "is_watermark": batch["is_watermark"].to(self.device),
            "corners": batch["corners"].to(self.device),
            "has_corners": batch["has_corners"].to(self.device),
        }

        # Forward (no grad)
        with torch.no_grad():
            outputs = self.model(images)
            losses = self.loss_fn(outputs, targets)

        return {k: v.item() for k, v in losses.items()}

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch.

        Returns:
            Dict with average training metrics
        """
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            losses = self.train_step(batch)
            total_loss += losses["total"]
            num_batches += 1

        return {"train_loss": total_loss / max(num_batches, 1)}

    def val_epoch(self) -> dict[str, float]:
        """Run one validation epoch.

        Returns:
            Dict with average validation metrics
        """
        if self.val_loader is None:
            return {"val_loss": 0.0}

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            losses = self.val_step(batch)
            total_loss += losses["total"]
            num_batches += 1

        return {"val_loss": total_loss / max(num_batches, 1)}

    def fit(
        self,
        num_epochs: int = 50,
        save_dir: str | Path | None = None,
        save_every: int = 10,
    ) -> list[dict[str, float]]:
        """Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints (optional)
            save_every: Save checkpoint every N epochs

        Returns:
            List of metrics dicts for each epoch
        """
        # Update scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=self.optimizer.param_groups[0]["lr"] * 0.01,
        )

        history = []

        for epoch in range(num_epochs):
            self.epoch = epoch + 1

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.val_epoch()

            # Update scheduler
            self.scheduler.step()

            # Combine metrics
            metrics = {**train_metrics, **val_metrics, "epoch": self.epoch}
            history.append(metrics)

            # Save checkpoint
            if save_dir is not None and (epoch + 1) % save_every == 0:
                save_path = Path(save_dir) / f"checkpoint_epoch_{self.epoch}.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                self.save_checkpoint(save_path)

        return history

    def save_checkpoint(self, path: str | Path) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "model_config": {
                "input_size": getattr(self.model, "input_size", 320),
            },
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
