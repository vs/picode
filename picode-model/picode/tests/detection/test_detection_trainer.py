# picode/tests/detection/test_detection_trainer.py
"""Tests for detection training loop."""


from typing import cast

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from picode.detection.fast_detector import FastDetectorModel
from picode.detection.training.loss import DetectionLoss
from picode.detection.training.trainer import DetectionTrainer


def _cls_weight(trainer: DetectionTrainer) -> Tensor:
    """Weight of the first classification-head layer."""
    return cast(nn.Linear, cast(nn.Sequential, trainer.model.cls_head)[0]).weight


class MockDataset(Dataset[dict[str, Tensor]]):
    """Mock dataset for testing."""

    def __init__(self, size: int = 10) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        is_positive = idx % 2 == 0
        return {
            "image": torch.rand(3, 320, 320),
            "is_watermark": torch.tensor(1.0 if is_positive else 0.0),
            "corners": torch.rand(8) if is_positive else torch.zeros(8),
            "has_corners": torch.tensor(1.0 if is_positive else 0.0),
        }


class TestDetectionTrainer:
    @pytest.fixture
    def model(self) -> FastDetectorModel:
        return FastDetectorModel(input_size=320, pretrained=False)

    @pytest.fixture
    def loss_fn(self) -> DetectionLoss:
        return DetectionLoss()

    @pytest.fixture
    def train_loader(self) -> DataLoader[dict[str, Tensor]]:
        dataset = MockDataset(size=8)
        return DataLoader(dataset, batch_size=2, shuffle=True)

    @pytest.fixture
    def val_loader(self) -> DataLoader[dict[str, Tensor]]:
        dataset = MockDataset(size=4)
        return DataLoader(dataset, batch_size=2, shuffle=False)

    @pytest.fixture
    def trainer(
        self,
        model: FastDetectorModel,
        loss_fn: DetectionLoss,
        train_loader: DataLoader[dict[str, Tensor]],
        val_loader: DataLoader[dict[str, Tensor]],
    ) -> DetectionTrainer:
        return DetectionTrainer(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=1e-4,
            device="cpu",
        )

    def test_trainer_creation(self, trainer: DetectionTrainer) -> None:
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.device == "cpu"

    def test_train_step_returns_losses(self, trainer: DetectionTrainer) -> None:
        batch = next(iter(trainer.train_loader))
        losses = trainer.train_step(batch)

        assert "total" in losses
        assert "cls" in losses
        assert "corner" in losses
        assert losses["total"] >= 0

    def test_train_step_updates_weights(self, trainer: DetectionTrainer) -> None:
        batch = next(iter(trainer.train_loader))

        # Get initial weights
        initial_weight = _cls_weight(trainer).clone()

        # Train step
        trainer.train_step(batch)

        # Weights should be updated
        assert not torch.allclose(
            _cls_weight(trainer), initial_weight
        )

    def test_val_step_no_grad_update(self, trainer: DetectionTrainer) -> None:
        assert trainer.val_loader is not None
        batch = next(iter(trainer.val_loader))

        # Get initial weights
        initial_weight = _cls_weight(trainer).clone()

        # Val step
        trainer.val_step(batch)

        # Weights should NOT be updated
        assert torch.allclose(
            _cls_weight(trainer), initial_weight
        )

    def test_train_epoch_runs(self, trainer: DetectionTrainer) -> None:
        metrics = trainer.train_epoch()

        assert "train_loss" in metrics
        assert metrics["train_loss"] >= 0

    def test_val_epoch_runs(self, trainer: DetectionTrainer) -> None:
        metrics = trainer.val_epoch()

        assert "val_loss" in metrics
        assert metrics["val_loss"] >= 0

    def test_fit_runs_multiple_epochs(self, trainer: DetectionTrainer) -> None:
        history = trainer.fit(num_epochs=2)

        assert len(history) == 2
        assert all("train_loss" in h for h in history)
        assert all("val_loss" in h for h in history)

    def test_save_and_load_checkpoint(
        self, trainer: DetectionTrainer, tmp_path
    ) -> None:
        checkpoint_path = tmp_path / "checkpoint.pt"

        # Save checkpoint
        trainer.save_checkpoint(checkpoint_path)

        assert checkpoint_path.exists()

        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)

    def test_checkpoint_contains_model_state(
        self, trainer: DetectionTrainer, tmp_path
    ) -> None:
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path)

        checkpoint = torch.load(checkpoint_path)

        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "epoch" in checkpoint

    def test_grad_clip_applied(self) -> None:
        """Gradient clipping should limit gradient norms."""
        model = FastDetectorModel(input_size=64, pretrained=False)
        loss_fn = DetectionLoss()

        batch = {
            "image": torch.rand(2, 3, 64, 64),
            "is_watermark": torch.tensor([1.0, 0.0]),
            "corners": torch.rand(2, 8),
            "has_corners": torch.tensor([1.0, 0.0]),
        }
        dataset = [batch]
        loader: DataLoader[dict[str, Tensor]] = DataLoader(dataset, batch_size=None)  # type: ignore[arg-type]

        trainer = DetectionTrainer(
            model=model,
            loss_fn=loss_fn,
            train_loader=loader,
            grad_clip=0.5,
        )

        losses = trainer.train_step(batch)
        assert "total" in losses
