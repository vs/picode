"""Tests for Trainer."""

from pathlib import Path

import torch
from PIL import Image

from picode.training.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    LoggingConfig,
    TrainingConfig,
)
from picode.training.trainer import Trainer


def _create_test_images(path: Path, count: int = 10) -> None:
    """Create test images in a directory."""
    path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = Image.new("RGB", (400, 400), color=(i * 20, i * 15, i * 10))
        img.save(path / f"img_{i}.jpg")


class TestTrainer:
    def test_init_from_config(self, tmp_path: Path) -> None:
        _create_test_images(tmp_path / "data", count=10)

        config = Config(
            experiment_name="test",
            data=DataConfig(
                source="folder", path=str(tmp_path / "data"), batch_size=2, num_workers=0
            ),
            training=TrainingConfig(num_steps=10, num_bits=10),
            logging=LoggingConfig(backends=["console"], log_every_steps=5),
            checkpoint=CheckpointConfig(dir=str(tmp_path / "ckpt"), save_every_steps=5),
        )

        trainer = Trainer(config)
        assert trainer.global_step == 0
        assert trainer.encoder is not None
        assert trainer.decoder is not None

    def test_train_step(self, tmp_path: Path) -> None:
        _create_test_images(tmp_path / "data", count=10)

        config = Config(
            experiment_name="test",
            data=DataConfig(
                source="folder", path=str(tmp_path / "data"), batch_size=2, num_workers=0
            ),
            training=TrainingConfig(num_steps=10, num_bits=10),
            logging=LoggingConfig(backends=[], log_every_steps=100),
        )

        trainer = Trainer(config)
        images = torch.rand(2, 3, 400, 400).to(trainer.device)

        metrics = trainer._train_step(images)
        assert "loss" in metrics
        assert "loss_msg" in metrics
        assert "loss_l2" in metrics

    def test_ramp(self) -> None:
        # Test the ramping function
        # At step 0, should be 0
        assert Trainer._ramp(1.0, 100, 0) == 0.0

        # At step 50, should be 0.5 * scale
        assert Trainer._ramp(1.0, 100, 50) == 0.5

        # At step 100+, should be full scale
        assert Trainer._ramp(1.0, 100, 100) == 1.0
        assert Trainer._ramp(1.0, 100, 200) == 1.0

        # With ramp_steps=0, should be immediate
        assert Trainer._ramp(2.0, 0, 0) == 2.0
