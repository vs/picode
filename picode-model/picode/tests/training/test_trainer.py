"""Tests for Trainer."""

from pathlib import Path

import torch
from PIL import Image

from picode.training.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    DelayedLossRamp,
    GANConfig,
    LoggingConfig,
    LossConfig,
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

    def test_delayed_ramp(self) -> None:
        # Test the delayed ramping function
        config = DelayedLossRamp(scale=1.0, ramp_steps=100, delay_steps=50)

        # Before delay: should be 0
        assert Trainer._delayed_ramp(config, 0) == 0.0
        assert Trainer._delayed_ramp(config, 25) == 0.0
        assert Trainer._delayed_ramp(config, 49) == 0.0

        # At delay start: should start ramping from 0
        assert Trainer._delayed_ramp(config, 50) == 0.0

        # During ramp: should increase linearly
        assert Trainer._delayed_ramp(config, 100) == 0.5  # 50 steps into ramp

        # After ramp complete: should be full scale
        assert Trainer._delayed_ramp(config, 150) == 1.0
        assert Trainer._delayed_ramp(config, 200) == 1.0

        # With ramp_steps=0, should be immediate after delay
        config_immediate = DelayedLossRamp(scale=2.0, ramp_steps=0, delay_steps=50)
        assert Trainer._delayed_ramp(config_immediate, 49) == 0.0
        assert Trainer._delayed_ramp(config_immediate, 50) == 2.0

    def test_init_with_gan_enabled(self, tmp_path: Path) -> None:
        """Test that GAN components are created when enabled."""
        _create_test_images(tmp_path / "data", count=10)

        config = Config(
            experiment_name="test_gan",
            model="picode_v2",
            data=DataConfig(
                source="folder", path=str(tmp_path / "data"), batch_size=2, num_workers=0
            ),
            training=TrainingConfig(num_steps=10, num_bits=10),
            loss=LossConfig(
                ffl=DelayedLossRamp(scale=0.1, ramp_steps=40, delay_steps=20),
                gan=DelayedLossRamp(scale=0.01, ramp_steps=50, delay_steps=30),
                gan_config=GANConfig(enabled=True, discriminator_lr=4e-4),
            ),
            logging=LoggingConfig(backends=[], log_every_steps=100),
        )

        trainer = Trainer(config)

        # GAN components should be created
        assert trainer.discriminator is not None
        assert trainer.optimizer_d is not None
        assert trainer._ffl_fn is not None

    def test_init_with_gan_disabled(self, tmp_path: Path) -> None:
        """Test that GAN components are NOT created when disabled."""
        _create_test_images(tmp_path / "data", count=10)

        config = Config(
            experiment_name="test_no_gan",
            model="picode_v2",
            data=DataConfig(
                source="folder", path=str(tmp_path / "data"), batch_size=2, num_workers=0
            ),
            training=TrainingConfig(num_steps=10, num_bits=10),
            loss=LossConfig(
                # GAN loss ramp is configured but gan_config.enabled is False (default)
                gan=DelayedLossRamp(scale=0.01, ramp_steps=50, delay_steps=30),
            ),
            logging=LoggingConfig(backends=[], log_every_steps=100),
        )

        trainer = Trainer(config)

        # GAN components should NOT be created
        assert trainer.discriminator is None
        assert trainer.optimizer_d is None

    def test_train_step_with_gan(self, tmp_path: Path) -> None:
        """Test training step with GAN enabled."""
        _create_test_images(tmp_path / "data", count=10)

        config = Config(
            experiment_name="test_gan_step",
            model="picode_v2",
            data=DataConfig(
                source="folder", path=str(tmp_path / "data"), batch_size=2, num_workers=0
            ),
            training=TrainingConfig(num_steps=100, num_bits=10, warmup_steps=0),
            loss=LossConfig(
                ffl=DelayedLossRamp(scale=0.1, ramp_steps=20, delay_steps=5),
                gan=DelayedLossRamp(scale=0.01, ramp_steps=20, delay_steps=5),
                gan_config=GANConfig(enabled=True, discriminator_lr=4e-4),
            ),
            logging=LoggingConfig(backends=[], log_every_steps=100),
        )

        trainer = Trainer(config)
        images = torch.rand(2, 3, 400, 400).to(trainer.device)

        # Before delay: no FFL/GAN losses
        trainer.global_step = 0
        metrics = trainer._train_step(images)
        assert "loss" in metrics
        assert "loss_msg" in metrics
        assert "loss_l2" in metrics
        assert "loss_ffl" not in metrics
        assert "loss_gan_g" not in metrics
        assert "loss_d" not in metrics

        # After delay: FFL and GAN losses should appear
        trainer.global_step = 10
        metrics = trainer._train_step(images)
        assert "loss" in metrics
        assert "loss_ffl" in metrics
        assert "loss_gan_g" in metrics
        assert "loss_d" in metrics  # Discriminator loss

    def test_should_train_discriminator(self, tmp_path: Path) -> None:
        """Test discriminator training logic."""
        _create_test_images(tmp_path / "data", count=10)

        config = Config(
            experiment_name="test_d_logic",
            model="picode_v2",
            data=DataConfig(
                source="folder", path=str(tmp_path / "data"), batch_size=2, num_workers=0
            ),
            training=TrainingConfig(num_steps=100, num_bits=10),
            loss=LossConfig(
                gan=DelayedLossRamp(scale=0.01, ramp_steps=50, delay_steps=30),
                gan_config=GANConfig(enabled=True),
            ),
            logging=LoggingConfig(backends=[], log_every_steps=100),
        )

        trainer = Trainer(config)

        # Before delay: should not train discriminator
        trainer.global_step = 0
        assert not trainer._should_train_discriminator()

        trainer.global_step = 29
        assert not trainer._should_train_discriminator()

        # At and after delay: should train discriminator
        trainer.global_step = 30
        assert trainer._should_train_discriminator()

        trainer.global_step = 100
        assert trainer._should_train_discriminator()
