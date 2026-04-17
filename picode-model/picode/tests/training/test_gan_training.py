"""Tests for GAN training integration."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from picode.training.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    DistortionConfig,
    GANConfig,
    LoggingConfig,
    LossConfig,
    TrainingConfig,
)
from picode.training.trainer import Trainer


class TestGANTraining:
    """Tests for GAN training in Trainer."""

    @pytest.fixture
    def gan_config(self, tmp_path: Path) -> Config:
        """Create config with GAN enabled."""
        # Create dummy image data (actual images, not tensors)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        for i in range(10):
            img = Image.fromarray(
                np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
            )
            img.save(data_dir / f"img_{i}.png")

        return Config(
            experiment_name="test_gan",
            data=DataConfig(source="folder", path=str(data_dir), batch_size=2),
            training=TrainingConfig(num_steps=10, num_bits=20),
            loss=LossConfig(
                gan_config=GANConfig(enabled=True, discriminator_lr=0.0001),
            ),
            distortion=DistortionConfig(),
            checkpoint=CheckpointConfig(dir=str(tmp_path / "ckpt")),
            logging=LoggingConfig(backends=[]),
        )

    def test_trainer_creates_discriminator_when_enabled(
        self, gan_config: Config
    ) -> None:
        """Trainer creates discriminator when GAN is enabled."""
        trainer = Trainer(gan_config)

        assert hasattr(trainer, "discriminator")
        assert trainer.discriminator is not None

    def test_trainer_no_discriminator_when_disabled(self, gan_config: Config) -> None:
        """Trainer has no discriminator when GAN is disabled."""
        gan_config.loss.gan_config.enabled = False
        trainer = Trainer(gan_config)

        assert trainer.discriminator is None

    def test_gan_loss_in_metrics(self, gan_config: Config) -> None:
        """GAN training produces loss_D metric."""
        trainer = Trainer(gan_config)
        trainer.encoder.train()
        trainer.decoder.train()

        images = trainer._get_batch()
        metrics = trainer._train_step(images)

        assert "loss_D" in metrics

    def test_discriminator_weights_clipped(self, gan_config: Config) -> None:
        """Discriminator weights are clipped after training step."""
        gan_config.loss.gan_config.clip_weights = 0.01
        trainer = Trainer(gan_config)
        trainer.encoder.train()
        trainer.decoder.train()

        # Run a training step
        images = trainer._get_batch()
        trainer._train_step(images)

        # Check that all discriminator weights are within clipping range
        clip_val = gan_config.loss.gan_config.clip_weights
        assert trainer.discriminator is not None
        for p in trainer.discriminator.parameters():
            assert p.data.min() >= -clip_val
            assert p.data.max() <= clip_val

    def test_gan_disabled_no_loss_d(self, gan_config: Config) -> None:
        """When GAN is disabled, no loss_D in metrics."""
        gan_config.loss.gan_config.enabled = False
        trainer = Trainer(gan_config)
        trainer.encoder.train()
        trainer.decoder.train()

        images = trainer._get_batch()
        metrics = trainer._train_step(images)

        # loss_D should either not exist or be 0
        assert "loss_D" not in metrics or metrics.get("loss_D", 0) == 0.0
