"""Tests for training configuration."""

from pathlib import Path

from picode.training.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    DelayedLossRamp,
    DistortionConfig,
    DistortionRamp,
    GANConfig,
    LoggingConfig,
    LossConfig,
    LossRamp,
    TrainingConfig,
    load_config,
)


class TestLossRamp:
    def test_defaults(self) -> None:
        ramp = LossRamp(scale=1.0)
        assert ramp.scale == 1.0
        assert ramp.ramp_steps == 0

    def test_with_ramp_steps(self) -> None:
        ramp = LossRamp(scale=2.0, ramp_steps=10000)
        assert ramp.scale == 2.0
        assert ramp.ramp_steps == 10000


class TestDelayedLossRamp:
    def test_defaults(self) -> None:
        ramp = DelayedLossRamp(scale=1.0)
        assert ramp.scale == 1.0
        assert ramp.ramp_steps == 40000
        assert ramp.delay_steps == 20000

    def test_custom_values(self) -> None:
        ramp = DelayedLossRamp(scale=0.1, ramp_steps=50000, delay_steps=30000)
        assert ramp.scale == 0.1
        assert ramp.ramp_steps == 50000
        assert ramp.delay_steps == 30000


class TestGANConfig:
    def test_defaults(self) -> None:
        cfg = GANConfig()
        assert cfg.enabled is False
        assert cfg.discriminator_lr == 4e-4
        assert cfg.lambda_gp == 10.0
        assert cfg.n_critic == 1
        assert cfg.clip_weights == 0.01

    def test_enabled(self) -> None:
        cfg = GANConfig(enabled=True, discriminator_lr=1e-4, lambda_gp=5.0)
        assert cfg.enabled is True
        assert cfg.discriminator_lr == 1e-4
        assert cfg.lambda_gp == 5.0

    def test_clip_weights(self) -> None:
        cfg = GANConfig(enabled=True, clip_weights=0.05)
        assert cfg.clip_weights == 0.05


class TestDistortionRamp:
    def test_defaults(self) -> None:
        ramp = DistortionRamp(strength=0.5)
        assert ramp.strength == 0.5
        assert ramp.ramp_steps == 1000

    def test_custom_ramp(self) -> None:
        ramp = DistortionRamp(strength=0.1, ramp_steps=5000)
        assert ramp.ramp_steps == 5000


class TestLossConfig:
    def test_defaults(self) -> None:
        cfg = LossConfig()
        assert cfg.message.scale == 7.0  # StegaStamp uses ~7x higher message loss
        assert cfg.l2.scale == 1.0
        assert cfg.l2.ramp_steps == 20000
        assert cfg.lpips.scale == 1.5
        assert cfg.l2_edge_gain == 10.0
        assert cfg.l2_edge_delay_steps == 60000
        assert cfg.yuv_weights == (1.0, 1.0, 1.0)
        assert cfg.ffl is None
        assert cfg.gan is None
        assert cfg.gan_config.enabled is False

    def test_with_ffl_and_gan(self) -> None:
        cfg = LossConfig(
            ffl=DelayedLossRamp(scale=0.1, ramp_steps=40000, delay_steps=20000),
            gan=DelayedLossRamp(scale=0.01, ramp_steps=50000, delay_steps=30000),
            gan_config=GANConfig(enabled=True),
        )
        assert cfg.ffl is not None
        assert cfg.ffl.scale == 0.1
        assert cfg.ffl.delay_steps == 20000
        assert cfg.gan is not None
        assert cfg.gan.scale == 0.01
        assert cfg.gan.delay_steps == 30000
        assert cfg.gan_config.enabled is True


class TestDistortionConfig:
    def test_defaults(self) -> None:
        cfg = DistortionConfig()
        assert cfg.strategy == "curriculum"
        assert cfg.brightness.strength == 0.3
        assert cfg.jpeg_quality.strength == 25
        assert cfg.enable_jpeg is True


class TestTrainingConfig:
    def test_defaults(self) -> None:
        cfg = TrainingConfig()
        assert cfg.num_steps == 140000
        assert cfg.lr == 1e-4
        assert cfg.num_bits == 100
        assert cfg.image_size == 400
        assert cfg.warmup_steps == 500


class TestDataConfig:
    def test_required_fields(self) -> None:
        cfg = DataConfig(source="folder", path="/data")
        assert cfg.source == "folder"
        assert cfg.path == "/data"
        assert cfg.batch_size == 4


class TestCheckpointConfig:
    def test_defaults(self) -> None:
        cfg = CheckpointConfig()
        assert cfg.dir == "checkpoints"
        assert cfg.save_every_steps == 10000
        assert cfg.keep_last == 3


class TestLoggingConfig:
    def test_defaults(self) -> None:
        cfg = LoggingConfig()
        assert cfg.backends == ["console", "tensorboard"]
        assert cfg.log_every_steps == 100
        assert cfg.tensorboard_dir == "runs"


class TestConfig:
    def test_minimal(self) -> None:
        cfg = Config(
            experiment_name="test",
            data=DataConfig(source="folder", path="/data"),
        )
        assert cfg.experiment_name == "test"
        assert cfg.training.num_steps == 140000
        assert cfg.loss.l2.scale == 1.0


class TestLoadConfig:
    def test_load_minimal_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
experiment_name: test_exp
data:
  source: folder
  path: /data/images
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(str(config_file))
        assert cfg.experiment_name == "test_exp"
        assert cfg.data.source == "folder"
        assert cfg.data.path == "/data/images"
        assert cfg.training.num_steps == 140000  # default

    def test_load_with_overrides(self, tmp_path: Path) -> None:
        yaml_content = """
experiment_name: test_exp
data:
  source: folder
  path: /data/images
training:
  lr: 0.001
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(str(config_file), overrides={"training": {"lr": 0.0003}})
        assert cfg.training.lr == 0.0003

    def test_load_nested_loss_ramp(self, tmp_path: Path) -> None:
        yaml_content = """
experiment_name: test_exp
data:
  source: folder
  path: /data
loss:
  l2:
    scale: 1.5
    ramp_steps: 10000
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(str(config_file))
        assert cfg.loss.l2.scale == 1.5
        assert cfg.loss.l2.ramp_steps == 10000

    def test_load_ffl_and_gan_config(self, tmp_path: Path) -> None:
        yaml_content = """
experiment_name: test_gan_exp
data:
  source: folder
  path: /data
loss:
  ffl:
    scale: 0.1
    ramp_steps: 40000
    delay_steps: 20000
  gan:
    scale: 0.01
    ramp_steps: 50000
    delay_steps: 30000
  gan_config:
    enabled: true
    discriminator_lr: 0.0004
    lambda_gp: 10.0
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(str(config_file))
        assert cfg.model == "stegastamp"  # Default model
        assert cfg.loss.ffl is not None
        assert cfg.loss.ffl.scale == 0.1
        assert cfg.loss.ffl.ramp_steps == 40000
        assert cfg.loss.ffl.delay_steps == 20000
        assert cfg.loss.gan is not None
        assert cfg.loss.gan.scale == 0.01
        assert cfg.loss.gan.delay_steps == 30000
        assert cfg.loss.gan_config.enabled is True
        assert cfg.loss.gan_config.discriminator_lr == 0.0004
