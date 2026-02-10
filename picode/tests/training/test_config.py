"""Tests for training configuration."""

from picode.training.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    DistortionConfig,
    DistortionRamp,
    LoggingConfig,
    LossConfig,
    LossRamp,
    TrainingConfig,
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
        assert cfg.message.scale == 1.0
        assert cfg.l2.scale == 2.0
        assert cfg.l2.ramp_steps == 20000
        assert cfg.lpips.scale == 1.0
        assert cfg.l2_edge_gain == 10.0
        assert cfg.l2_edge_delay_steps == 60000
        assert cfg.yuv_weights == (1.0, 1.0, 1.0)
        assert cfg.gan is None


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
        assert cfg.loss.l2.scale == 2.0
