"""Training configuration dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LossRamp:
    """A loss weight that ramps up over training steps."""

    scale: float
    ramp_steps: int = 0


@dataclass
class DistortionRamp:
    """A distortion strength that ramps up over training steps."""

    strength: float
    ramp_steps: int = 1000


@dataclass
class LossConfig:
    """Loss function configuration with ramping."""

    message: LossRamp = field(default_factory=lambda: LossRamp(1.0, 1))
    l2: LossRamp = field(default_factory=lambda: LossRamp(2.0, 20000))
    lpips: LossRamp = field(default_factory=lambda: LossRamp(1.0, 20000))

    l2_edge_gain: float = 10.0
    l2_edge_ramp_steps: int = 20000
    l2_edge_delay_steps: int = 60000

    yuv_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    gan: LossRamp | None = None


@dataclass
class DistortionConfig:
    """Distortion configuration for training."""

    strategy: str = "curriculum"

    perspective: DistortionRamp = field(default_factory=lambda: DistortionRamp(0.1, 10000))
    brightness: DistortionRamp = field(default_factory=lambda: DistortionRamp(0.3, 1000))
    saturation: DistortionRamp = field(default_factory=lambda: DistortionRamp(1.0, 1000))
    hue: DistortionRamp = field(default_factory=lambda: DistortionRamp(0.1, 1000))
    noise: DistortionRamp = field(default_factory=lambda: DistortionRamp(0.02, 1000))
    contrast: tuple[float, float] = (0.5, 1.5)
    contrast_ramp_steps: int = 1000
    jpeg_quality: DistortionRamp = field(default_factory=lambda: DistortionRamp(25, 1000))

    enable_jpeg: bool = True
    enable_blur: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    num_steps: int = 140000
    lr: float = 1e-4
    num_bits: int = 100
    image_size: int = 400
    warmup_steps: int = 500


@dataclass
class DataConfig:
    """Data loading configuration."""

    source: str
    path: str
    batch_size: int = 4
    num_workers: int = 4


@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""

    dir: str = "checkpoints"
    save_every_steps: int = 10000
    keep_last: int = 3


@dataclass
class LoggingConfig:
    """Logging configuration."""

    backends: list[str] = field(default_factory=lambda: ["console", "tensorboard"])
    log_every_steps: int = 100
    tensorboard_dir: str = "runs"


@dataclass
class Config:
    """Complete training configuration."""

    experiment_name: str
    data: DataConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    distortion: DistortionConfig = field(default_factory=DistortionConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(data: dict[str, Any]) -> Config:
    """Convert nested dict to Config, handling nested dataclasses."""
    # Handle DataConfig (required)
    data["data"] = DataConfig(**data["data"])

    # Handle TrainingConfig
    if "training" in data:
        data["training"] = TrainingConfig(**data["training"])

    # Handle LossConfig with nested LossRamps
    if "loss" in data:
        loss_data = data["loss"]
        for key in ["message", "l2", "lpips"]:
            if key in loss_data and isinstance(loss_data[key], dict):
                loss_data[key] = LossRamp(**loss_data[key])
        if "gan" in loss_data and loss_data["gan"] is not None:
            loss_data["gan"] = LossRamp(**loss_data["gan"])
        if "yuv_weights" in loss_data:
            loss_data["yuv_weights"] = tuple(loss_data["yuv_weights"])
        data["loss"] = LossConfig(**loss_data)

    # Handle DistortionConfig with nested DistortionRamps
    if "distortion" in data:
        dist_data = data["distortion"]
        for key in ["perspective", "brightness", "saturation", "hue", "noise", "jpeg_quality"]:
            if key in dist_data and isinstance(dist_data[key], dict):
                dist_data[key] = DistortionRamp(**dist_data[key])
        if "contrast" in dist_data:
            dist_data["contrast"] = tuple(dist_data["contrast"])
        data["distortion"] = DistortionConfig(**dist_data)

    # Handle CheckpointConfig
    if "checkpoint" in data:
        data["checkpoint"] = CheckpointConfig(**data["checkpoint"])

    # Handle LoggingConfig
    if "logging" in data:
        data["logging"] = LoggingConfig(**data["logging"])

    return Config(**data)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load config from YAML file with optional overrides.

    Args:
        path: Path to YAML config file.
        overrides: Optional dict of overrides to apply.

    Returns:
        Parsed Config object.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    if overrides:
        data = _deep_merge(data, overrides)

    return _dict_to_config(data)
