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
class DelayedLossRamp:
    """A loss weight that starts after a delay and ramps up.

    Timeline: [0, delay_steps) -> 0, [delay_steps, delay_steps + ramp_steps) -> ramps to scale
    """

    scale: float
    ramp_steps: int = 40000
    delay_steps: int = 20000


@dataclass
class DistortionRamp:
    """A distortion strength that ramps up over training steps."""

    strength: float
    ramp_steps: int = 1000


@dataclass
class GANConfig:
    """GAN training configuration.

    Controls discriminator training and generator adversarial loss.
    """

    enabled: bool = False
    discriminator_lr: float = 4e-4  # Typically higher than generator LR
    lambda_gp: float = 10.0  # Gradient penalty weight for WGAN-GP
    n_critic: int = 1  # Discriminator updates per generator update
    clip_weights: float = 0.01  # WGAN weight clipping bound


@dataclass
class LossConfig:
    """Loss function configuration with ramping.

    StegaStamp uses ~7x higher message loss than image loss to ensure
    the encoder prioritizes message encoding over image preservation.
    Without this, the encoder learns to output the original image unchanged.
    """

    message: LossRamp = field(default_factory=lambda: LossRamp(7.0, 1))
    l2: LossRamp = field(default_factory=lambda: LossRamp(1.0, 20000))
    lpips: LossRamp = field(default_factory=lambda: LossRamp(1.5, 20000))

    # Message loss type: "bce" (binary cross entropy) or "mse" (mean squared error)
    # MSE avoids the trivial solution where decoder outputs 0.5 for all bits
    message_loss_type: str = "bce"

    l2_edge_gain: float = 10.0
    l2_edge_ramp_steps: int = 20000
    l2_edge_delay_steps: int = 60000

    yuv_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Focal Frequency Loss (FFL) - reduces frequency-domain artifacts
    # delay 20k, ramp 20k→60k
    ffl: DelayedLossRamp | None = None

    # GAN loss weight (for generator adversarial loss)
    # delay 30k, ramp 30k→80k
    gan: DelayedLossRamp | None = None

    # GAN training settings
    gan_config: GANConfig = field(default_factory=GANConfig)


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
    residual_scale: float = 0.1  # picode_v2 encoder residual magnitude
    encoder_lr_scale: float = 1.0  # Multiplier for encoder learning rate (decoder uses base lr)
    no_im_loss_steps: int = 0  # Steps to train message loss only (no L2/LPIPS), like StegaStamp


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
    model: str = "stegastamp"  # Only stegastamp is supported
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

    # Handle LossConfig with nested LossRamps and DelayedLossRamps
    if "loss" in data:
        loss_data = data["loss"]
        # Standard LossRamp fields
        for key in ["message", "l2", "lpips"]:
            if key in loss_data and isinstance(loss_data[key], dict):
                loss_data[key] = LossRamp(**loss_data[key])
        # DelayedLossRamp fields (ffl, gan)
        for key in ["ffl", "gan"]:
            if key in loss_data and loss_data[key] is not None:
                if isinstance(loss_data[key], dict):
                    loss_data[key] = DelayedLossRamp(**loss_data[key])
        # GANConfig
        if "gan_config" in loss_data and isinstance(loss_data["gan_config"], dict):
            loss_data["gan_config"] = GANConfig(**loss_data["gan_config"])
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
