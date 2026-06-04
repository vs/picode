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
    Defaults match original StegaStamp TensorFlow implementation.
    """

    enabled: bool = True  # Enable by default to match original
    discriminator_lr: float = 1e-5  # Original: 0.00001
    lambda_gp: float = 10.0  # Gradient penalty weight for WGAN-GP
    n_critic: int = 1  # Discriminator updates per generator update
    clip_weights: float = 0.01  # Original: 0.01
    g_loss_scale: float = 1.0  # Original: 1.0 (not 0.001)
    g_loss_ramp_steps: int = 20000  # Steps to ramp up generator loss
    gradient_clip: float = 0.25  # Original: clips D gradients to [-0.25, 0.25]
    discriminator_type: str = "wgan"  # "wgan" or "patchgan"


@dataclass
class LossConfig:
    """Loss function configuration with ramping.

    Defaults match original StegaStamp TensorFlow implementation.
    """

    message: LossRamp = field(default_factory=lambda: LossRamp(1.0, 1))  # Original: 1.0
    l2: LossRamp = field(default_factory=lambda: LossRamp(1.5, 20000))  # Original: 1.5
    lpips: LossRamp = field(default_factory=lambda: LossRamp(1.0, 20000))  # Original: 1.0

    # Message loss type: "bce" (binary cross entropy) or "mse" (mean squared error)
    # MSE avoids the trivial solution where decoder outputs 0.5 for all bits
    message_loss_type: str = "bce"

    # Border falloff mask (original StegaStamp approach)
    # Amplifies loss at image borders using cosine falloff
    use_border_falloff: bool = True  # Enable by default to match original
    border_falloff_speed: int = 4  # Original: falloff_speed=4 (25% border region)

    l2_edge_gain: float = 10.0  # Gain for border falloff amplification
    l2_edge_ramp_steps: int = 20000
    l2_edge_delay_steps: int = 60000

    yuv_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Chrominance preservation loss — penalizes cross-channel residual variance.
    # Keeps colour shifts imperceptible while allowing full 3ch residual capacity.
    # Delayed to avoid interfering with encoder-decoder bootstrap.
    chroma: DelayedLossRamp | None = None

    # Focal Frequency Loss (FFL) - reduces frequency-domain artifacts
    # delay 20k, ramp 20k→60k
    ffl: DelayedLossRamp | None = None

    # GAN loss weight (for generator adversarial loss)
    # delay 30k, ramp 30k→80k
    gan: DelayedLossRamp | None = None

    # SSIM loss — structural similarity for image quality
    ssim: DelayedLossRamp | None = None

    # Mask regularization — encourages learned mask to correlate with texture
    mask_reg: LossRamp | None = None

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
    """Training hyperparameters.

    Defaults match original StegaStamp TensorFlow implementation.
    """

    num_steps: int = 140000
    lr: float = 1e-4
    num_bits: int = 100
    image_size: int = 400
    warmup_steps: int = 500
    residual_scale: float = 0.1  # picode_v2 encoder residual magnitude
    encoder_lr_scale: float = 1.0  # Multiplier for encoder learning rate (decoder uses base lr)
    stn_lr_scale: float = 0.01  # Multiplier for STN linear params (stn_fc_weight, stn_fc_bias)
    no_im_loss_steps: int = 500  # Original: 500 steps message-only training
    decoder_warmup_steps: int = 0  # Steps with message-loss-only (no frame L2)
    generator_grad_clip: float = 0.25  # Original: clips to [-0.25, 0.25]
    seed: int | None = None  # Random seed for reproducibility (None = non-deterministic)

    # Collapse defense: gradient norm clipping
    # Replaces per-element clamping with clip_grad_norm_ for both encoder and decoder.
    # Set to 0 to disable (falls back to generator_grad_clip per-element clamping).
    grad_clip_norm: float = 1.0

    # Learning rate schedule: "constant" or "cosine"
    lr_schedule: str = "constant"
    lr_min_ratio: float = 0.1  # eta_min = lr * lr_min_ratio (for cosine)

    # EMA + collapse detection
    ema_decay: float = 0.999
    collapse_threshold: float = 0.05  # prob_std below this triggers recovery
    collapse_recovery_cooldown: int = 500  # Steps to wait after recovery before checking again

    # Pre-encode warp / post-encode unwarp (StegaStamp training strategy)
    # This creates a canonical encoding space that is robust to geometric transforms
    borders: str = "black"  # Border mode: no_edge, black, random, randomrgb, white, image
    rnd_trans: float = 0.1  # Max perspective translation (fraction of image size)
    rnd_trans_ramp: int = 10000  # Steps to ramp up perspective strength from 0 to rnd_trans

    # Residual amplitude control (PicoTrust v2)
    residual_strength: float = 0.0  # 0 = disabled (v1 compat). Max residual = strength * tanh
    residual_strength_anneal_target: float = 0.03  # Anneal to this in phase 2
    residual_strength_anneal_start: int = 60000  # Step to start annealing
    residual_strength_anneal_steps: int = 20000  # Steps to anneal over

    # Two-phase training
    phase2_step: int = 0  # 0 means disabled. Step to enter phase 2
    phase2_decoder_lr_scale: float = 0.1  # Multiply decoder LR by this in phase 2


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
class FrameConfig:
    """PicodeFrame-specific configuration.

    Controls frame generation parameters for the PicodeFrame model.
    """

    min_frame_pct: float = 0.02  # Min frame width as fraction of image
    max_frame_pct: float = 0.05  # Max frame width as fraction of image
    fixed_frame_steps: int = 0  # Use max_frame_pct for first N steps, then randomize
    frame_l2_scale: float = 2.0
    frame_l2_ramp_steps: int = 1
    frame_lpips_scale: float = 1.5
    frame_lpips_ramp_steps: int = 10000
    frame_color_scale: float = 2.0  # Penalize color shifts in frame residual
    frame_color_ramp_steps: int = 1  # Active from step 0
    stn_reg_scale: float = 0.1
    residual_max_amplitude: float = 0.0  # 0 = disabled. Max residual = tanh * this value


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Attributes:
        type: Model type: "stegastamp", "picodelite", "picodeframe", or "picotrust".
        encoder_size: Image size for the encoder (e.g., 400 for StegaStamp, 256 for PicoTrust).
        decoder_size: Image size for the decoder (e.g., 400 for StegaStamp, 256 for PicoTrust).
    """

    type: str = "stegastamp"  # stegastamp, picodelite, picodeframe, or picotrust
    encoder_size: int = 400
    decoder_size: int = 400

    def __post_init__(self) -> None:
        """Apply model-specific size defaults when sizes are left at default."""
        if self.type == "picotrust" and self.encoder_size == 400 and self.decoder_size == 400:
            self.encoder_size = 256
            self.decoder_size = 256


@dataclass
class Config:
    """Complete training configuration."""

    experiment_name: str
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    distortion: DistortionConfig = field(default_factory=DistortionConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    frame: FrameConfig | None = None


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

    # Handle ModelConfig (BEFORE TrainingConfig)
    if "model" in data:
        if isinstance(data["model"], str):
            # Backward compat: "model: stegastamp" -> ModelConfig(type="stegastamp")
            data["model"] = ModelConfig(type=data["model"])
        elif isinstance(data["model"], dict):
            data["model"] = ModelConfig(**data["model"])
    else:
        data["model"] = ModelConfig()

    # Handle TrainingConfig
    if "training" in data:
        data["training"] = TrainingConfig(**data["training"])

    # Handle LossConfig with nested LossRamps and DelayedLossRamps
    if "loss" in data:
        loss_data = data["loss"]
        # Standard LossRamp fields
        for key in ["message", "l2", "lpips", "mask_reg"]:
            if key in loss_data and isinstance(loss_data[key], dict):
                loss_data[key] = LossRamp(**loss_data[key])
        # DelayedLossRamp fields (ffl, gan, ssim, chroma)
        for key in ["ffl", "gan", "ssim", "chroma"]:
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

    # Handle FrameConfig (PicodeFrame-specific)
    if "frame" in data:
        if data["frame"] is not None and isinstance(data["frame"], dict):
            data["frame"] = FrameConfig(**data["frame"])

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
