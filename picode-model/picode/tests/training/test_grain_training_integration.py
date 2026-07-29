"""Integration test: PicoGrain grain-aware losses wired into the trainer."""

import os

# Enable MPS CPU fallback for grid_sampler_2d_backward (not supported on MPS)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pytest
import torch

from picode.training.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    DistortionConfig,
    GANConfig,
    GrainLossConfig,
    LoggingConfig,
    LossConfig,
    LossRamp,
    ModelConfig,
    TrainingConfig,
    load_config,
)


@pytest.fixture
def grain_config(test_images_dir):
    """Minimal PicoGrain config for testing."""
    return Config(
        experiment_name="test_grain",
        data=DataConfig(source="folder", path=str(test_images_dir), batch_size=2),
        model=ModelConfig(type="picograin", encoder_size=64, decoder_size=64),
        training=TrainingConfig(
            num_steps=10,
            lr=1e-4,
            num_bits=127,
            image_size=64,
            warmup_steps=1,
            no_im_loss_steps=1,
            residual_strength=0.10,
        ),
        loss=LossConfig(
            message=LossRamp(5.0, 1),
            l2=LossRamp(1.5, 2),
            lpips=LossRamp(0.0, 1),  # Skip LPIPS in test (no network download)
            message_loss_type="mse",
            grain=GrainLossConfig(
                structure_blur_sigma=2.0,
                lum_fidelity=LossRamp(1.0, 2),
                envelope_tv=LossRamp(0.5, 2),
                lum_floor=0.1,
            ),
            gan_config=GANConfig(enabled=False),
        ),
        distortion=DistortionConfig(strategy="none"),
        checkpoint=CheckpointConfig(dir="/tmp/test_grain_ckpt", save_every_steps=999),
        logging=LoggingConfig(backends=["console"], log_every_steps=1),
    )


def test_grain_loss_config_parses_from_yaml(tmp_path):
    """A `loss.grain` YAML block should become a GrainLossConfig."""
    cfg_path = tmp_path / "grain.yaml"
    cfg_path.write_text(
        "experiment_name: g\n"
        "data:\n"
        "  source: folder\n"
        "  path: ./data\n"
        "model:\n"
        "  type: picograin\n"
        "loss:\n"
        "  grain:\n"
        "    structure_blur_sigma: 2.5\n"
        "    lum_floor: 0.15\n"
        "    lum_fidelity:\n"
        "      scale: 1.0\n"
        "      ramp_steps: 100\n"
        "    envelope_tv:\n"
        "      scale: 0.5\n"
        "      ramp_steps: 200\n"
    )
    cfg = load_config(cfg_path)

    assert cfg.loss.grain is not None
    assert cfg.loss.grain.structure_blur_sigma == 2.5
    assert cfg.loss.grain.lum_floor == 0.15
    assert cfg.loss.grain.lum_fidelity.scale == 1.0
    assert cfg.loss.grain.lum_fidelity.ramp_steps == 100
    assert cfg.loss.grain.envelope_tv.scale == 0.5
    assert cfg.loss.grain.envelope_tv.ramp_steps == 200


def test_grain_loss_config_defaults_to_none():
    """Configs without a grain block leave it unset (PicoTrust unaffected)."""
    assert LossConfig().grain is None


def test_grain_l2_ignores_grain_but_catches_structure(grain_config):
    """Blurred L2 should barely react to grain but still react to a real shift."""
    from picode.training.trainer import Trainer

    trainer = Trainer(grain_config)
    trainer.global_step = 100  # past no_im_loss_steps so image losses are live

    torch.manual_seed(0)
    original = torch.rand(2, 3, 64, 64, device=trainer.device)
    messages = torch.randint(0, 2, (2, 127), device=trainer.device).float()
    logits = torch.zeros(2, 127, device=trainer.device)

    # Case A: heavy grain — the aesthetic we intentionally want
    grain = original + 0.10 * torch.randn_like(original)
    # Case B: same magnitude, but a structural brightness shift
    shift = original + 0.10

    loss_grain = trainer._compute_ramped_losses(original, grain, messages, logits)["loss_l2"]
    loss_shift = trainer._compute_ramped_losses(original, shift, messages, logits)["loss_l2"]

    # Grain is blurred away; the structural shift is not.
    assert loss_grain < loss_shift * 0.5


def test_grain_losses_present_in_output(grain_config):
    """Luminance fidelity and envelope TV should appear when an envelope is passed."""
    from picode.training.trainer import Trainer

    trainer = Trainer(grain_config)
    trainer.global_step = 100

    original = torch.rand(2, 3, 64, 64, device=trainer.device)
    encoded = original + 0.05 * torch.randn_like(original)
    envelope = torch.rand(2, 1, 64, 64, device=trainer.device)
    messages = torch.randint(0, 2, (2, 127), device=trainer.device).float()
    logits = torch.zeros(2, 127, device=trainer.device)

    losses = trainer._compute_ramped_losses(
        original, encoded, messages, logits, envelope=envelope,
    )

    assert "loss_lum_fidelity" in losses
    assert "loss_envelope_tv" in losses
    assert losses["loss_envelope_tv"] > 0  # random envelope is not smooth


def test_grain_training_step_runs(grain_config):
    """A full PicoGrain training step should run and report grain losses."""
    from picode.training.trainer import Trainer

    trainer = Trainer(grain_config)
    # _train_step does not advance global_step (fit() owns that), so set it
    # past no_im_loss_steps and the loss ramps explicitly.
    trainer.global_step = 100

    metrics = None
    for _ in range(3):
        images = trainer._get_batch()
        metrics = trainer._train_step(images)

    assert metrics is not None
    assert "loss_lum_fidelity" in metrics
    assert "loss_envelope_tv" in metrics
    assert metrics["loss"] > 0
