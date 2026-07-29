"""Integration test: PicoTrust v2 full training step."""

import os

# Enable MPS CPU fallback for grid_sampler_2d_backward (not supported on MPS)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pytest

from picode.training.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    DelayedLossRamp,
    DistortionConfig,
    GANConfig,
    LoggingConfig,
    LossConfig,
    LossRamp,
    ModelConfig,
    TrainingConfig,
)


@pytest.fixture
def v2_config(test_images_dir):
    """Minimal PicoTrust v2 config for testing."""
    return Config(
        experiment_name="test_v2",
        data=DataConfig(source="folder", path=str(test_images_dir), batch_size=2),
        model=ModelConfig(type="picotrust", encoder_size=64, decoder_size=64),
        training=TrainingConfig(
            num_steps=10,
            lr=1e-4,
            num_bits=100,
            image_size=64,
            warmup_steps=1,
            no_im_loss_steps=1,
            residual_strength=0.05,
            residual_strength_anneal_target=0.03,
            residual_strength_anneal_start=3,
            residual_strength_anneal_steps=2,
            phase2_step=3,
            phase2_decoder_lr_scale=0.1,
        ),
        loss=LossConfig(
            message=LossRamp(2.0, 1),
            l2=LossRamp(3.0, 2),
            lpips=LossRamp(0.0, 1),  # Skip LPIPS in test
            ssim=DelayedLossRamp(scale=1.0, ramp_steps=2, delay_steps=1),
            mask_reg=LossRamp(0.1, 1),
            message_loss_type="mse",
            gan_config=GANConfig(
                enabled=True,
                discriminator_type="patchgan",
                discriminator_lr=1e-4,
                g_loss_scale=1.0,
                g_loss_ramp_steps=2,
            ),
        ),
        distortion=DistortionConfig(strategy="none"),
        checkpoint=CheckpointConfig(dir="/tmp/test_v2_ckpt", save_every_steps=999),
        logging=LoggingConfig(backends=["console"], log_every_steps=1),
    )


def test_v2_training_step_runs(v2_config):
    """Full training step should complete without error."""
    from picode.training.trainer import Trainer

    trainer = Trainer(v2_config)
    assert trainer.encoder.strength is not None
    assert trainer.encoder.mask_head is not None
    assert trainer.discriminator is not None
    # Run a few steps (step 0 may have loss=0 due to ramp warmup)
    any_positive = False
    for _ in range(3):
        images = trainer._get_batch()
        metrics = trainer._train_step(images)
        trainer.global_step += 1
        assert "loss" in metrics
        assert "loss_msg" in metrics
        assert metrics["loss"] >= 0
        if metrics["loss"] > 0:
            any_positive = True
    assert any_positive, "Loss should be positive for at least one step after warmup"


def test_v2_strength_anneals(v2_config):
    """Strength should anneal during training."""
    from picode.training.trainer import Trainer

    trainer = Trainer(v2_config)
    assert trainer.encoder.strength == 0.05
    for _ in range(6):
        images = trainer._get_batch()
        trainer._train_step(images)
        trainer.global_step += 1
    assert trainer.encoder.strength < 0.05


def test_v2_phase2_lr_reduces(v2_config):
    """Decoder LR should drop at phase2_step."""
    from picode.training.trainer import Trainer

    trainer = Trainer(v2_config)
    initial_decoder_lr = trainer.optimizer.param_groups[1]["lr"]
    for _ in range(6):
        images = trainer._get_batch()
        trainer._train_step(images)
        trainer.global_step += 1
    final_decoder_lr = trainer.optimizer.param_groups[1]["lr"]
    assert final_decoder_lr < initial_decoder_lr
