"""Tests for PicoTrust v2 configuration fields."""

from picode.training.config import (
    GANConfig,
    LossConfig,
    LossRamp,
    TrainingConfig,
    load_config,
)


def test_loss_config_has_ssim_field():
    cfg = LossConfig()
    assert cfg.ssim is None


def test_loss_config_has_mask_reg_field():
    cfg = LossConfig()
    assert cfg.mask_reg is None


def test_training_config_has_strength_fields():
    cfg = TrainingConfig()
    assert cfg.residual_strength == 0.0
    assert cfg.residual_strength_anneal_target == 0.03
    assert cfg.residual_strength_anneal_start == 60000
    assert cfg.residual_strength_anneal_steps == 20000


def test_training_config_has_phase2_fields():
    cfg = TrainingConfig()
    assert cfg.phase2_step == 0
    assert cfg.phase2_decoder_lr_scale == 0.1


def test_gan_config_has_discriminator_type():
    cfg = GANConfig()
    assert cfg.discriminator_type == "wgan"


def test_load_picotrust_v2_config(tmp_path):
    yaml_content = tmp_path / "test.yaml"
    yaml_content.write_text(
        "experiment_name: test_v2\n"
        "data:\n"
        "  source: folder\n"
        "  path: ./data/train\n"
        "model:\n"
        "  type: picotrust\n"
        "  encoder_size: 512\n"
        "  decoder_size: 512\n"
        "training:\n"
        "  residual_strength: 0.05\n"
        "  residual_strength_anneal_target: 0.03\n"
        "  residual_strength_anneal_start: 60000\n"
        "  residual_strength_anneal_steps: 20000\n"
        "  phase2_step: 60000\n"
        "  phase2_decoder_lr_scale: 0.1\n"
        "loss:\n"
        "  ssim:\n"
        "    scale: 1.0\n"
        "    ramp_steps: 20000\n"
        "    delay_steps: 30000\n"
        "  mask_reg:\n"
        "    scale: 0.1\n"
        "    ramp_steps: 1\n"
        "  gan_config:\n"
        "    discriminator_type: patchgan\n"
    )
    cfg = load_config(str(yaml_content))
    assert cfg.training.residual_strength == 0.05
    assert cfg.training.phase2_step == 60000
    assert cfg.loss.ssim is not None
    assert cfg.loss.ssim.scale == 1.0
    assert cfg.loss.mask_reg is not None
    assert cfg.loss.mask_reg.scale == 0.1
    assert cfg.loss.gan_config.discriminator_type == "patchgan"
