"""Tests for collapse defense config and mechanics."""

import torch

from picode.training.config import TrainingConfig


class TestCollapseDefenseConfig:
    """Test new TrainingConfig fields have correct defaults."""

    def test_grad_clip_norm_default(self) -> None:
        cfg = TrainingConfig()
        assert cfg.grad_clip_norm == 1.0

    def test_lr_schedule_default(self) -> None:
        cfg = TrainingConfig()
        assert cfg.lr_schedule == "constant"

    def test_lr_min_ratio_default(self) -> None:
        cfg = TrainingConfig()
        assert cfg.lr_min_ratio == 0.1

    def test_ema_decay_default(self) -> None:
        cfg = TrainingConfig()
        assert cfg.ema_decay == 0.999

    def test_collapse_threshold_default(self) -> None:
        cfg = TrainingConfig()
        assert cfg.collapse_threshold == 0.05

    def test_collapse_recovery_cooldown_default(self) -> None:
        cfg = TrainingConfig()
        assert cfg.collapse_recovery_cooldown == 500
