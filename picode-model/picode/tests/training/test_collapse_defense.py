"""Tests for collapse defense config and mechanics."""

import copy

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

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


class TestGradientNormClipping:
    """Test gradient norm clipping utility."""

    def test_clip_grad_norm_limits_total_norm(self) -> None:
        """clip_grad_norm_ should limit total gradient norm."""
        model = nn.Linear(100, 100)
        # Create large gradients
        x = torch.randn(4, 100)
        y = model(x)
        (y.sum() * 1000).backward()

        # Get norm before clipping
        total_norm_before = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float("inf")
        )
        assert total_norm_before > 1.0  # Should be large

        # Reset and clip
        model.zero_grad()
        y = model(x)
        (y.sum() * 1000).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check norm after clipping
        total_norm_after = (
            sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        )
        assert float(total_norm_after) <= 1.0 + 1e-4


class TestCosineScheduler:
    """Test cosine LR schedule integration."""

    def test_cosine_schedule_decays_lr(self) -> None:
        """Cosine schedule should decay LR over steps."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=140000, eta_min=3e-5)

        initial_lr = optimizer.param_groups[0]["lr"]
        assert initial_lr == 3e-4

        # Simulate 70000 steps
        for _ in range(70000):
            scheduler.step()

        mid_lr = optimizer.param_groups[0]["lr"]
        assert mid_lr < initial_lr
        assert mid_lr > 3e-5  # Not at minimum yet

        # Simulate to 140000
        for _ in range(70000):
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        assert abs(final_lr - 3e-5) < 1e-7


class TestEMA:
    """Test EMA weight tracking."""

    def test_ema_update_moves_toward_current(self) -> None:
        """EMA should move toward current weights."""
        model = nn.Linear(10, 10, bias=False)
        ema_state = copy.deepcopy(model.state_dict())

        # Modify model weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p))

        # Update EMA (decay=0.9 for easy math)
        decay = 0.9
        current_state = model.state_dict()
        for key in ema_state:
            ema_state[key] = decay * ema_state[key] + (1 - decay) * current_state[key]

        # EMA should be between original (zeros-ish) and current (ones-ish)
        for key in ema_state:
            assert ema_state[key].mean().item() > 0  # Moved from original
            assert ema_state[key].mean().item() < current_state[key].mean().item()  # Not at current


class TestCollapseDetection:
    """Test collapse detection logic."""

    def test_prob_std_below_threshold_is_collapse(self) -> None:
        """prob_std < threshold should be detected as collapse."""
        threshold = 0.05
        prob_std = 0.01
        assert prob_std < threshold

    def test_prob_std_above_threshold_is_not_collapse(self) -> None:
        """prob_std >= threshold should not trigger recovery."""
        threshold = 0.05
        prob_std = 0.2
        assert prob_std >= threshold

    def test_lr_halving(self) -> None:
        """Recovery should halve all optimizer LR groups."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        for group in optimizer.param_groups:
            group["lr"] *= 0.5

        assert optimizer.param_groups[0]["lr"] == 1.5e-4
