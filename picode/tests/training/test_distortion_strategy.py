"""Tests for distortion strategies."""

import torch

from picode.training.distortion_strategy import (
    CurriculumDistortion,
    FixedDistortion,
    NoDistortion,
    RandomDistortion,
    create_distortion_strategy,
)
from picode.training.config import DistortionConfig, DistortionRamp


class TestNoDistortion:
    def test_identity(self) -> None:
        strategy = NoDistortion()
        image = torch.rand(2, 3, 64, 64)
        result = strategy(image, step=100)
        assert torch.equal(result, image)


class TestFixedDistortion:
    def test_applies_distortion(self) -> None:
        from picode.distortions.native import GaussianNoise

        strategy = FixedDistortion([GaussianNoise(std=0.1)])
        image = torch.rand(2, 3, 64, 64)
        result = strategy(image, step=100)
        assert result.shape == image.shape
        assert not torch.equal(result, image)  # Should be different


class TestRandomDistortion:
    def test_applies_random_subset(self) -> None:
        from picode.distortions.native import GaussianNoise, BrightnessHue, Saturation

        distortions = [
            GaussianNoise(std=0.1),
            BrightnessHue(rnd_bri=0.3, rnd_hue=0.1),
            Saturation(rnd_sat=0.5),
        ]
        strategy = RandomDistortion(distortions, num_apply=2)
        image = torch.rand(2, 3, 64, 64)
        result = strategy(image, step=100)
        assert result.shape == image.shape

    def test_num_apply_range(self) -> None:
        from picode.distortions.native import GaussianNoise, BrightnessHue, Saturation

        distortions = [
            GaussianNoise(std=0.1),
            BrightnessHue(rnd_bri=0.3, rnd_hue=0.1),
            Saturation(rnd_sat=0.5),
        ]
        strategy = RandomDistortion(distortions, num_apply=(1, 3))
        image = torch.rand(2, 3, 64, 64)
        result = strategy(image, step=100)
        assert result.shape == image.shape


class TestCurriculumDistortion:
    def test_ramps_strength(self) -> None:
        config = DistortionConfig(
            strategy="curriculum",
            noise=DistortionRamp(strength=0.1, ramp_steps=100),
        )
        strategy = CurriculumDistortion(config)

        image = torch.rand(2, 3, 64, 64)

        # At step 0, strength should be 0
        result_0 = strategy(image, step=0)
        # At step 50, strength should be 0.05
        result_50 = strategy(image, step=50)
        # At step 100+, strength should be 0.1
        result_100 = strategy(image, step=100)

        # All should have same shape
        assert result_0.shape == image.shape
        assert result_50.shape == image.shape
        assert result_100.shape == image.shape

    def test_at_step_zero_no_distortion(self) -> None:
        """At step 0, all ramps should produce 0 strength."""
        config = DistortionConfig(
            strategy="curriculum",
            noise=DistortionRamp(strength=0.1, ramp_steps=100),
            brightness=DistortionRamp(strength=0.3, ramp_steps=100),
            hue=DistortionRamp(strength=0.1, ramp_steps=100),
            saturation=DistortionRamp(strength=1.0, ramp_steps=100),
            enable_jpeg=False,  # Disable JPEG for simpler test
        )
        strategy = CurriculumDistortion(config)

        image = torch.rand(2, 3, 64, 64)
        result = strategy(image, step=0)

        # At step 0, should return unchanged image
        assert torch.equal(result, image)

    def test_with_jpeg_enabled(self) -> None:
        config = DistortionConfig(
            strategy="curriculum",
            enable_jpeg=True,
            jpeg_quality=DistortionRamp(strength=25, ramp_steps=100),
        )
        strategy = CurriculumDistortion(config)

        image = torch.rand(2, 3, 64, 64)
        result = strategy(image, step=100)
        assert result.shape == image.shape


class TestCreateDistortionStrategy:
    def test_creates_none(self) -> None:
        config = DistortionConfig(strategy="none")
        strategy = create_distortion_strategy(config)
        assert isinstance(strategy, NoDistortion)

    def test_creates_curriculum(self) -> None:
        config = DistortionConfig(strategy="curriculum")
        strategy = create_distortion_strategy(config)
        assert isinstance(strategy, CurriculumDistortion)

    def test_creates_fixed(self) -> None:
        config = DistortionConfig(
            strategy="fixed",
            noise=DistortionRamp(strength=0.1, ramp_steps=100),
        )
        strategy = create_distortion_strategy(config)
        assert isinstance(strategy, FixedDistortion)

    def test_creates_random(self) -> None:
        config = DistortionConfig(
            strategy="random",
            noise=DistortionRamp(strength=0.1, ramp_steps=100),
        )
        strategy = create_distortion_strategy(config)
        assert isinstance(strategy, RandomDistortion)

    def test_unknown_strategy_raises(self) -> None:
        config = DistortionConfig(strategy="unknown")
        try:
            create_distortion_strategy(config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "unknown" in str(e).lower()
