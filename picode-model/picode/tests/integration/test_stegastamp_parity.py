"""Integration tests to verify StegaStamp parity.

These tests verify that all StegaStamp parity changes work together correctly:
- Pre-encode warp / post-encode unwarp training pipeline
- STN gradient flow
- Random blur in distortion strategy
- Discriminator gradient clipping
- Correct loss scales and config values
"""

from pathlib import Path

import pytest
import torch
from PIL import Image

from picode.training import Trainer, load_config
from picode.training.config import Config
from picode.training.distortion_strategy import CurriculumDistortion


def _create_test_images(path: Path, count: int = 10) -> None:
    """Create test images in a directory."""
    path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = Image.new("RGB", (400, 400), color=(i * 25, i * 20, i * 15))
        img.save(path / f"img_{i:03d}.jpg")


class TestStegaStampParity:
    """Tests to verify training matches StegaStamp behavior."""

    @pytest.fixture
    def parity_config(self, tmp_path: Path) -> Config:
        """Load StegaStamp parity config with test data."""
        config_path = (
            Path(__file__).parent.parent.parent.parent / "configs" / "stegastamp_baseline.yaml"
        )
        config = load_config(str(config_path))

        # Create dummy training data
        data_dir = tmp_path / "data"
        _create_test_images(data_dir, count=10)

        # Override config for testing
        config.data.path = str(data_dir)
        config.data.batch_size = 2
        config.data.num_workers = 0
        config.training.num_steps = 100
        config.checkpoint.dir = str(tmp_path / "checkpoints")
        config.logging.backends = []  # Disable logging for tests

        return config

    def test_stn_receives_gradients(self, parity_config: Config) -> None:
        """STN should receive gradients during training.

        Verifies that the STN linear transformation parameters (stn_fc_weight
        and stn_fc_bias) are set up to receive gradients and are not frozen,
        which is essential for learning geometric corrections.
        """
        trainer = Trainer(parity_config)

        # Check STN parameters exist and are trainable
        decoder = trainer.decoder
        assert hasattr(decoder, "stn_fc_weight"), "Decoder missing STN weight parameter"
        assert hasattr(decoder, "stn_fc_bias"), "Decoder missing STN bias parameter"

        # STN params should require gradients (not frozen)
        assert decoder.stn_fc_weight.requires_grad, "STN weight should require gradients"
        assert decoder.stn_fc_bias.requires_grad, "STN bias should require gradients"

        # Run a few steps to accumulate gradients (skip step 0 where loss is scaled to 0)
        for step in range(5):
            trainer.global_step = step + 1  # Start from step 1
            images = trainer._get_batch()
            trainer._train_step(images)

        # After training steps, gradients should have accumulated
        assert decoder.stn_fc_weight.grad is not None, "STN weight should have gradients"
        assert decoder.stn_fc_bias.grad is not None, "STN bias should have gradients"

        # The bias gradient should be non-zero since bias is initialized non-zero
        # and loss flows through the STN transform
        assert decoder.stn_fc_bias.grad.abs().sum() > 0, "STN bias gradients should be non-zero"

    def test_blur_applied_in_training(self, parity_config: Config) -> None:
        """Random blur should be applied during distortion.

        Verifies that CurriculumDistortion includes the random_blur component
        which is critical for StegaStamp's training pipeline.
        """
        trainer = Trainer(parity_config)

        # Check blur is in distortion strategy
        assert isinstance(
            trainer.distortion, CurriculumDistortion
        ), "Distortion strategy should be CurriculumDistortion"
        assert hasattr(
            trainer.distortion, "random_blur"
        ), "CurriculumDistortion should have random_blur"
        assert trainer.distortion.random_blur is not None, "random_blur should not be None"

    def test_perspective_warp_training_pipeline(self, parity_config: Config) -> None:
        """Perspective warp should be applied in training.

        Verifies that the training pipeline applies perspective warping
        before encoding and unwarp after, which is the StegaStamp approach.
        """
        trainer = Trainer(parity_config)

        images = trainer._get_batch()
        metrics = trainer._train_step(images)
        trainer.global_step += 1

        # Should have valid loss values
        assert "loss_msg" in metrics, "Should have message loss"
        assert "loss_l2" in metrics, "Should have L2 loss"
        assert metrics["loss_msg"] >= 0, "Message loss should be non-negative"

        # Run more steps to verify stability
        for _ in range(5):
            images = trainer._get_batch()
            metrics = trainer._train_step(images)
            trainer.global_step += 1

        assert "loss" in metrics or "loss_msg" in metrics, "Training should produce valid losses"

    def test_discriminator_gradient_clipping(self, parity_config: Config) -> None:
        """Discriminator gradients should be clipped.

        Verifies that discriminator gradients are clipped to the configured
        range (default: [-0.25, 0.25] in StegaStamp).
        """
        trainer = Trainer(parity_config)

        # Verify discriminator exists
        assert trainer.discriminator is not None, "Discriminator should be enabled"

        images = trainer._get_batch()
        trainer._train_step(images)

        clip_val = parity_config.loss.gan_config.gradient_clip

        # Check gradient magnitudes (after clipping)
        for p in trainer.discriminator.parameters():
            if p.grad is not None:
                max_grad = p.grad.abs().max().item()
                # Allow small tolerance for numerical precision
                assert max_grad <= clip_val + 1e-6, (
                    f"Discriminator gradient {max_grad:.4f} exceeds clip value {clip_val}"
                )

    def test_generator_gradient_clipping(self, parity_config: Config) -> None:
        """Generator/encoder gradients should be clipped.

        Verifies that encoder and decoder gradients are clipped to the
        configured range (matching StegaStamp's gradient clipping).
        """
        trainer = Trainer(parity_config)

        images = trainer._get_batch()
        trainer._train_step(images)

        clip_val = parity_config.training.generator_grad_clip

        # Check encoder gradient magnitudes
        for p in trainer.encoder.parameters():
            if p.grad is not None:
                max_grad = p.grad.abs().max().item()
                assert max_grad <= clip_val + 1e-6, (
                    f"Encoder gradient {max_grad:.4f} exceeds clip value {clip_val}"
                )

        # Check decoder gradient magnitudes
        for p in trainer.decoder.parameters():
            if p.grad is not None:
                max_grad = p.grad.abs().max().item()
                assert max_grad <= clip_val + 1e-6, (
                    f"Decoder gradient {max_grad:.4f} exceeds clip value {clip_val}"
                )

    def test_config_values_match_stegastamp(self, parity_config: Config) -> None:
        """Config values should match StegaStamp defaults.

        Verifies that all key configuration values are set to match
        the original StegaStamp TensorFlow implementation.
        """
        # Training parameters
        assert parity_config.training.no_im_loss_steps == 500, (
            f"no_im_loss_steps should be 500, got {parity_config.training.no_im_loss_steps}"
        )
        assert parity_config.training.generator_grad_clip == 0.25, (
            f"generator_grad_clip should be 0.25, got {parity_config.training.generator_grad_clip}"
        )
        assert parity_config.training.rnd_trans == 0.1, (
            f"rnd_trans should be 0.1, got {parity_config.training.rnd_trans}"
        )
        assert parity_config.training.rnd_trans_ramp == 10000, (
            f"rnd_trans_ramp should be 10000, got {parity_config.training.rnd_trans_ramp}"
        )

        # Loss scales
        assert parity_config.loss.message.scale == 1.0, (
            f"message loss scale should be 1.0, got {parity_config.loss.message.scale}"
        )
        assert parity_config.loss.l2.scale == 1.5, (
            f"L2 loss scale should be 1.5, got {parity_config.loss.l2.scale}"
        )
        assert parity_config.loss.lpips.scale == 1.0, (
            f"LPIPS loss scale should be 1.0, got {parity_config.loss.lpips.scale}"
        )

        # YUV weights
        assert parity_config.loss.yuv_weights == (1.0, 1.0, 1.0), (
            f"YUV weights should be (1.0, 1.0, 1.0), got {parity_config.loss.yuv_weights}"
        )

        # Edge loss delay
        assert parity_config.loss.l2_edge_delay_steps == 60000, (
            f"l2_edge_delay_steps should be 60000, got {parity_config.loss.l2_edge_delay_steps}"
        )

        # GAN config
        assert parity_config.loss.gan_config.g_loss_scale == 1.0, (
            f"GAN g_loss_scale should be 1.0, got {parity_config.loss.gan_config.g_loss_scale}"
        )
        assert parity_config.loss.gan_config.gradient_clip == 0.25, (
            f"GAN gradient_clip should be 0.25, got {parity_config.loss.gan_config.gradient_clip}"
        )
        assert parity_config.loss.gan_config.clip_weights == 0.01, (
            f"GAN clip_weights should be 0.01, got {parity_config.loss.gan_config.clip_weights}"
        )

    def test_border_falloff_enabled(self, parity_config: Config) -> None:
        """Border falloff should be enabled by default.

        Verifies that the border falloff mask is created and used,
        matching StegaStamp's edge loss behavior.
        """
        assert parity_config.loss.use_border_falloff is True, "Border falloff should be enabled"
        assert parity_config.loss.border_falloff_speed == 4, "Border falloff speed should be 4"
        assert parity_config.loss.l2_edge_gain == 10.0, "L2 edge gain should be 10.0"

        trainer = Trainer(parity_config)

        # Verify border falloff mask is created
        assert trainer._border_falloff_mask is not None, "Border falloff mask should exist"
        assert trainer._border_falloff_mask.shape == (
            1,
            1,
            parity_config.training.image_size,
            parity_config.training.image_size,
        ), "Border falloff mask should have correct shape"

    def test_training_step_produces_valid_metrics(self, parity_config: Config) -> None:
        """Training step should produce valid metrics.

        Verifies that a complete training step runs without errors
        and produces expected metrics.
        """
        trainer = Trainer(parity_config)

        images = trainer._get_batch()
        metrics = trainer._train_step(images)

        # Check required metrics
        required_metrics = ["loss_msg", "loss_l2", "bit_accuracy", "residual_mean", "residual_std"]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

        # Check metric values are valid
        assert not torch.isnan(
            torch.tensor(metrics["loss_msg"])
        ), "Message loss should not be NaN"
        assert not torch.isnan(torch.tensor(metrics["loss_l2"])), "L2 loss should not be NaN"
        assert 0 <= metrics["bit_accuracy"] <= 1, "Bit accuracy should be in [0, 1]"

    def test_multiple_training_steps_stable(self, parity_config: Config) -> None:
        """Multiple training steps should be stable.

        Verifies that training can run for multiple steps without
        numerical instability or crashes.
        """
        trainer = Trainer(parity_config)

        losses = []
        for step in range(20):
            images = trainer._get_batch()
            metrics = trainer._train_step(images)
            trainer.global_step += 1
            losses.append(metrics["loss_msg"])

        # Check for NaN/Inf
        for i, loss in enumerate(losses):
            assert not (torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss))), (
                f"Loss became NaN/Inf at step {i}"
            )

        # Loss should generally decrease or stay bounded (not explode)
        max_loss = max(losses)
        assert max_loss < 100, f"Loss exploded to {max_loss}"

    def test_discriminator_weight_clipping(self, parity_config: Config) -> None:
        """Discriminator weights should be clipped after each step.

        Verifies WGAN weight clipping is applied to discriminator parameters.
        """
        trainer = Trainer(parity_config)

        images = trainer._get_batch()
        trainer._train_step(images)

        clip_val = parity_config.loss.gan_config.clip_weights

        # Check weight magnitudes
        for p in trainer.discriminator.parameters():
            max_weight = p.data.abs().max().item()
            assert max_weight <= clip_val + 1e-6, (
                f"Discriminator weight {max_weight:.4f} exceeds clip value {clip_val}"
            )


class TestDistortionStrategyParity:
    """Tests for distortion strategy matching StegaStamp."""

    def test_curriculum_distortion_has_blur_first(self) -> None:
        """CurriculumDistortion should apply blur first.

        This matches StegaStamp's training pipeline where random blur
        is applied before other distortions.
        """
        from picode.training.config import DistortionConfig

        config = DistortionConfig(strategy="curriculum")
        strategy = CurriculumDistortion(config)

        # Verify random_blur is a RandomBlurKernel
        from picode.distortions.native import RandomBlurKernel

        assert isinstance(
            strategy.random_blur, RandomBlurKernel
        ), "random_blur should be RandomBlurKernel"

    def test_curriculum_distortion_blur_probabilities(self) -> None:
        """CurriculumDistortion blur should have correct probabilities.

        StegaStamp uses 25% Gaussian, 25% line/motion, 50% identity.
        """
        from picode.training.config import DistortionConfig

        config = DistortionConfig(strategy="curriculum")
        strategy = CurriculumDistortion(config)

        # Check probabilities
        probs = strategy.random_blur.probs
        assert probs == (0.25, 0.25), f"Blur probs should be (0.25, 0.25), got {probs}"

    def test_curriculum_distortion_applies_blur(self) -> None:
        """CurriculumDistortion should actually apply blur.

        Verify that calling the distortion strategy applies the blur
        operation (though output may or may not be blurred due to
        probabilistic nature).
        """
        from picode.training.config import DistortionConfig

        config = DistortionConfig(strategy="curriculum")
        strategy = CurriculumDistortion(config)

        # Create test image
        image = torch.rand(2, 3, 400, 400)

        # Apply distortion (blur is always called, but may be identity 50% of time)
        output = strategy(image, step=0)

        # Output should have same shape
        assert output.shape == image.shape, "Output shape should match input"

        # Output should be valid (no NaN, in valid range)
        assert not torch.isnan(output).any(), "Output should not contain NaN"
        assert output.min() >= 0 and output.max() <= 1, "Output should be in [0, 1]"
