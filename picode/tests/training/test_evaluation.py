"""Tests for evaluation."""

import torch

from picode.models.stegastamp import Decoder, Encoder
from picode.training.evaluation import (
    DEFAULT_ROBUSTNESS_SWEEP,
    EvalMetrics,
    Evaluator,
    RobustnessResult,
)


class TestEvalMetrics:
    def test_dataclass(self) -> None:
        metrics = EvalMetrics(
            bit_accuracy=0.95,
            message_accuracy=0.8,
            psnr=35.0,
            ssim=0.98,
            lpips=None,
        )
        assert metrics.bit_accuracy == 0.95
        assert metrics.lpips is None


class TestEvaluator:
    def test_evaluate_basic(self) -> None:
        encoder = Encoder(num_bits=10)
        decoder = Decoder(num_bits=10)
        device = torch.device("cpu")

        evaluator = Evaluator(encoder, decoder, device)

        # Create simple dataloader
        images = [torch.rand(2, 3, 400, 400) for _ in range(2)]

        class SimpleLoader:
            def __iter__(self):
                return iter(images)

        metrics = evaluator.evaluate(SimpleLoader(), num_bits=10, max_batches=2)
        assert 0.0 <= metrics.bit_accuracy <= 1.0
        assert 0.0 <= metrics.message_accuracy <= 1.0
        assert metrics.psnr > 0

    def test_robustness_sweep(self) -> None:
        encoder = Encoder(num_bits=10)
        decoder = Decoder(num_bits=10)
        device = torch.device("cpu")

        evaluator = Evaluator(encoder, decoder, device)

        images = torch.rand(2, 3, 400, 400)
        messages = torch.randint(0, 2, (2, 10)).float()

        results = evaluator.robustness_sweep(
            images, messages, {"noise": [0.01, 0.05]}
        )

        assert len(results) == 2
        assert all(isinstance(r, RobustnessResult) for r in results)
        assert results[0].distortion == "noise"
        assert results[0].strength == 0.01

    def test_default_robustness_sweep_defined(self) -> None:
        """Test that DEFAULT_ROBUSTNESS_SWEEP contains expected distortions."""
        assert "jpeg" in DEFAULT_ROBUSTNESS_SWEEP
        assert "noise" in DEFAULT_ROBUSTNESS_SWEEP
        assert "blur" in DEFAULT_ROBUSTNESS_SWEEP
        assert "brightness" in DEFAULT_ROBUSTNESS_SWEEP
        assert "contrast" in DEFAULT_ROBUSTNESS_SWEEP
