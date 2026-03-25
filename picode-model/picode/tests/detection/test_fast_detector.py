# picode/tests/detection/test_fast_detector.py
"""Tests for FastDetector model and inference."""

import pytest
import torch
from torch import Tensor

from picode.detection.fast_detector import FastDetectorModel


class TestFastDetectorModel:
    @pytest.fixture
    def model(self) -> FastDetectorModel:
        return FastDetectorModel(input_size=320, pretrained=False)

    @pytest.fixture
    def sample_input(self) -> Tensor:
        """Batch of 2 images at 320x320."""
        return torch.rand(2, 3, 320, 320)

    def test_model_creation(self, model: FastDetectorModel) -> None:
        assert model.input_size == 320
        assert hasattr(model, "features")
        assert hasattr(model, "cls_head")
        assert hasattr(model, "corner_head")
        assert hasattr(model, "conf_head")

    def test_forward_output_shapes(
        self, model: FastDetectorModel, sample_input: Tensor
    ) -> None:
        output = model(sample_input)

        assert "is_watermark" in output
        assert "corners" in output
        assert "corner_confidence" in output

        assert output["is_watermark"].shape == (2, 1)
        assert output["corners"].shape == (2, 8)
        assert output["corner_confidence"].shape == (2, 1)

    def test_corners_normalized(
        self, model: FastDetectorModel, sample_input: Tensor
    ) -> None:
        output = model(sample_input)
        corners = output["corners"]

        # Corners should be in [0, 1] due to sigmoid
        assert corners.min() >= 0.0
        assert corners.max() <= 1.0

    def test_confidence_normalized(
        self, model: FastDetectorModel, sample_input: Tensor
    ) -> None:
        output = model(sample_input)
        conf = output["corner_confidence"]

        # Confidence should be in [0, 1] due to sigmoid
        assert conf.min() >= 0.0
        assert conf.max() <= 1.0

    def test_gradient_flow(
        self, model: FastDetectorModel, sample_input: Tensor
    ) -> None:
        sample_input.requires_grad = True
        output = model(sample_input)

        # Backward through classification
        loss = output["is_watermark"].sum()
        loss.backward()

        assert sample_input.grad is not None
        assert sample_input.grad.shape == sample_input.shape

    def test_eval_mode(
        self, model: FastDetectorModel, sample_input: Tensor
    ) -> None:
        model.eval()
        with torch.no_grad():
            output1 = model(sample_input)
            output2 = model(sample_input)

        # Deterministic in eval mode
        assert torch.allclose(output1["corners"], output2["corners"])

    def test_device_handling(self, sample_input: Tensor) -> None:
        model = FastDetectorModel(input_size=320, pretrained=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        input_on_device = sample_input.to(device)

        output = model(input_on_device)

        assert output["is_watermark"].device == device
        assert output["corners"].device == device

    def test_parameter_count(self, model: FastDetectorModel) -> None:
        total_params = sum(p.numel() for p in model.parameters())
        # Should be around 1.2M params
        assert 1_000_000 < total_params < 2_000_000
