# picode/tests/detection/test_fast_detector.py
"""Tests for FastDetector model and inference."""

from unittest.mock import Mock

import pytest
import torch
from torch import Tensor

from picode.detection.fast_detector import FastDetector, FastDetectorModel
from picode.detection.types import Detection, Quadrilateral


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


class TestFastDetector:
    """Tests for high-level FastDetector API."""

    @pytest.fixture
    def mock_model(self) -> Mock:
        model = Mock(spec=FastDetectorModel)
        model.input_size = 320
        model.eval = Mock(return_value=model)
        model.to = Mock(return_value=model)
        # Return positive detection with corners
        model.return_value = {
            "is_watermark": torch.tensor([[2.0]]),  # Positive logit
            "corners": torch.tensor([[0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]]),
            "corner_confidence": torch.tensor([[0.8]]),
        }
        return model

    @pytest.fixture
    def detector(self, mock_model: Mock) -> FastDetector:
        return FastDetector(model=mock_model, threshold=0.5, device="cpu")

    @pytest.fixture
    def sample_image_tensor(self) -> Tensor:
        return torch.rand(3, 480, 640)

    def test_detector_creation(self, detector: FastDetector) -> None:
        assert detector.threshold == 0.5
        assert detector.device == "cpu"

    def test_detect_returns_detection(
        self, detector: FastDetector, sample_image_tensor: Tensor
    ) -> None:
        result = detector.detect(sample_image_tensor)

        assert result is not None
        assert isinstance(result, Detection)
        assert isinstance(result.corners, Quadrilateral)
        assert result.detector_type == "fast"

    def test_detect_returns_none_below_threshold(
        self, mock_model: Mock, sample_image_tensor: Tensor
    ) -> None:
        # Set model to return low confidence
        mock_model.return_value = {
            "is_watermark": torch.tensor([[-2.0]]),  # Negative logit
            "corners": torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]),
            "corner_confidence": torch.tensor([[0.1]]),
        }
        detector = FastDetector(model=mock_model, threshold=0.5, device="cpu")

        result = detector.detect(sample_image_tensor)

        assert result is None

    def test_detect_corners_denormalized(
        self, detector: FastDetector, sample_image_tensor: Tensor
    ) -> None:
        result = detector.detect(sample_image_tensor)

        assert result is not None
        # Corners should be in pixel coordinates, not normalized
        # Original image is 640x480, so corners should be scaled
        assert result.corners.top_left.x > 1.0 or result.corners.top_left.y > 1.0

    def test_detect_confidence_in_result(
        self, detector: FastDetector, sample_image_tensor: Tensor
    ) -> None:
        result = detector.detect(sample_image_tensor)

        assert result is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_batch(
        self, detector: FastDetector, mock_model: Mock
    ) -> None:
        images = [torch.rand(3, 480, 640) for _ in range(3)]

        # Mock batch output
        mock_model.return_value = {
            "is_watermark": torch.tensor([[2.0], [2.0], [-2.0]]),
            "corners": torch.tensor([
                [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
                [0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            ]),
            "corner_confidence": torch.tensor([[0.8], [0.7], [0.1]]),
        }

        results = detector.detect_batch(images)

        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is not None
        assert results[2] is None  # Below threshold

    def test_detect_with_low_corner_confidence(
        self, mock_model: Mock, sample_image_tensor: Tensor
    ) -> None:
        # High classification confidence but low corner confidence
        mock_model.return_value = {
            "is_watermark": torch.tensor([[2.0]]),
            "corners": torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]),
            "corner_confidence": torch.tensor([[0.1]]),  # Low confidence
        }
        detector = FastDetector(
            model=mock_model,
            threshold=0.5,
            corner_confidence_threshold=0.3,
            device="cpu",
        )

        result = detector.detect(sample_image_tensor)

        assert result is None  # Rejected due to low corner confidence
