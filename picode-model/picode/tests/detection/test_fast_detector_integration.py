# picode/tests/detection/test_fast_detector_integration.py
"""Integration tests for FastDetector pipeline."""

import pytest
import torch
from torch import Tensor

from picode.detection import (
    FastDetector,
    FastDetectorModel,
    Quadrilateral,
    Rectifier,
)
from picode.detection.types import Detection, Point


class TestFastDetectorIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def model(self) -> FastDetectorModel:
        """Untrained model for shape/flow testing."""
        return FastDetectorModel(input_size=320, pretrained=False)

    @pytest.fixture
    def detector(self, model: FastDetectorModel) -> FastDetector:
        return FastDetector(
            model=model, threshold=0.5, corner_confidence_threshold=0.0, device="cpu"
        )

    @pytest.fixture
    def rectifier(self) -> Rectifier:
        return Rectifier(output_size=400)

    @pytest.fixture
    def sample_image(self) -> Tensor:
        return torch.rand(3, 480, 640)

    def test_model_to_detector_flow(
        self, model: FastDetectorModel, sample_image: Tensor
    ) -> None:
        """Test model output can be used by detector."""
        # Direct model call
        output = model(sample_image.unsqueeze(0))

        assert output["is_watermark"].shape == (1, 1)
        assert output["corners"].shape == (1, 8)

    def test_detector_to_rectifier_flow(
        self, detector: FastDetector, rectifier: Rectifier, sample_image: Tensor
    ) -> None:
        """Test detector output can be used by rectifier."""
        # Get detection (may be None with untrained model)
        result = detector.detect(sample_image)

        # If no detection, create synthetic one for testing
        if result is None:
            result = Detection(
                corners=Quadrilateral(
                    top_left=Point(100.0, 100.0),
                    top_right=Point(500.0, 100.0),
                    bottom_right=Point(500.0, 400.0),
                    bottom_left=Point(100.0, 400.0),
                ),
                confidence=0.8,
                detector_type="fast",
            )

        # Rectify using detection corners
        rectified = rectifier.rectify(sample_image, result.corners)

        assert rectified.shape == (3, 400, 400)

    def test_full_pipeline_shapes(
        self, model: FastDetectorModel, rectifier: Rectifier, sample_image: Tensor
    ) -> None:
        """Test full pipeline maintains correct shapes."""
        # 1. Model inference
        model_input = torch.nn.functional.interpolate(
            sample_image.unsqueeze(0), size=(320, 320), mode="bilinear"
        )
        output = model(model_input)

        # 2. Extract corners (denormalize)
        corners_norm = output["corners"][0]
        h, w = sample_image.shape[1], sample_image.shape[2]
        corners_px = corners_norm.clone()
        corners_px[0::2] *= w
        corners_px[1::2] *= h

        # 3. Rectify
        rectified = rectifier.rectify(sample_image, corners_px)

        assert rectified.shape == (3, 400, 400)
        assert rectified.min() >= 0.0
        assert rectified.max() <= 1.0

    def test_batch_consistency(
        self, detector: FastDetector, sample_image: Tensor
    ) -> None:
        """Test batch detection gives same results as individual."""
        images = [sample_image, sample_image.clone()]

        # Batch detection
        batch_results = detector.detect_batch(images)

        # Individual detection
        individual_results = [detector.detect(img) for img in images]

        # Results should be consistent (both None or both Detection)
        for batch_res, ind_res in zip(batch_results, individual_results):
            if batch_res is None:
                assert ind_res is None
            else:
                assert ind_res is not None
                # Confidence should be very close
                assert abs(batch_res.confidence - ind_res.confidence) < 0.01

    def test_device_consistency(self, sample_image: Tensor) -> None:
        """Test model works on available device."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = FastDetectorModel(input_size=320, pretrained=False)
        detector = FastDetector(model=model, threshold=0.5, device=device)

        # Should not raise
        result = detector.detect(sample_image)
        # Result is None or Detection (either is fine for untrained model)
        assert result is None or result.detector_type == "fast"
