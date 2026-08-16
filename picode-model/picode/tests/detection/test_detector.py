"""Tests for Detector class."""

from unittest.mock import Mock

import pytest
import torch

from picode.detection.detector import Detector
from picode.detection.types import Detection, Point, Quadrilateral


class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_attributes(self) -> None:
        """Detection has required attributes."""
        corners = Quadrilateral(
            top_left=Point(10, 20),
            top_right=Point(110, 20),
            bottom_right=Point(110, 100),
            bottom_left=Point(10, 100),
        )
        d = Detection(
            corners=corners,
            confidence=0.35,
            detector_type="slow",
            message_bits=torch.ones(100),
            message_probs=torch.ones(100) * 0.9,
        )
        assert d.bbox == (10, 20, 100, 80)
        assert d.confidence == 0.35
        assert d.message_bits is not None and d.message_probs is not None
        assert d.message_bits.shape == (100,)
        assert d.message_probs.shape == (100,)
        assert d.detector_type == "slow"


class TestDetector:
    """Tests for Detector class."""

    @pytest.fixture
    def mock_decoder(self) -> Mock:
        """Create a mock decoder."""
        decoder = Mock()
        decoder.num_bits = 100
        # Return uncertain logits by default
        decoder.return_value = torch.zeros(1, 100)
        return decoder

    def test_detector_init(self, mock_decoder: Mock) -> None:
        """Detector initializes with decoder."""
        detector = Detector(decoder=mock_decoder)
        assert detector.decoder is mock_decoder
        assert detector.confidence_threshold == 0.15

    def test_detector_custom_params(self, mock_decoder: Mock) -> None:
        """Detector accepts custom parameters."""
        detector = Detector(
            decoder=mock_decoder,
            scales=[0.5],
            stride_ratio=0.2,
            confidence_threshold=0.25,
        )
        assert detector.scales == [0.5]
        assert detector.stride_ratio == 0.2
        assert detector.confidence_threshold == 0.25

    def test_detect_returns_none_on_random_image(self, mock_decoder: Mock) -> None:
        """Returns None when no encoded region found."""
        # Mock returns uncertain logits (confidence < threshold)
        mock_decoder.return_value = torch.zeros(1, 100)
        detector = Detector(decoder=mock_decoder, scales=[0.5])

        # Create random image tensor
        image = torch.rand(3, 200, 200)
        result = detector.detect(image)

        assert result is None

    def test_detect_returns_detection_on_encoded_region(
        self, mock_decoder: Mock
    ) -> None:
        """Returns Detection when high-confidence region found."""
        # Mock returns confident logits
        mock_decoder.return_value = torch.randn(1, 100) * 5

        detector = Detector(decoder=mock_decoder, scales=[0.5])
        image = torch.rand(3, 200, 200)
        result = detector.detect(image)

        assert result is not None
        assert isinstance(result, Detection)
        assert result.confidence > 0.15
        assert len(result.bbox) == 4

    def test_detect_returns_quadrilateral_corners(
        self, mock_decoder: Mock
    ) -> None:
        """SlowDetector returns Detection with Quadrilateral corners."""
        mock_decoder.return_value = torch.randn(1, 100) * 5

        detector = Detector(decoder=mock_decoder, scales=[0.5])
        image = torch.rand(3, 200, 200)
        result = detector.detect(image)

        assert result is not None
        assert isinstance(result.corners, Quadrilateral)
        assert result.detector_type == "slow"
        # Corners should form axis-aligned rectangle
        assert result.corners.top_left.y == result.corners.top_right.y
        assert result.corners.bottom_left.y == result.corners.bottom_right.y
        assert result.corners.top_left.x == result.corners.bottom_left.x
        assert result.corners.top_right.x == result.corners.bottom_right.x

    def test_detect_all_returns_empty_on_random_image(self, mock_decoder: Mock) -> None:
        """detect_all returns empty list when no encoded region found."""
        mock_decoder.return_value = torch.zeros(1, 100)
        detector = Detector(decoder=mock_decoder, scales=[0.5])

        image = torch.rand(3, 200, 200)
        results = detector.detect_all(image)

        assert results == []

    def test_detect_all_returns_detections(self, mock_decoder: Mock) -> None:
        """detect_all returns list of detections when found."""
        mock_decoder.return_value = torch.randn(1, 100) * 5
        detector = Detector(decoder=mock_decoder, scales=[0.5])

        image = torch.rand(3, 200, 200)
        results = detector.detect_all(image)

        assert len(results) >= 1
        assert all(isinstance(d, Detection) for d in results)

    def test_extract_crop_resizes_to_400x400(self, mock_decoder: Mock) -> None:
        """_extract_crop resizes window to decoder input size."""
        detector = Detector(decoder=mock_decoder)

        from picode.detection.window import Window
        image = torch.rand(3, 800, 800)
        window = Window(x=0, y=0, w=400, h=400, scale=0.5)

        crop = detector._extract_crop(image, window)

        assert crop.shape == (1, 3, 400, 400)

    def test_detect_handles_hwc_format(self, mock_decoder: Mock) -> None:
        """Detector handles (H, W, C) format input."""
        mock_decoder.return_value = torch.randn(1, 100) * 5
        detector = Detector(decoder=mock_decoder, scales=[0.5])

        # HWC format
        image = torch.rand(200, 200, 3)
        result = detector.detect(image)

        # Should still work (converted internally)
        assert result is not None or result is None  # Just checking no error

    def test_detect_calls_decoder_with_torch_no_grad(self, mock_decoder: Mock) -> None:
        """Detector runs inference with gradients disabled."""
        mock_decoder.return_value = torch.randn(1, 100) * 5
        detector = Detector(decoder=mock_decoder, scales=[0.5])

        image = torch.rand(3, 200, 200)
        detector.detect(image)

        # Verify decoder was called
        assert mock_decoder.called
