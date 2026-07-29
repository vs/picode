# picode/tests/detection/test_pipeline.py
"""Tests for DetectionPipeline."""

from unittest.mock import Mock

import pytest
import torch
from torch import Tensor

from picode.detection.pipeline import DetectionPipeline, PipelineResult
from picode.detection.types import Detection, Point, Quadrilateral


class TestPipelineResult:
    def test_result_creation(self) -> None:
        detection = Detection(
            corners=Quadrilateral(
                top_left=Point(100, 100),
                top_right=Point(500, 100),
                bottom_right=Point(500, 400),
                bottom_left=Point(100, 400),
            ),
            confidence=0.9,
            detector_type="fast",
        )
        result = PipelineResult(
            detection=detection,
            rectified=torch.rand(3, 400, 400),
            message_bits=torch.randint(0, 2, (100,)),
            message_probs=torch.rand(100),
            decode_confidence=0.85,
        )

        assert result.detection is not None
        assert result.rectified.shape == (3, 400, 400)
        assert result.message_bits.shape == (100,)
        assert result.decode_confidence == 0.85

    def test_result_to_dict(self) -> None:
        detection = Detection(
            corners=Quadrilateral(
                top_left=Point(100, 100),
                top_right=Point(500, 100),
                bottom_right=Point(500, 400),
                bottom_left=Point(100, 400),
            ),
            confidence=0.9,
            detector_type="fast",
        )
        result = PipelineResult(
            detection=detection,
            rectified=torch.rand(3, 400, 400),
            message_bits=torch.randint(0, 2, (100,)),
            message_probs=torch.rand(100),
            decode_confidence=0.85,
        )

        d = result.to_dict()

        assert "detection_confidence" in d
        assert "decode_confidence" in d
        assert "message_bits" in d
        assert "bbox" in d


class TestDetectionPipeline:
    @pytest.fixture
    def mock_detector(self) -> Mock:
        detector = Mock()
        detector.detect = Mock(
            return_value=Detection(
                corners=Quadrilateral(
                    top_left=Point(100, 100),
                    top_right=Point(500, 100),
                    bottom_right=Point(500, 400),
                    bottom_left=Point(100, 400),
                ),
                confidence=0.9,
                detector_type="fast",
            )
        )
        return detector

    @pytest.fixture
    def mock_decoder(self) -> Mock:
        decoder = Mock()
        # Return logits (will be sigmoidized)
        decoder.return_value = torch.randn(1, 100)
        decoder.to = Mock(return_value=decoder)
        decoder.eval = Mock(return_value=decoder)
        return decoder

    @pytest.fixture
    def pipeline(self, mock_detector: Mock, mock_decoder: Mock) -> DetectionPipeline:
        return DetectionPipeline(
            detector=mock_detector,
            decoder=mock_decoder,
            device="cpu",
        )

    @pytest.fixture
    def sample_image(self) -> Tensor:
        return torch.rand(3, 480, 640)

    def test_pipeline_creation(self, pipeline: DetectionPipeline) -> None:
        assert pipeline.detector is not None
        assert pipeline.decoder is not None
        assert pipeline.rectifier is not None

    def test_process_returns_result(
        self, pipeline: DetectionPipeline, sample_image: Tensor
    ) -> None:
        result = pipeline.process(sample_image)

        assert result is not None
        assert isinstance(result, PipelineResult)
        assert result.detection is not None
        assert result.rectified is not None

    def test_process_returns_none_no_detection(
        self, mock_decoder: Mock, sample_image: Tensor
    ) -> None:
        # Detector returns None
        detector = Mock()
        detector.detect = Mock(return_value=None)

        pipeline = DetectionPipeline(
            detector=detector,
            decoder=mock_decoder,
            device="cpu",
        )

        result = pipeline.process(sample_image)

        assert result is None

    def test_process_calls_detector(
        self, pipeline: DetectionPipeline, mock_detector: Mock, sample_image: Tensor
    ) -> None:
        pipeline.process(sample_image)

        mock_detector.detect.assert_called_once()

    def test_process_calls_decoder(
        self, pipeline: DetectionPipeline, mock_decoder: Mock, sample_image: Tensor
    ) -> None:
        pipeline.process(sample_image)

        mock_decoder.assert_called_once()

    def test_rectified_image_shape(
        self, pipeline: DetectionPipeline, sample_image: Tensor
    ) -> None:
        result = pipeline.process(sample_image)

        assert result is not None
        assert result.rectified.shape == (3, 400, 400)

    def test_message_bits_binary(
        self, pipeline: DetectionPipeline, sample_image: Tensor
    ) -> None:
        result = pipeline.process(sample_image)

        assert result is not None
        # Message bits should be 0 or 1
        assert torch.all((result.message_bits == 0) | (result.message_bits == 1))

    def test_process_batch(
        self, pipeline: DetectionPipeline, mock_detector: Mock
    ) -> None:
        images = [torch.rand(3, 480, 640) for _ in range(3)]

        # Return detection for first two, None for third
        mock_detector.detect = Mock(
            side_effect=[
                Detection(
                    corners=Quadrilateral(
                        top_left=Point(100, 100),
                        top_right=Point(500, 100),
                        bottom_right=Point(500, 400),
                        bottom_left=Point(100, 400),
                    ),
                    confidence=0.9,
                    detector_type="fast",
                ),
                Detection(
                    corners=Quadrilateral(
                        top_left=Point(100, 100),
                        top_right=Point(500, 100),
                        bottom_right=Point(500, 400),
                        bottom_left=Point(100, 400),
                    ),
                    confidence=0.8,
                    detector_type="fast",
                ),
                None,
            ]
        )

        results = pipeline.process_batch(images)

        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is not None
        assert results[2] is None

    def test_decode_confidence_computed(
        self, pipeline: DetectionPipeline, sample_image: Tensor
    ) -> None:
        result = pipeline.process(sample_image)

        assert result is not None
        assert 0.0 <= result.decode_confidence <= 1.0
