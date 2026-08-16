"""Tests for video detection."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from picode.detection import Detector
from picode.models.stegastamp import Decoder


class TestVideoDetection:
    """Tests for video detection."""

    @pytest.fixture
    def decoder(self) -> Decoder:
        """Create decoder."""
        return Decoder(num_bits=100)

    @pytest.fixture
    def temp_video(self, tmp_path: Path) -> Path:
        """Create a temporary test video (5 frames)."""
        video_path = tmp_path / "test.mp4"

        # Create video writer
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10, (200, 200))

        # Write 5 random frames
        for _ in range(5):
            frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            writer.write(frame)

        writer.release()
        return video_path

    def test_detect_video_returns_iterator(
        self, decoder: Decoder, temp_video: Path
    ) -> None:
        """detect_video returns frame iterator."""
        detector = Detector(decoder=decoder, scales=[0.5])

        results = list(detector.detect_video(temp_video, sample_rate=1))

        assert len(results) == 5  # 5 frames
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_detect_video_sample_rate(
        self, decoder: Decoder, temp_video: Path
    ) -> None:
        """sample_rate skips frames."""
        detector = Detector(decoder=decoder, scales=[0.5])

        results = list(detector.detect_video(temp_video, sample_rate=2))

        # 5 frames, sample every 2nd = frames 0, 2, 4 = 3 results
        assert len(results) == 3

    def test_detect_video_frame_numbers(
        self, decoder: Decoder, temp_video: Path
    ) -> None:
        """Frame numbers are correctly reported."""
        detector = Detector(decoder=decoder, scales=[0.5])

        results = list(detector.detect_video(temp_video, sample_rate=2))

        # Should have frames 0, 2, 4
        frame_nums = [r[0] for r in results]
        assert frame_nums == [0, 2, 4]

    def test_detect_video_invalid_path_raises(self, decoder: Decoder, tmp_path: Path) -> None:
        """Invalid video path raises ValueError."""
        detector = Detector(decoder=decoder, scales=[0.5])

        with pytest.raises(ValueError, match="Cannot open video"):
            list(detector.detect_video(tmp_path / "nonexistent.mp4"))

    def test_detect_video_result_types(
        self, decoder: Decoder, temp_video: Path
    ) -> None:
        """Each result is (int, Detection | None)."""
        detector = Detector(decoder=decoder, scales=[0.5])

        results = list(detector.detect_video(temp_video, sample_rate=1))

        for frame_num, detection in results:
            assert isinstance(frame_num, int)
            # Detection can be None or Detection object
            from picode.detection import Detection
            assert detection is None or isinstance(detection, Detection)
