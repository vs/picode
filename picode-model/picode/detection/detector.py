"""Blind detector for steganographic images."""

from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from picode.detection.confidence import compute_confidence
from picode.detection.types import Detection, Point, Quadrilateral
from picode.detection.window import Window, WindowGenerator
from picode.models.base import Decoder as BaseDecoder


class Detector:
    """Blind detector using decoder confidence sweeping.

    Slides windows across the image, runs decoder on each crop,
    and identifies regions with high bit confidence.

    Args:
        decoder: Trained decoder model.
        scales: Window scales relative to frame size.
        stride_ratio: Stride as fraction of window size.
        confidence_threshold: Minimum confidence to report detection.
        device: Device for inference.
    """

    def __init__(
        self,
        decoder: BaseDecoder,
        scales: list[float] | None = None,
        stride_ratio: float = 0.15,
        confidence_threshold: float = 0.15,
        device: str | torch.device = "cpu",
    ) -> None:
        self.decoder = decoder
        self.scales = scales or [0.25, 0.35, 0.5, 0.65, 0.75]
        self.stride_ratio = stride_ratio
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device)

        self.window_gen = WindowGenerator(
            scales=self.scales,
            stride_ratio=self.stride_ratio,
        )

    def _extract_crop(self, image: Tensor, window: Window) -> Tensor:
        """Extract and resize window crop to 400x400.

        Args:
            image: (C, H, W) image tensor.
            window: Window specifying crop region.

        Returns:
            (1, C, 400, 400) tensor ready for decoder.
        """
        crop = image[:, window.y : window.y + window.h, window.x : window.x + window.w]
        # Resize to decoder input size (400x400)
        crop = F.interpolate(
            crop.unsqueeze(0), size=(400, 400), mode="bilinear", align_corners=False
        )
        return crop

    def detect(self, image: Tensor | Path | str) -> Detection | None:
        """Find encoded region in image.

        Args:
            image: Image as tensor (C, H, W) or (H, W, C), or path to image file.

        Returns:
            Detection with highest confidence above threshold, or None.
        """
        # Load image if path
        img_tensor: Tensor
        if isinstance(image, (str, Path)):
            from PIL import Image as PILImage
            from torchvision.transforms.functional import to_tensor

            pil_image = PILImage.open(image).convert("RGB")
            img_tensor = to_tensor(pil_image)
        else:
            img_tensor = image

        # Ensure (C, H, W) format
        if img_tensor.dim() == 3 and img_tensor.shape[2] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)

        img_tensor = img_tensor.to(self.device)
        _, frame_h, frame_w = img_tensor.shape

        best_detection: Detection | None = None
        best_confidence: float = 0.0

        # Sweep all windows
        for window in self.window_gen.generate(frame_h, frame_w):
            crop = self._extract_crop(img_tensor, window)
            crop = crop.to(self.device)

            # Run decoder
            with torch.no_grad():
                logits = self.decoder(crop)
                if logits.dim() == 2:
                    logits = logits.squeeze(0)

            # Compute confidence
            confidence = compute_confidence(logits)

            if confidence > best_confidence:
                best_confidence = confidence
                probs = torch.sigmoid(logits)
                bits = (probs > 0.5).float()
                # Create Quadrilateral from axis-aligned bounding box
                corners = Quadrilateral(
                    top_left=Point(float(window.x), float(window.y)),
                    top_right=Point(float(window.x + window.w), float(window.y)),
                    bottom_right=Point(float(window.x + window.w), float(window.y + window.h)),
                    bottom_left=Point(float(window.x), float(window.y + window.h)),
                )
                best_detection = Detection(
                    corners=corners,
                    confidence=confidence,
                    detector_type="slow",
                    message_bits=bits.cpu(),
                    message_probs=probs.cpu(),
                )

        # Return best if above threshold
        if best_detection and best_detection.confidence >= self.confidence_threshold:
            return best_detection
        return None

    def detect_all(self, image: Tensor | Path | str) -> list[Detection]:
        """Find all encoded regions above threshold.

        Args:
            image: Image as tensor or path.

        Returns:
            List of detections sorted by confidence (highest first).
        """
        # For V1, just return single best detection
        result = self.detect(image)
        return [result] if result else []

    def detect_video(
        self,
        video_path: Path | str,
        sample_rate: int = 1,
    ) -> Iterator[tuple[int, Detection | None]]:
        """Detect in video frames.

        Args:
            video_path: Path to video file.
            sample_rate: Process every Nth frame.

        Yields:
            Tuples of (frame_number, detection_or_none).
        """
        import cv2

        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_num = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % sample_rate == 0:
                    # Convert BGR to RGB and to tensor
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = (
                        torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                    )

                    detection = self.detect(frame_tensor)
                    yield (frame_num, detection)

                frame_num += 1
        finally:
            cap.release()
