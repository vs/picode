# picode/detection/pipeline.py
"""End-to-end detection pipeline: detect → rectify → decode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn
from torch import Tensor

from picode.detection.confidence import compute_confidence
from picode.detection.rectifier import Rectifier
from picode.detection.types import Detection


class DetectorProtocol(Protocol):
    """Protocol for detector classes."""

    def detect(self, image: Tensor | str | Path) -> Detection | None:
        ...


@dataclass
class PipelineResult:
    """Result from the detection pipeline.

    Attributes:
        detection: Detection with corners and confidence
        rectified: Rectified image (3, 400, 400)
        message_bits: Decoded binary message
        message_probs: Bit probabilities
        decode_confidence: Confidence from decoder
    """

    detection: Detection
    rectified: Tensor
    message_bits: Tensor
    message_probs: Tensor
    decode_confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "detection_confidence": self.detection.confidence,
            "detector_type": self.detection.detector_type,
            "bbox": self.detection.bbox,
            "corners": self.detection.corners.to_tensor().tolist(),
            "message_bits": self.message_bits.tolist(),
            "decode_confidence": self.decode_confidence,
        }


class DetectionPipeline:
    """End-to-end pipeline for watermark detection and decoding.

    Combines:
    1. Detection (FastDetector or SlowDetector)
    2. Rectification (perspective correction)
    3. Decoding (message extraction)

    Args:
        detector: FastDetector or SlowDetector instance
        decoder: Decoder model for message extraction
        rectifier_size: Output size for rectification (default 400)
        device: Device for inference

    Example:
        >>> detector = FastDetector.from_checkpoint("detector.pt")
        >>> decoder = Decoder()
        >>> decoder.load_state_dict(torch.load("decoder.pt"))
        >>> pipeline = DetectionPipeline(detector, decoder)
        >>> result = pipeline.process("photo.jpg")
        >>> if result:
        ...     print(f"Message: {result.message_bits}")
    """

    def __init__(
        self,
        detector: DetectorProtocol,
        decoder: nn.Module,
        rectifier_size: int = 400,
        device: str = "cpu",
    ) -> None:
        self.detector = detector
        self.decoder = decoder
        self.decoder.to(device)
        self.decoder.eval()
        self.rectifier = Rectifier(output_size=rectifier_size)
        self.device = device

    def process(self, image: Tensor | str | Path) -> PipelineResult | None:
        """Process single image through full pipeline.

        Args:
            image: Input image (tensor, path, or PIL-compatible)

        Returns:
            PipelineResult if watermark detected, None otherwise
        """
        # Step 1: Detect
        detection = self.detector.detect(image)
        if detection is None:
            return None

        # Ensure we have a tensor for rectification
        if isinstance(image, (str, Path)):
            from PIL import Image
            from torchvision import transforms

            pil_image = Image.open(image).convert("RGB")
            image_tensor = transforms.ToTensor()(pil_image)
        else:
            image_tensor = image

        # Step 2: Rectify
        rectified = self.rectifier.rectify(image_tensor, detection.corners)

        # Step 3: Decode
        with torch.no_grad():
            rectified_batch = rectified.unsqueeze(0).to(self.device)
            logits = self.decoder(rectified_batch)

        # Convert logits to bits and probabilities
        probs = torch.sigmoid(logits).squeeze(0)
        bits = (probs > 0.5).float()

        # Compute decode confidence
        decode_conf = compute_confidence(logits.squeeze(0))

        return PipelineResult(
            detection=detection,
            rectified=rectified,
            message_bits=bits,
            message_probs=probs,
            decode_confidence=decode_conf,
        )

    def process_batch(
        self, images: list[Tensor | str | Path]
    ) -> list[PipelineResult | None]:
        """Process multiple images.

        Args:
            images: List of input images

        Returns:
            List of PipelineResult or None for each image
        """
        return [self.process(img) for img in images]
