"""Multi-model decode pipeline for watermark extraction.

Supports two routing strategies:
- **Strategy C** (default): Try each decoder sequentially; LDPC parity check
  validates whether the decoder matched. First success wins.
- **Strategy B**: A classifier picks the most likely model, then a single
  decode attempt is made.

Usage:
    >>> specs = [
    ...     ModelSpec(name="b72", num_bits=72, checkpoint_path="b72.pt"),
    ...     ModelSpec(name="b96", num_bits=96, checkpoint_path="b96.pt"),
    ... ]
    >>> pipeline = DecodePipeline(model_specs=specs, strategy="C")
    >>> result = pipeline.decode(image_tensor)
    >>> if result.success:
    ...     print(result.model_name, result.message)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """Specification for a single decoder model.

    Args:
        name: Human-readable model name (e.g. "b72").
        num_bits: Number of codeword bits this model produces.
        checkpoint_path: Path to the saved checkpoint file.
        decoder_size: Input image size for the decoder (default 512).
    """

    name: str
    num_bits: int
    checkpoint_path: str
    decoder_size: int = 512


@dataclass
class DecodeResult:
    """Result of a decode attempt.

    Args:
        success: Whether decoding succeeded (LDPC parity check passed).
        model_name: Name of the model that decoded successfully.
        message: Decoded message bits after LDPC decoding (k bits).
        raw_bits: Raw decoder output bits before LDPC (n bits).
        ldpc_success: Whether the LDPC parity check passed.
    """

    success: bool
    model_name: str | None = None
    message: Tensor | None = None
    raw_bits: Tensor | None = None
    ldpc_success: bool = False


class DecodePipeline:
    """Multi-model decode pipeline with configurable routing strategy.

    Args:
        model_specs: List of model specifications, tried in order for Strategy C.
        strategy: Routing strategy -- "C" (sequential) or "B" (classifier).
        detector_checkpoint: Path to FastDetector checkpoint for detection.
            If None, input images are assumed to be already rectified.
        classifier_checkpoint: Path to ModelClassifier checkpoint (Strategy B only).
        device: Torch device for inference.
        detection_threshold: Confidence threshold for watermark detection.
    """

    def __init__(
        self,
        model_specs: list[ModelSpec],
        strategy: Literal["B", "C"] = "C",
        detector_checkpoint: str | Path | None = None,
        classifier_checkpoint: str | Path | None = None,
        device: str = "cpu",
        detection_threshold: float = 0.5,
    ) -> None:
        self.model_specs = model_specs
        self.strategy = strategy
        self.device = device
        self.detection_threshold = detection_threshold

        # Lazy-loaded decoder cache: name -> decoder module
        self._decoders: dict[str, torch.nn.Module] = {}
        # LDPC codec cache: name -> LDPC instance
        # Typed as Any to avoid importing LDPC at module level; actual type is LDPC.
        from typing import Any

        self._ldpc_codecs: dict[str, Any] = {}

        # Load detector if checkpoint provided
        self._detector = None
        if detector_checkpoint is not None:
            from picode.detection.fast_detector import FastDetector

            self._detector = FastDetector.from_checkpoint(
                detector_checkpoint,
                device=device,
                threshold=detection_threshold,
            )

        # Load classifier for Strategy B if checkpoint provided
        self._classifier = None
        if classifier_checkpoint is not None:
            from picode.detection.model_classifier import ModelClassifier

            self._classifier = ModelClassifier.from_checkpoint(
                classifier_checkpoint, device=device,
            )

        # Rectifier for perspective correction
        from picode.detection.rectifier import Rectifier

        self._rectifier = Rectifier(output_size=512)

    def _get_decoder(self, spec: ModelSpec) -> torch.nn.Module:
        """Lazily load and cache a decoder for the given model spec.

        Args:
            spec: Model specification to load.

        Returns:
            The decoder module, ready for inference.
        """
        if spec.name not in self._decoders:
            from picode.models.factory import create_decoder
            from picode.training.config import ModelConfig

            model_config = ModelConfig(
                type="picotrust",
                decoder_size=spec.decoder_size,
            )
            decoder = create_decoder(model_config, num_bits=spec.num_bits)

            # Load checkpoint weights
            ckpt = torch.load(spec.checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("decoder_state_dict", ckpt.get("state_dict", ckpt))
            decoder.load_state_dict(state_dict)
            decoder.to(self.device)
            decoder.eval()
            self._decoders[spec.name] = decoder

            # Create matching LDPC codec
            from picode.ecc.ldpc import LDPC

            self._ldpc_codecs[spec.name] = LDPC(n=spec.num_bits, seed=42)

            logger.info("Loaded decoder '%s' (%d bits) from %s",
                        spec.name, spec.num_bits, spec.checkpoint_path)

        return self._decoders[spec.name]

    def _try_decode(self, rectified: Tensor, spec: ModelSpec) -> DecodeResult:
        """Attempt to decode a rectified image with a single model.

        Args:
            rectified: Rectified image tensor (3, H, W) in [0, 1].
            spec: Model specification to try.

        Returns:
            DecodeResult indicating success or failure.
        """
        decoder = self._get_decoder(spec)
        ldpc = self._ldpc_codecs[spec.name]

        with torch.no_grad():
            # Decoder expects (B, 3, H, W)
            batch = rectified.unsqueeze(0).to(self.device)
            logits = decoder(batch)  # (1, num_bits)

        # Convert logits to probabilities
        probs = torch.sigmoid(logits)  # (1, num_bits)

        # LDPC decode -- returns (decoded, success) tensors
        decoded, success = ldpc.decode(probs)

        ldpc_ok = bool(success[0].item())
        if ldpc_ok:
            return DecodeResult(
                success=True,
                model_name=spec.name,
                message=decoded[0],
                raw_bits=(probs[0] > 0.5).float(),
                ldpc_success=True,
            )
        return DecodeResult(success=False)

    def _try_all_decoders(self, rectified: Tensor) -> DecodeResult:
        """Strategy C: try each decoder sequentially until one succeeds.

        Args:
            rectified: Rectified image tensor (3, H, W) in [0, 1].

        Returns:
            DecodeResult from the first successful decoder, or failure.
        """
        for spec in self.model_specs:
            result = self._try_decode(rectified, spec)
            if result.success:
                logger.info("Strategy C: decoded with model '%s'", spec.name)
                return result
            logger.debug("Strategy C: model '%s' failed LDPC check", spec.name)

        return DecodeResult(success=False)

    def _classify_and_decode(self, rectified: Tensor) -> DecodeResult:
        """Strategy B: use classifier to pick a model, then decode.

        Args:
            rectified: Rectified image tensor (3, H, W) in [0, 1].

        Returns:
            DecodeResult from the classified model.
        """
        if self._classifier is None:
            raise RuntimeError(
                "Strategy B requires a classifier_checkpoint to be provided."
            )

        # Classify which model to use
        predicted_label = self._classifier.predict(rectified.unsqueeze(0).to(self.device))

        # Find matching spec
        for spec in self.model_specs:
            if spec.name == predicted_label:
                logger.info("Strategy B: classifier picked model '%s'", spec.name)
                return self._try_decode(rectified, spec)

        logger.warning(
            "Strategy B: classifier returned unknown label '%s'", predicted_label,
        )
        return DecodeResult(success=False)

    def decode(self, image: Tensor | str | Path) -> DecodeResult:
        """Run the full decode pipeline on an image.

        Pipeline steps:
        1. If a detector is configured, detect the watermark region.
        2. Rectify the detected region to a square tensor.
        3. Route to the appropriate decoding strategy (C or B).

        If no detector is configured, the image tensor is assumed to be
        already rectified (3, H, W).

        Args:
            image: Input image as a (3, H, W) tensor, or a file path string.

        Returns:
            DecodeResult with the decoded message on success.
        """
        # Load image from path if needed
        img_tensor: Tensor
        if isinstance(image, (str, Path)):
            from PIL import Image as PILImage
            from torchvision import transforms

            pil_img = PILImage.open(image).convert("RGB")
            img_tensor = transforms.ToTensor()(pil_img)
        else:
            img_tensor = image

        if self._detector is not None:
            detection = self._detector.detect(img_tensor)
            if detection is None:
                logger.info("No watermark detected in image.")
                return DecodeResult(success=False)

            # Rectify detected region
            rectified = self._rectifier.rectify(img_tensor, detection.corners)
        else:
            # Assume already rectified
            rectified = img_tensor

        # Route based on strategy
        if self.strategy == "C":
            return self._try_all_decoders(rectified)
        elif self.strategy == "B":
            return self._classify_and_decode(rectified)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}. Use 'B' or 'C'.")
