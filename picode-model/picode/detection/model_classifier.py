# picode/detection/model_classifier.py
"""Lightweight model classifier for Strategy B routing.

Predicts which encoder model was used to produce a watermarked image so the
decode pipeline can skip trying every decoder (Strategy C) and go directly to
the right one.

Architecture: MobileNetV3-Small backbone (576 channels) + AdaptiveAvgPool2d(1)
+ classification head (Linear → ReLU6 → Dropout → Linear).

Usage:
    >>> model = ModelClassifierModel(num_classes=4, class_names=["b30", "b48", "b72", "b96"])
    >>> logits = model(torch.rand(2, 3, 320, 320))  # (2, 4)

    >>> classifier = ModelClassifier.from_checkpoint("classifier.pt")
    >>> label = classifier.classify(image_tensor)  # e.g. "b72"
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor


class ModelClassifierModel(nn.Module):
    """MobileNetV3-Small classifier that predicts the encoder model type.

    Architecture:
        - MobileNetV3-Small backbone (576 channels)
        - AdaptiveAvgPool2d(1) global pooling
        - Head: Linear(576, 128) → ReLU6 → Dropout(0.2) → Linear(128, num_classes)
        - Kaiming init on head weights

    Args:
        num_classes: Number of model classes to distinguish (e.g. 4 for b30/b48/b72/b96).
        class_names: Human-readable label for each class index.
        input_size: Expected spatial input size (used for documentation; not enforced).
        pretrained: Whether to use ImageNet-pretrained backbone weights.

    Example:
        >>> model = ModelClassifierModel(num_classes=4, class_names=["b30", "b48", "b72", "b96"])
        >>> x = torch.rand(2, 3, 320, 320)
        >>> model(x).shape
        torch.Size([2, 4])
    """

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        input_size: int = 320,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if len(class_names) != num_classes:
            raise ValueError(
                f"len(class_names)={len(class_names)} must equal num_classes={num_classes}"
            )
        self.num_classes = num_classes
        self.class_names = class_names
        self.input_size = input_size

        # MobileNetV3-Small backbone — same as FastDetectorModel
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features  # Output: (B, 576, H/32, W/32)

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        # Kaiming init on head layers
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming normal init to all Linear layers in the head."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input images (B, 3, H, W) in [0, 1].

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        features = self.features(x)  # (B, 576, H/32, W/32)
        pooled = self.pool(features).flatten(1)  # (B, 576)
        result: Tensor = self.head(pooled)
        return result  # (B, num_classes)

    def predict(self, x: Tensor) -> str:
        """Predict the model class for a single image.

        Args:
            x: Single image tensor — either (3, H, W) or (1, 3, H, W).

        Returns:
            Predicted class name string.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            logits = self(x)  # (1, num_classes)
        idx = int(logits.argmax(dim=1).item())
        return self.class_names[idx]

    def predict_batch(self, x: Tensor) -> list[str]:
        """Predict model classes for a batch of images.

        Args:
            x: Batch tensor of shape (B, 3, H, W).

        Returns:
            List of predicted class name strings, length B.
        """
        with torch.no_grad():
            logits = self(x)  # (B, num_classes)
        indices = logits.argmax(dim=1).tolist()
        return [self.class_names[i] for i in indices]


class ModelClassifier:
    """High-level API for model classification (Strategy B).

    Wraps :class:`ModelClassifierModel` with checkpoint loading and tensor
    preprocessing, mirroring the ``FastDetector`` high-level API.

    Example:
        >>> classifier = ModelClassifier.from_checkpoint("classifier.pt")
        >>> label = classifier.classify(image_tensor)
        >>> print(label)  # e.g. "b72"
    """

    def __init__(
        self,
        model: ModelClassifierModel,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize ModelClassifier.

        Args:
            model: Trained ModelClassifierModel instance.
            device: Device to run inference on.
        """
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.device = device if isinstance(device, str) else str(device)
        self.input_size = model.input_size

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> ModelClassifier:
        """Load from a saved checkpoint.

        Expected checkpoint keys:
            - ``model_state_dict``: state dict for :class:`ModelClassifierModel`.
            - ``num_classes``: int.
            - ``class_names``: list[str].
            - ``input_size``: int (optional, defaults to 320).

        Args:
            checkpoint_path: Path to the ``.pt`` checkpoint file.
            device: Device string to load onto.

        Returns:
            Initialized :class:`ModelClassifier`.
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        num_classes: int = ckpt["num_classes"]
        class_names: list[str] = ckpt["class_names"]
        input_size: int = ckpt.get("input_size", 320)

        model = ModelClassifierModel(
            num_classes=num_classes,
            class_names=class_names,
            input_size=input_size,
            pretrained=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model, device=device)

    def classify(self, image: Tensor) -> str:
        """Classify a single image and return the predicted model name.

        Args:
            image: Image tensor — either (3, H, W) or (1, 3, H, W), values in [0, 1].
                If the spatial dimensions differ from ``input_size``, the image
                will be resized bilinearly before inference.

        Returns:
            Predicted model name string (e.g. ``"b72"``).
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # → (1, 3, H, W)

        # Resize to expected input size if needed
        h, w = image.shape[-2], image.shape[-1]
        if h != self.input_size or w != self.input_size:
            image = F.interpolate(
                image,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        image = image.to(self.device)
        return self.model.predict(image)

    def predict(self, image: Tensor) -> str:
        """Alias for :meth:`classify` — used by :class:`DecodePipeline`.

        Args:
            image: Image tensor — (3, H, W) or (1, 3, H, W).

        Returns:
            Predicted model name string.
        """
        return self.classify(image)
