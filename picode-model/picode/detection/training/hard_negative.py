# picode/detection/training/hard_negative.py
"""Hard negative mining for FastDetector training."""

from __future__ import annotations

import io
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class HardNegativeTransform:
    """Transforms to create hard negative samples.

    Hard negatives are images with visual features that could confuse
    a naive detector (JPEG artifacts, resize artifacts, filter effects)
    but don't contain actual watermarks.

    Transform types:
        - JPEG compression artifacts (block patterns)
        - Resize/interpolation artifacts
        - Filter effects (sharpen, blur, edge enhance)
        - Screenshot simulation (sharp edges)
    """

    def __init__(
        self,
        jpeg_quality_range: tuple[int, int] = (20, 70),
        resize_scale_range: tuple[float, float] = (0.3, 0.7),
    ) -> None:
        self.jpeg_quality_range = jpeg_quality_range
        self.resize_scale_range = resize_scale_range

        self.filter_effects = ["sharpen", "edge_enhance", "smooth", "detail"]

    def __call__(self, image: Tensor) -> Tensor:
        """Apply random hard negative transform.

        Args:
            image: (C, H, W) tensor in [0, 1]

        Returns:
            Transformed image (C, H, W) in [0, 1]
        """
        transform_type = random.choice(["jpeg", "resize", "filter", "combined"])

        if transform_type == "jpeg":
            quality = random.randint(*self.jpeg_quality_range)
            return self.apply_jpeg_artifacts(image, quality)
        elif transform_type == "resize":
            scale = random.uniform(*self.resize_scale_range)
            return self.apply_resize_artifacts(image, scale)
        elif transform_type == "filter":
            effect = random.choice(self.filter_effects)
            return self.apply_filter_effect(image, effect)
        else:  # combined
            # Apply multiple transforms
            quality = random.randint(*self.jpeg_quality_range)
            image = self.apply_jpeg_artifacts(image, quality)
            if random.random() < 0.5:
                effect = random.choice(self.filter_effects)
                image = self.apply_filter_effect(image, effect)
            return image

    def apply_jpeg_artifacts(self, image: Tensor, quality: int) -> Tensor:
        """Apply JPEG compression artifacts.

        Args:
            image: (C, H, W) tensor in [0, 1]
            quality: JPEG quality (1-100, lower = more artifacts)

        Returns:
            Image with JPEG artifacts
        """
        # Convert to PIL
        pil_image = self._tensor_to_pil(image)

        # Compress with JPEG
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)

        # Convert back to tensor
        return self._pil_to_tensor(compressed)

    def apply_resize_artifacts(self, image: Tensor, scale: float) -> Tensor:
        """Apply resize/interpolation artifacts.

        Args:
            image: (C, H, W) tensor in [0, 1]
            scale: Downscale factor (0-1)

        Returns:
            Image with resize artifacts
        """
        c, h, w = image.shape

        # Downscale
        small_h, small_w = int(h * scale), int(w * scale)
        small = F.interpolate(
            image.unsqueeze(0),
            size=(small_h, small_w),
            mode="bilinear",
            align_corners=False,
        )

        # Upscale back
        restored = F.interpolate(
            small,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return restored.clamp(0.0, 1.0)

    def apply_filter_effect(self, image: Tensor, effect: str) -> Tensor:
        """Apply filter effect (Instagram-style processing).

        Args:
            image: (C, H, W) tensor in [0, 1]
            effect: Filter type ("sharpen", "edge_enhance", "smooth", "detail")

        Returns:
            Filtered image
        """
        pil_image = self._tensor_to_pil(image)

        if effect == "sharpen":
            pil_image = pil_image.filter(ImageFilter.SHARPEN)
        elif effect == "edge_enhance":
            pil_image = pil_image.filter(ImageFilter.EDGE_ENHANCE)
        elif effect == "smooth":
            pil_image = pil_image.filter(ImageFilter.SMOOTH)
        elif effect == "detail":
            pil_image = pil_image.filter(ImageFilter.DETAIL)

        # Random contrast/brightness adjustment
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(random.uniform(0.8, 1.2))

        return self._pil_to_tensor(pil_image)

    def _tensor_to_pil(self, tensor: Tensor) -> Image.Image:
        """Convert (C, H, W) tensor to PIL Image."""
        array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        return Image.fromarray(array)

    def _pil_to_tensor(self, image: Image.Image) -> Tensor:
        """Convert PIL Image to (C, H, W) tensor."""
        import numpy as np
        array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


class HardNegativeDataset(Dataset[dict[str, Tensor]]):
    """Dataset of hard negative samples for detector training.

    Loads images from a directory and applies hard negative transforms
    to create samples that should be classified as non-watermarked.

    Args:
        image_dir: Directory containing negative images
        input_size: Output image size (default 320)
        transform_p: Probability of applying hard negative transform

    Example:
        >>> dataset = HardNegativeDataset("./negatives", input_size=320)
        >>> item = dataset[0]
        >>> item["is_watermark"]
        tensor(0.)
    """

    def __init__(
        self,
        image_dir: str | Path,
        input_size: int = 320,
        transform_p: float = 0.8,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.input_size = input_size
        self.transform_p = transform_p

        # Collect image paths
        self.image_paths = (
            list(self.image_dir.glob("*.jpg"))
            + list(self.image_dir.glob("*.jpeg"))
            + list(self.image_dir.glob("*.png"))
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        # Transforms
        self.hard_negative = HardNegativeTransform()
        self._load_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        # Load image
        img_path = self.image_paths[idx % len(self.image_paths)]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self._load_transform(image)

        # Apply hard negative transform
        if random.random() < self.transform_p:
            image_tensor = self.hard_negative(image_tensor)

        return {
            "image": image_tensor.clamp(0.0, 1.0),
            "is_watermark": torch.tensor(0.0),
            "corners": torch.zeros(8),
            "has_corners": torch.tensor(0.0),
        }
