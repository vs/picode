# Detection Improvements: Mobile-Optimized Watermark Detection

## Executive Summary

The current detection system uses an **exhaustive sliding window approach** that reuses the decoder to measure confidence across image regions. While accurate, this approach is too slow for mobile deployment (100-500ms latency). This document proposes a **dedicated FastDetector model** optimized for mobile real-time detection (< 50ms) while preserving the existing SlowDetector for high-accuracy server-side analysis.

**Key improvements:**
- New lightweight FastDetector with MobileNetV3-Small backbone (~1.2M params)
- Quadrilateral output (4 corners) instead of bounding box for perspective-aware detection
- Rectification pipeline to transform detected regions for decoder input
- Content-focused training with hard negative mining
- Dual-detector architecture: FastDetector (mobile) + SlowDetector (server)

---

## 1. Current System Limitations

### 1.1 Existing Detection Approach

The current `Detector` class uses sliding windows with decoder confidence scoring:

```python
# Current approach (simplified)
for window in generate_windows(image, scales=[0.25, 0.35, 0.5, 0.65, 0.75]):
    crop = extract_window(image, window)
    resized = resize_to_400x400(crop)
    logits = decoder(resized)
    confidence = mean(|sigmoid(logits) - 0.5|)
    if confidence > threshold:
        detections.append(window)
```

**Problems for mobile:**

| Aspect | Current | Mobile Requirement | Gap |
|--------|---------|-------------------|-----|
| **Architecture** | Decoder + sliding window | Single-pass detector | Complete redesign |
| **Latency** | 100-500ms (exhaustive) | < 50ms real-time | 10-20x speedup needed |
| **Output** | Confidence + bbox | Binary + quadrilateral | Need corner regression |
| **Forward passes** | 50-200 per image | 1 per image | Dedicated model needed |

### 1.2 Mobile Use Cases

| Use Case | Latency Target | Description |
|----------|---------------|-------------|
| **Real-time viewfinder** | < 50ms | Highlight watermarks as user points camera |
| **On-demand check** | < 100ms | User taps to check if current photo is watermarked |

### 1.3 Perspective Problem

When watermarked images are captured by camera (photo of screen, printed image), the watermark region becomes a **quadrilateral**, not an axis-aligned rectangle:

```
Original watermark:          Captured photo:
+----------------+           +------------------+
|                |           |    /"""""""""\   |
|   WATERMARK    |    ->     |   / WATERMARK \  |
|                |           |  /              \ |
+----------------+           | /______________\ |
                             +------------------+
```

The decoder expects a **rectified 400x400 image**. A simple bounding box crop won't work - we need quadrilateral corners and perspective correction.

---

## 2. Dual-Detector Architecture

We maintain two detection systems for different use cases:

| Detector | Use Case | Latency | Accuracy | Deployment |
|----------|----------|---------|----------|------------|
| **SlowDetector** (existing) | Server analysis, high-accuracy scans | 100-500ms | Highest (exhaustive) | Server only |
| **FastDetector** (new) | Real-time viewfinder, on-demand mobile | < 50ms | Good (single-pass) | Mobile + Server |

**Why keep both:**
- **SlowDetector**: Thorough exhaustive search, no training needed, guaranteed coverage
- **FastDetector**: Trained model, single forward pass, mobile-optimized

**Detection result format (shared):**

```python
@dataclass
class Detection:
    corners: Quadrilateral          # 4 corner points (TL, TR, BR, BL)
    confidence: float               # [0, 1]
    detector_type: Literal["slow", "fast"]
    message_bits: Tensor | None     # Only SlowDetector decodes
    message_probs: Tensor | None    # Only SlowDetector provides
```

---

## 3. FastDetector Architecture

### 3.1 Overview

Single-pass CNN outputting binary classification + quadrilateral corners:

```
Input Image (320x320x3)
        |
        v
+---------------------+
|  MobileNetV3-Small  |  <- Backbone (feature extraction)
|  (BatchNorm + ReLU6)|
+---------------------+
        |
        v
   Feature Map (10x10x576)
        |
        +----------------+----------------+
        v                v                v
+---------------+ +---------------+ +---------------+
| Classification| | Corner Regress| |  Confidence   |
|     Head      | |     Head      | |     Head      |
+---------------+ +---------------+ +---------------+
        |                |                |
        v                v                v
   is_watermark      corners[8]      corner_conf
     (logit)        (normalized)       [0, 1]
```

### 3.2 Model Implementation

```python
class FastDetectorModel(nn.Module):
    """Single-pass watermark detector with quadrilateral output."""

    def __init__(self, input_size: int = 320):
        super().__init__()
        self.input_size = input_size

        # MobileNetV3-Small backbone (mobile-safe ops)
        # - BatchNorm (fuses with conv) - NO GroupNorm
        # - ReLU6/HardSwish (hardware accelerated) - NO LeakyReLU
        backbone = torchvision.models.mobilenet_v3_small(weights=None)
        self.features = backbone.features  # Output: 576 channels

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classification head: is there a watermark?
        self.cls_head = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # Corner regression head: 4 corners x 2 coords = 8 values
        self.corner_head = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 8),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

        # Corner confidence head
        self.conf_head = nn.Sequential(
            nn.Linear(576, 64),
            nn.ReLU6(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        features = self.features(x)              # (B, 576, 10, 10)
        pooled = self.pool(features).flatten(1)  # (B, 576)

        return {
            "is_watermark": self.cls_head(pooled),      # (B, 1) logit
            "corners": self.corner_head(pooled),         # (B, 8) normalized
            "corner_confidence": self.conf_head(pooled), # (B, 1)
        }
```

### 3.3 Model Statistics

| Component | Parameters | Notes |
|-----------|------------|-------|
| MobileNetV3-Small backbone | ~930K | ImageNet pretrained available |
| Classification head | ~74K | 576 -> 128 -> 1 |
| Corner head | ~150K | 576 -> 256 -> 8 |
| Confidence head | ~37K | 576 -> 64 -> 1 |
| **Total** | **~1.2M** | Acceptable for detection model |

### 3.4 Mobile Compatibility

| Constraint | Requirement | FastDetector Status |
|------------|-------------|---------------------|
| **No GroupNorm** | BatchNorm only | OK - MobileNetV3 uses BatchNorm |
| **No LeakyReLU** | ReLU6/HardSwish | OK - Uses ReLU6 |
| **Inference < 50ms** | Single pass | OK - ~40ms estimated |
| **Model size** | < 5MB INT8 | OK - ~1.2M params -> ~1.5MB INT8 |

---

## 4. Rectification Pipeline

### 4.1 Corner Representation

FastDetector outputs 8 normalized values representing 4 corners in clockwise order:

```python
corners = [x1, y1, x2, y2, x3, y3, x4, y4]
#          TL      TR      BR      BL

# Example: watermark with perspective distortion
corners = [0.4, 0.2,   # Top-left
           0.9, 0.25,  # Top-right (lower due to perspective)
           0.85, 0.8,  # Bottom-right
           0.35, 0.75] # Bottom-left
```

### 4.2 Rectifier Implementation

```python
class Rectifier:
    """Extracts and rectifies watermarked region from detected corners."""

    def __init__(self, output_size: int = 400):
        self.output_size = output_size
        self.dst_corners = np.array([
            [0, 0],                              # TL
            [output_size - 1, 0],                # TR
            [output_size - 1, output_size - 1], # BR
            [0, output_size - 1],                # BL
        ], dtype=np.float32)

    def rectify(self, image: Tensor, corners: Quadrilateral) -> Tensor:
        """
        Extract and rectify region defined by corners.

        Args:
            image: Source image (C, H, W)
            corners: Quadrilateral with 4 corner points

        Returns:
            Rectified (3, 400, 400) tensor ready for decoder
        """
        img_np = to_numpy_hwc(image)
        src_corners = corners.to_numpy()

        # Compute perspective transform
        M = cv2.getPerspectiveTransform(src_corners, self.dst_corners)

        # Apply warp
        rectified = cv2.warpPerspective(
            img_np, M,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return to_tensor_chw(rectified)
```

### 4.3 Mobile-Native Rectification

On mobile, use platform APIs instead of OpenCV:

| Platform | API | Notes |
|----------|-----|-------|
| **iOS** | `CIPerspectiveCorrection` | Core Image filter, GPU-accelerated |
| **iOS** | `vImagePerspectiveWarp` | Accelerate framework, CPU optimized |
| **Android** | `Matrix.setPolyToPoly()` | Canvas/Bitmap transform |

**iOS Swift example:**
```swift
func rectifyRegion(image: CIImage, corners: [CGPoint]) -> CIImage? {
    let filter = CIFilter(name: "CIPerspectiveCorrection")!
    filter.setValue(image, forKey: kCIInputImageKey)
    filter.setValue(CIVector(cgPoint: corners[0]), forKey: "inputTopLeft")
    filter.setValue(CIVector(cgPoint: corners[1]), forKey: "inputTopRight")
    filter.setValue(CIVector(cgPoint: corners[2]), forKey: "inputBottomRight")
    filter.setValue(CIVector(cgPoint: corners[3]), forKey: "inputBottomLeft")
    return filter.outputImage
}
```

### 4.4 End-to-End Pipeline

```
+-------------+    +--------------+    +-----------+    +---------+
| Camera/Photo| -> | FastDetector | -> | Rectifier | -> | Decoder |
|  (H x W)    |    | (320 x 320)  |    | (corners->|    |(400x400)|
+-------------+    +--------------+    |  400x400) |    +---------+
                          |            +-----------+         |
                          v                                  v
                   is_watermark?                      message_bits
                   corners[8]
                   confidence
```

### 4.5 Latency Budget (Mobile)

| Stage | Target | Notes |
|-------|--------|-------|
| Resize to 320x320 | < 2ms | GPU texture sampling |
| FastDetector inference | < 40ms | Core ML / TFLite |
| Rectification | < 5ms | CIPerspectiveCorrection |
| Decoder inference | < 50ms | Only if watermark detected |
| **Total (detection only)** | **< 47ms** | Meets 50ms target |
| **Total (with decode)** | **< 97ms** | Meets 100ms target |

---

## 5. Training Strategy

### 5.1 Content-Focused Detection

**Key insight:** The detector should learn **watermark encoding artifacts** (wave patterns, color imbalance, smooth-region perturbations), NOT scene context (screens, bezels, backgrounds).

We train with:
1. **Positives**: Watermarked images with perspective + photometric augmentations
2. **Negatives**: Diverse non-watermarked images including hard negatives

The watermarked image fills the frame (MVP: cooperative framing assumed).

### 5.2 Synthetic Data Generation

```python
class DetectionDataset(Dataset):
    """Synthetic dataset for FastDetector training."""

    def __init__(
        self,
        image_dir: str,
        encoder: nn.Module,
        negative_dirs: list[str],  # ImageNet, COCO, etc.
        num_bits: int = 100,
        positive_ratio: float = 0.5,
    ):
        self.encoder = encoder
        self.positive_ratio = positive_ratio
        self.positive_images = list(Path(image_dir).glob("*.jpg"))
        self.negative_images = self._collect_negatives(negative_dirs)

    def _generate_positive(self, image: Tensor) -> dict:
        """Watermarked image with perspective transform."""
        # 1. Encode watermark
        message = torch.randint(0, 2, (1, self.num_bits)).float()
        watermarked = self.encoder(image, message)

        # 2. Apply random perspective transform
        corners, warped = self._apply_perspective(watermarked)

        # 3. Apply photometric augmentations
        augmented = self._augment(warped)

        return {
            "image": F.interpolate(augmented, size=(320, 320)),
            "is_watermark": torch.tensor(1.0),
            "corners": corners.flatten(),  # (8,) normalized
            "has_corners": torch.tensor(1.0),
        }

    def _generate_negative(self, image: Tensor) -> dict:
        """Non-watermarked image (same augmentations)."""
        augmented = self._augment(image)
        return {
            "image": F.interpolate(augmented, size=(320, 320)),
            "is_watermark": torch.tensor(0.0),
            "corners": torch.zeros(8),
            "has_corners": torch.tensor(0.0),
        }
```

### 5.3 Hard Negative Mining

To ensure the model learns watermark features (not shortcuts), negatives must include images that could fool a naive detector:

| Negative Type | Ratio | Rationale |
|---------------|-------|-----------|
| ImageNet (diverse) | 30% | General natural images |
| COCO (complex scenes) | 20% | Multi-object scenes |
| Heavy JPEG compression | 15% | Has block artifacts like encoding |
| Upscaled/resized images | 10% | Has interpolation artifacts |
| Filtered images | 10% | Instagram-style processing |
| Screenshots | 10% | Sharp edges, UI elements |
| Scanned documents | 5% | Has noise patterns |

**What the model should learn:**

| Feature | In encoded images | In hard negatives | Discriminative? |
|---------|-------------------|-------------------|-----------------|
| JPEG artifacts | Sometimes | Yes | No |
| Sharp edges | Yes | Yes (screenshots) | No |
| **Diagonal wave patterns** | Yes | No | YES |
| **Color channel imbalance** | Yes | No | YES |
| **Smooth-region perturbations** | Yes | No | YES |
| **Specific frequency signature** | Yes | No | YES |

### 5.4 Augmentation Pipeline

```python
class DetectionAugmentation:
    """Augmentations simulating real-world capture conditions."""

    # Photometric (don't affect corners)
    photometric = [
        RandomBrightness(0.2),
        RandomContrast(0.2),
        RandomSaturation(0.2),
        GaussianNoise(std=0.02),
        GaussianBlur(kernel_range=(3, 7), p=0.3),
        JPEGCompression(quality_range=(30, 95), p=0.5),
    ]

    # Capture simulation
    capture_sim = [
        MoirePattern(p=0.2),       # Screen capture artifact
        MotionBlur(p=0.1),         # Camera shake
        VignetteEffect(p=0.2),     # Camera vignette
    ]

    # Geometric (require corner adjustment)
    geometric = [
        RandomPerspective(strength=(0.0, 0.15)),
        RandomRotation(degrees=(-15, 15)),
    ]
```

### 5.5 Loss Functions

```python
class DetectionLoss(nn.Module):
    """Combined classification + corner regression loss."""

    def __init__(
        self,
        cls_weight: float = 1.0,
        corner_weight: float = 5.0,
        conf_weight: float = 0.5,
    ):
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.corner_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred: dict, target: dict) -> dict[str, Tensor]:
        # Classification loss (all samples)
        cls_loss = self.cls_loss(
            pred["is_watermark"].squeeze(),
            target["is_watermark"],
        )

        # Corner loss (positive samples only)
        mask = target["has_corners"].bool()
        if mask.any():
            corner_loss = self.corner_loss(
                pred["corners"][mask],
                target["corners"][mask],
            ).mean()
        else:
            corner_loss = torch.tensor(0.0)

        total = cls_weight * cls_loss + corner_weight * corner_loss

        return {"total": total, "cls": cls_loss, "corner": corner_loss}
```

### 5.6 Training Configuration

```yaml
# configs/detection_training.yaml
experiment_name: fast_detector_v1

model:
  type: fast_detector
  backbone: mobilenetv3_small
  input_size: 320
  pretrained_backbone: true

data:
  positive_dir: ./data/train
  negative_dirs:
    - ./data/imagenet_subset
    - ./data/coco_subset
  batch_size: 32
  positive_ratio: 0.5
  num_workers: 4

training:
  num_steps: 100000
  optimizer: adamw
  lr: 0.001
  weight_decay: 0.01
  scheduler: cosine
  warmup_steps: 1000

loss:
  cls_weight: 1.0
  corner_weight: 5.0
  conf_weight: 0.5

augmentation:
  jpeg_quality: [30, 95]
  noise_std: 0.02
  blur_prob: 0.3
  perspective_strength: [0.0, 0.15]
  rotation_range: [-15, 15]

checkpoint:
  dir: checkpoints/detection
  save_every_steps: 5000
```

---

## 6. Evaluation Metrics

### 6.1 Classification Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Precision | TP / (TP + FP) | > 95% |
| Recall | TP / (TP + FN) | > 90% |
| F1 Score | 2 * P * R / (P + R) | > 92% |
| ROC-AUC | Area under ROC curve | > 0.97 |
| False Positive Rate | FP / (FP + TN) | < 5% |

### 6.2 Localization Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Mean Corner Error | L2 distance (normalized coords) | < 0.03 |
| Mean IoU | Quadrilateral intersection/union | > 0.85 |
| IoU@0.75 | % samples with IoU > 0.75 | > 90% |
| IoU@0.90 | % samples with IoU > 0.90 | > 70% |
| Homography Error | Reprojection error in pixels | < 10px |

### 6.3 End-to-End Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Decode Success Rate | Detection + rectification + >90% bits correct | > 85% |
| Perfect Decode Rate | 100% bits correct after rectification | > 70% |
| False Decode Rate | Confident decode on non-watermarked | < 5% |

### 6.4 Robustness Evaluation

Sweep across distortion types and strengths:

| Distortion | Parameters | Recall Target |
|------------|------------|---------------|
| JPEG | Q95, Q80, Q60, Q40, Q20 | > 80% at Q40 |
| Gaussian Blur | sigma 0.5, 1.0, 2.0, 3.0 | > 80% at 2.0 |
| Gaussian Noise | std 0.01, 0.02, 0.05, 0.1 | > 80% at 0.05 |
| Perspective | strength 0.02, 0.05, 0.1, 0.15 | > 85% at 0.1 |
| Brightness | delta -0.3, -0.15, 0, +0.15, +0.3 | > 90% |
| Resize | scale 1.0, 0.75, 0.5, 0.35 | > 80% at 0.5 |

### 6.5 Mobile Metrics

| Metric | Target (iPhone 12+) | Target (Android Flagship) |
|--------|---------------------|---------------------------|
| Model size (INT8) | < 2 MB | < 2 MB |
| Inference latency | < 40ms | < 50ms |
| Memory usage | < 50 MB | < 50 MB |
| Pipeline latency (detect+rectify) | < 50ms | < 60ms |

---

## 7. API Design

### 7.1 Module Structure

```
picode/detection/
+-- __init__.py              # Public exports
+-- types.py                 # Point, Quadrilateral, Detection
+-- confidence.py            # Existing - confidence scoring
+-- window.py                # Existing - sliding window
+-- detector.py              # SlowDetector (renamed from Detector)
+-- fast_detector.py         # NEW - FastDetector
+-- rectifier.py             # NEW - perspective rectification
+-- pipeline.py              # NEW - integrated pipeline
+-- training/                # NEW - training infrastructure
|   +-- dataset.py
|   +-- augmentation.py
|   +-- loss.py
|   +-- trainer.py
+-- export/                  # NEW - mobile export
    +-- coreml.py
    +-- tflite.py
```

### 7.2 Core Types

```python
@dataclass
class Quadrilateral:
    """Four corners in clockwise order: TL, TR, BR, BL."""
    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point

    def to_tensor(self) -> Tensor:
        """Returns (8,) tensor."""
        ...

    def to_numpy(self) -> np.ndarray:
        """Returns (4, 2) array for OpenCV."""
        ...

@dataclass
class Detection:
    """Detection result - works with both SlowDetector and FastDetector."""
    corners: Quadrilateral
    confidence: float
    detector_type: Literal["slow", "fast"]
    message_bits: Tensor | None = None   # SlowDetector only
    message_probs: Tensor | None = None  # SlowDetector only
```

### 7.3 FastDetector API

```python
class FastDetector:
    """Fast single-pass watermark detection for mobile."""

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "FastDetector":
        """Load from saved checkpoint."""
        ...

    def detect(self, image: Tensor | str) -> Detection | None:
        """Detect watermark, return Detection or None."""
        ...

    def detect_batch(self, images: list) -> list[Detection | None]:
        """Batch detection for efficiency."""
        ...
```

### 7.4 Pipeline API

```python
class DetectionPipeline:
    """Complete detect -> rectify -> decode pipeline."""

    @classmethod
    def from_checkpoints(
        cls,
        detector_path: str,
        decoder_path: str,
        device: str = "cpu",
    ) -> "DetectionPipeline":
        ...

    def detect_and_decode(self, image: Tensor | str) -> Result | None:
        """Full pipeline: detect, rectify, decode."""
        ...
```

### 7.5 CLI Usage

```bash
# Fast detection only (no decode)
detect photo.jpg -d fast -f fast_detector.pt --no-decode

# Fast detection + decode
detect photo.jpg -d fast -f fast_detector.pt -c decoder.pt

# Slow detection (existing behavior)
detect photo.jpg -d slow -c decoder.pt

# Save rectified region for inspection
detect photo.jpg -d fast -f fast_detector.pt --save-rectified rect.png
```

### 7.6 Public Exports

```python
# picode/detection/__init__.py
__all__ = [
    # Types
    "Point", "Quadrilateral", "Detection",
    # Slow detector (existing)
    "SlowDetector", "compute_confidence", "Window", "WindowGenerator",
    # Fast detector (new)
    "FastDetector", "FastDetectorModel",
    # Utilities
    "Rectifier", "DetectionPipeline",
]

# Backward compatibility
Detector = SlowDetector
```

---

## 8. Implementation Roadmap

### Phase 1: Core FastDetector (2-3 training runs)

1. Implement `FastDetectorModel` with MobileNetV3-Small backbone
2. Implement `DetectionDataset` with synthetic positive generation
3. Implement `DetectionLoss` (classification + corner regression)
4. Train initial model, validate on held-out set
5. Implement `Rectifier` with OpenCV backend

### Phase 2: Hard Negative Mining (2-3 training runs)

6. Collect hard negative datasets (JPEG artifacts, resized, filtered, etc.)
7. Implement `HardNegativeStrategy` sampling
8. Retrain with balanced hard negatives
9. Evaluate robustness across distortion types

### Phase 3: API & Integration (1-2 weeks implementation)

10. Implement `FastDetector` high-level API
11. Implement `DetectionPipeline` (detect + rectify + decode)
12. Update CLI with `--detector` flag
13. Add comprehensive tests
14. Update `SlowDetector` to output `Quadrilateral` for consistency

### Phase 4: Mobile Export & Validation

15. Export to Core ML (iOS)
16. Export to TFLite (Android)
17. Benchmark inference latency on target devices
18. Validate end-to-end accuracy on mobile
19. Implement native rectification (CIPerspectiveCorrection / Matrix)

### Phase 5: iOS Integration

20. Add `FastDetector` to picode-ios package
21. Implement real-time viewfinder detection
22. Implement on-demand photo check
23. User testing and iteration

---

## 9. Expected Outcomes

### 9.1 Performance Targets

| Metric | Current (SlowDetector) | Target (FastDetector) |
|--------|------------------------|----------------------|
| Latency (mobile) | 100-500ms | < 50ms |
| Model size | N/A (uses decoder) | < 2MB INT8 |
| Precision | ~95% | > 95% |
| Recall | ~98% (exhaustive) | > 90% |
| IoU (localization) | N/A (bbox only) | > 0.85 |

### 9.2 Mobile Deployment Targets

| Metric | iPhone 12+ | Android Flagship |
|--------|------------|------------------|
| Detection latency | < 40ms | < 50ms |
| Full pipeline latency | < 100ms | < 120ms |
| Memory usage | < 50MB | < 50MB |
| Model size | < 2MB | < 2MB |

### 9.3 User Experience Goals

- **Real-time viewfinder**: Smooth 20+ FPS detection highlighting
- **On-demand check**: Near-instant feedback (< 100ms)
- **Decode accuracy**: > 85% successful message recovery after rectification

---

## 10. References

### Related Project Documents
- [Model Improvements](./model_improvements.md) - Encoder/decoder quality improvements
- [Mobile Deployment Guide](./mobile_deployment.md) - Mobile constraints and optimization

### Techniques
- MobileNetV3: Howard et al., "Searching for MobileNetV3", ICCV 2019
- Perspective correction: Standard homography estimation via DLT
- Hard negative mining: Shrivastava et al., "Training Region-based Object Detectors with Online Hard Example Mining", CVPR 2016

---

*Document created: 2026-03-04*
*Status: Proposed improvements pending implementation*
