# Mobile Deployment Guide for Picode

> **Note:** All file paths in this document are relative to `picode-model/` unless otherwise specified.

This document analyzes requirements for running steganography models on mobile devices and proposes architectural changes to optimize the Picode model for mobile inference.

## Table of Contents

1. [Mobile ML Requirements](#mobile-ml-requirements)
2. [Current Model Analysis](#current-model-analysis)
3. [Proposed Mobile Architecture](#proposed-mobile-architecture)
4. [Deployment Frameworks](#deployment-frameworks)
5. [Optimization Techniques](#optimization-techniques)
6. [Implementation Roadmap](#implementation-roadmap)

---

## Mobile ML Requirements

### Hardware Constraints

| Constraint | iOS (iPhone 12+) | Android (Flagship) | Android (Mid-range) |
|------------|------------------|-------------------|---------------------|
| RAM | 4-6 GB (shared) | 8-12 GB (shared) | 4-6 GB (shared) |
| Neural Engine | A14+ (16 cores) | Qualcomm Hexagon / Mali | Limited NPU |
| GPU Memory | Shared | Shared | Shared |
| Thermal Budget | Low | Low | Very Low |

### Model Size Targets

| Category | Target | Rationale |
|----------|--------|-----------|
| **Decoder (primary)** | < 5 MB | Decoder is the main mobile use case (reading encoded images) |
| **Encoder (optional)** | < 20 MB | Encoding can be done server-side in most cases |
| **Total app impact** | < 50 MB | iOS 200MB cellular limit; user experience |

### Inference Speed Targets

| Operation | Target | Device |
|-----------|--------|--------|
| Decode 400x400 | < 100ms | iPhone 12+ |
| Decode 400x400 | < 150ms | Flagship Android |
| Decode 1080p | < 300ms | iPhone 12+ |

### Operations Compatibility Matrix

| Operation | Core ML | TF Lite | ONNX Mobile | Notes |
|-----------|---------|---------|-------------|-------|
| Conv2d | Native | Native | Native | Well optimized |
| DepthwiseSeparable | Native | Native | Native | Highly efficient |
| BatchNorm | Native | Native | Native | Fused with conv |
| **GroupNorm** | Custom | Limited | Limited | **Poor support** |
| ReLU/ReLU6 | Native | Native | Native | Hardware accelerated |
| **LeakyReLU** | Slow | Slow | Slow | **No hardware accel** |
| Sigmoid | Native | Native | Native | Good support |
| AdaptiveAvgPool | Native | Native | Native | Good support |
| **affine_grid** | Limited | Limited | Limited | **STN problematic** |
| **grid_sample** | Limited | Limited | Limited | **STN problematic** |

---

## Current Model Analysis

### Picode Encoder (Current)

```
Parameters: ~2.1M
Operations:
- 17x Conv2d with GroupNorm + LeakyReLU
- 1x Linear (100 -> 7500)
- U-Net skip connections
- F.interpolate (nearest)

Mobile Issues:
✗ GroupNorm: Not hardware-accelerated on NPUs
✗ LeakyReLU: No dedicated silicon, runs on CPU
✗ Large channel counts (256 at bottleneck)
✗ 400x400 fixed input assumption
```

### Picode Decoder (Current)

```
Parameters: ~1.8M
Operations:
- STN: 3x Conv2d + 1x Linear + affine_grid + grid_sample
- 4x Conv2d + 3x ResBlock (each with GroupNorm + LeakyReLU)
- AdaptiveAvgPool2d
- 1x Linear (256 -> 100)

Mobile Issues:
✗ STN (affine_grid/grid_sample): Poor mobile support, often CPU-only
✗ GroupNorm: Not hardware-accelerated
✗ LeakyReLU: CPU fallback
✗ ResBlocks with GroupNorm compound the issue
```

### StegaStamp Model (Current)

```
Encoder: ~1.9M parameters
Decoder: ~1.6M parameters

Better for mobile because:
✓ No normalization layers (simpler)
✓ ReLU instead of LeakyReLU
✗ Still has STN in decoder
✗ No depthwise separable convolutions
```

### Parameter Counts

| Model | Encoder | Decoder | Total |
|-------|---------|---------|-------|
| StegaStamp | 1.9M | 1.6M | 3.5M |
| Picode | 2.1M | 1.8M | 3.9M |
| **Target Mobile** | N/A | **< 500K** | < 500K |

---

## Proposed Mobile Architecture

### Design Principles

1. **Decoder-only deployment**: Mobile devices read watermarks; encoding happens server-side
2. **Depthwise separable convolutions**: 8-9x fewer parameters than standard conv
3. **No normalization or BatchNorm only**: Fuses into preceding conv at export
4. **ReLU6 activation**: Hardware-accelerated, bounded output
5. **No STN**: Remove or replace with simpler geometric handling
6. **Inverted residual blocks**: MobileNetV2-style efficiency

### MobileDecoder Architecture

```python
class MobileDecoder(nn.Module):
    """Mobile-optimized decoder for watermark extraction.

    Design choices for mobile:
    - Depthwise separable convolutions (8-9x fewer params)
    - BatchNorm (fuses with conv at export time)
    - ReLU6 (hardware accelerated, bounded)
    - No STN (handled by preprocessing or removed)
    - Inverted residual blocks (MobileNetV2 style)

    Target: < 500K parameters, < 2MB model size
    """

    def __init__(self, num_bits: int = 100):
        super().__init__()

        # Stem: standard conv to expand channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        # Inverted residual blocks (MobileNetV2 style)
        # expand_ratio, out_channels, num_blocks, stride
        self.blocks = nn.Sequential(
            InvertedResidual(16, 24, stride=2, expand_ratio=6),   # 100x100
            InvertedResidual(24, 24, stride=1, expand_ratio=6),
            InvertedResidual(24, 32, stride=2, expand_ratio=6),   # 50x50
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 64, stride=2, expand_ratio=6),   # 25x25
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 96, stride=2, expand_ratio=6),   # 13x13
        )

        # Head: pool and classify
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, num_bits),
        )

    def forward(self, image: Tensor) -> Tensor:
        x = image - 0.5  # Normalize
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block.

    Structure: 1x1 expand -> 3x3 depthwise -> 1x1 project

    Why inverted? Standard residuals compress then expand.
    Inverted residuals expand then compress - more efficient
    for mobile because depthwise conv works on expanded features.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_residual = stride == 1 and in_ch == out_ch

        layers = []
        # Expand (if expand_ratio > 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1,
                      groups=hidden, bias=False),  # groups=hidden makes it depthwise
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
        ])

        # Project (linear, no activation)
        layers.extend([
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)
```

### Parameter Comparison

| Component | Current Picode | Mobile Decoder | Reduction |
|-----------|---------------|----------------|-----------|
| Stem | ~5K | ~500 | 10x |
| Body | ~1.5M | ~300K | 5x |
| Head | ~26K | ~10K | 2.6x |
| **Total** | **~1.8M** | **~310K** | **5.8x** |

### Handling STN Removal

The Spatial Transformer Network handles geometric distortions (rotation, perspective). Options for mobile:

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Remove entirely** | Simplest, fastest | Less robust to perspective | Use if images are already aligned |
| **Server preprocessing** | Best accuracy | Requires network | Use for high-value decoding |
| **Train without STN** | No mobile overhead | Model learns robustness | Recommended for v1 |
| **Lightweight alignment** | Compromise | Still some overhead | Future enhancement |

**Recommendation**: Train a mobile decoder without STN but with aggressive perspective distortion augmentation. The model learns inherent robustness rather than explicit correction.

---

## Deployment Frameworks

### iOS: Core ML

```bash
# Export workflow
pip install coremltools

# Convert PyTorch -> Core ML
python -c "
import torch
import coremltools as ct
from picode.models.mobile import MobileDecoder

model = MobileDecoder(num_bits=100)
model.eval()

# Trace with example input
example = torch.randn(1, 3, 400, 400)
traced = torch.jit.trace(model, example)

# Convert to Core ML
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name='image', shape=(1, 3, 400, 400))],
    minimum_deployment_target=ct.target.iOS15,
)
mlmodel.save('PicodeDecoder.mlpackage')
"
```

**Core ML Optimizations:**
- BatchNorm automatically fuses with Conv
- Runs on Neural Engine (16 TOPS on A14+)
- Supports FP16 inference automatically

### Android: TensorFlow Lite

```bash
# Export workflow
pip install tensorflow onnx onnx-tf

# PyTorch -> ONNX -> TensorFlow -> TFLite
python -c "
import torch
from picode.models.mobile import MobileDecoder

model = MobileDecoder(num_bits=100)
model.eval()

# Export to ONNX
torch.onnx.export(
    model,
    torch.randn(1, 3, 400, 400),
    'decoder.onnx',
    input_names=['image'],
    output_names=['logits'],
    dynamic_axes={'image': {0: 'batch'}},
    opset_version=13,
)
"

# Then use onnx-tf and tflite_convert
```

**TFLite Optimizations:**
- GPU delegate for Adreno/Mali
- NNAPI delegate for Hexagon DSP
- INT8 quantization for 4x speedup

### Cross-Platform: ONNX Runtime Mobile

```bash
# Simplest cross-platform option
pip install onnxruntime

# Export directly
python -c "
import torch
from picode.models.mobile import MobileDecoder

model = MobileDecoder(num_bits=100)
model.eval()

torch.onnx.export(
    model,
    torch.randn(1, 3, 400, 400),
    'decoder.onnx',
    opset_version=13,
)
"
```

---

## Optimization Techniques

### 1. Quantization

| Type | Size Reduction | Speed Improvement | Accuracy Impact |
|------|---------------|-------------------|-----------------|
| FP32 (baseline) | 1x | 1x | 0% |
| FP16 | 2x | 1.5-2x | < 0.1% |
| INT8 dynamic | 4x | 2-3x | 0.5-1% |
| INT8 static | 4x | 3-4x | 1-2% |

**Recommended**: FP16 for iOS (automatic), INT8 dynamic for Android

```python
# PyTorch dynamic quantization
import torch.quantization

model_int8 = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)
```

### 2. Quantization-Aware Training (QAT)

For best INT8 accuracy, train with fake quantization:

```python
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert

class MobileDecoderQAT(MobileDecoder):
    def __init__(self, num_bits: int = 100):
        super().__init__(num_bits)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = super().forward(x)
        x = self.dequant(x)
        return x

# During training
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_prepared = prepare_qat(model)
# ... train ...
model_quantized = convert(model_prepared)
```

### 3. Pruning

Remove unimportant weights for smaller models:

```python
import torch.nn.utils.prune as prune

# Prune 30% of weights with lowest magnitude
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)
```

### 4. Knowledge Distillation

Train mobile model to match full model outputs:

```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7):
    """Combined distillation and hard label loss."""
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)

    hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

---

## Implementation Roadmap

### Phase 1: Mobile Decoder (Priority)

1. **Create `picode/models/mobile/` module**
   - `decoder.py`: MobileDecoder with inverted residuals
   - `blocks.py`: InvertedResidual, DepthwiseSeparableConv
   - `export.py`: ONNX/CoreML/TFLite export utilities

2. **Training without STN**
   - Modify training config to disable STN
   - Increase perspective distortion augmentation
   - Train mobile decoder with knowledge distillation from full decoder

3. **Validation**
   - Benchmark on target devices (iPhone, Android flagship)
   - Measure accuracy degradation vs full model
   - Target: < 2% bit accuracy drop

### Phase 2: Optimization

1. **Quantization-aware training**
   - Implement QAT wrapper
   - Train INT8-friendly weights
   - Validate accuracy post-quantization

2. **Model variants**
   - `MobileDecoder-Tiny`: < 200K params, fastest
   - `MobileDecoder-Base`: ~300K params, balanced
   - `MobileDecoder-Large`: ~500K params, most accurate

### Phase 3: SDK & Integration

1. **iOS SDK**
   - Swift wrapper for Core ML model
   - Camera integration for real-time decoding
   - Sample app

2. **Android SDK**
   - Kotlin wrapper for TFLite model
   - CameraX integration
   - Sample app

### File Structure

```
picode-model/
├── picode/
│   ├── models/
│   │   ├── mobile/
│   │   │   ├── __init__.py
│   │   │   ├── decoder.py      # MobileDecoder
│   │   │   ├── blocks.py       # InvertedResidual, DSConv
│   │   │   └── export.py       # ONNX/CoreML/TFLite export
│   │   └── ...
│   ├── training/
│   │   ├── mobile_trainer.py   # Training with distillation
│   │   └── ...
│   └── ...
└── ...
```

---

## Summary

| Aspect | Current | Mobile Target |
|--------|---------|---------------|
| Decoder params | 1.8M | < 500K |
| Model size | ~7 MB | < 2 MB (INT8) |
| Inference time | N/A | < 100ms (iPhone) |
| Normalization | GroupNorm | BatchNorm (fused) |
| Activation | LeakyReLU | ReLU6 |
| STN | Yes | No (train robustness) |
| Conv type | Standard | Depthwise separable |

**Key Insight**: The decoder is the critical mobile component. Encoding watermarks can happen server-side, but decoding must be fast and efficient on-device. Focus optimization efforts on the decoder path.

**Next Step**: Implement `picode/models/mobile/decoder.py` with the MobileDecoder architecture and validate accuracy with knowledge distillation training.
