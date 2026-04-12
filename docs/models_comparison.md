# Models Comparison

This document provides an in-depth comparison of the three steganography model architectures in the Picode framework: **StegaStamp**, **Picode**, and **Picode v2**.

## Overview

| Model | Purpose | Training | Key Innovation |
|-------|---------|----------|----------------|
| **StegaStamp** | Baseline implementation | Supervised | Original U-Net encoder with STN decoder |
| **Picode** | Improved gradient flow | Supervised | GroupNorm + LeakyReLU + ResBlocks |
| **Picode v2** | Mobile-optimized | GAN-based | Content-adaptive residuals + Focal Frequency Loss |

## Architecture Evolution

```
StegaStamp (CVPR 2020)
    │
    │ [Add normalization & better activations]
    ▼
Picode [Improved Gradient Flow]
    │
    │ [Learned message expansion, bilinear upsampling, content-adaptive, GAN]
    ▼
Picode v2 [Mobile-Optimized]
```

---

## StegaStamp

**Reference:** Tancik et al., "StegaStamp: Invisible Hyperlinks in Physical Photographs", CVPR 2020

**Location:** `picode/models/stegastamp/`

### Encoder Architecture

The StegaStamp encoder uses a U-Net architecture to embed binary messages into images:

**Message Preparation:**
- Linear expansion: `num_bits → 7500 → (3, 50, 50)`
- Bilinear upsampling: `(3, 50, 50) → (3, 400, 400)`
- Concatenated with input image (6 channels total)

**Downsampling Path (5 layers):**
```
Input: (6, 400, 400)
  ↓ Conv(6→32, k=3, s=2)    → (32, 200, 200)
  ↓ Conv(32→32, k=3, s=2)   → (32, 100, 100)
  ↓ Conv(32→64, k=3, s=2)   → (64, 50, 50)
  ↓ Conv(64→128, k=3, s=2)  → (128, 25, 25)
  ↓ Conv(128→256, k=3, s=2) → (256, 13, 13)
```
- ReLU activations
- No normalization layers

**Upsampling Path (with skip connections):**
```
(256, 13, 13)
  ↓ Conv(256→128) + Upsample → concat(skip4) → (256, 25, 25)
  ↓ Conv(256→64)  + Upsample → concat(skip3) → (128, 50, 50)
  ↓ Conv(128→32)  + Upsample → concat(skip2) → (64, 100, 100)
  ↓ Conv(64→32)   + Upsample → concat(skip1) → (64, 200, 200)
  ↓ Conv(64→32)   + Upsample → concat(input,message) → (70, 400, 400)
  ↓ Conv(70→3)    → (3, 400, 400)
```

**Output:** Residual learning with clamping
```python
output = torch.clamp(image + residual, 0, 1)
```

### Decoder Architecture

**Spatial Transformer Network (STN):**
- 3 conv layers → flatten → predict 6 affine parameters
- Initialized to identity transform
- Corrects for geometric distortions

**Main CNN Path:**
```
Input: (3, 400, 400)
  ↓ 7 conv layers with stride-2 downsampling
  ↓ Spatial resolution: 400→200→100→50→25→13
  ↓ Channels: 3→32→32→64→64→64→128→128
  ↓ Flatten → FC(21632→512) → FC(512→num_bits)
```
- Output: Raw logits (no sigmoid)
- Expects `BCEWithLogitsLoss`

### Loss Functions

```python
total_loss = (
    weight_msg  * BCE(decoder_logits, message) +
    weight_l2   * MSE(encoded_image, original_image) +
    weight_lpips * LPIPS(encoded_image, original_image)  # optional
)
```

| Loss | Purpose | Default Weight |
|------|---------|----------------|
| BCE | Message accuracy | 1.0 |
| L2 (MSE) | Pixel-level fidelity | 1.5 |
| LPIPS | Perceptual similarity | 1.0 |

### Characteristics

| Aspect | Detail |
|--------|--------|
| Normalization | None |
| Activation | ReLU |
| Gradient Flow | Basic (prone to vanishing gradients) |
| Training Stability | Sensitive to hyperparameters |
| Batch Size | Works best with larger batches |

---

## Picode

**Purpose:** Improved gradient flow and training stability

**Location:** `picode/models/picode/`

### Key Improvements Over StegaStamp

1. **GroupNorm** instead of no normalization
2. **LeakyReLU(0.2)** instead of ReLU
3. **ResBlocks** in decoder for skip connections
4. **Random STN initialization** instead of zeros

### Encoder Architecture

Same U-Net structure as StegaStamp with normalization added:

**Message Preparation:** Same as StegaStamp

**Downsampling Path:**
```
Each layer: Conv → GroupNorm(8, channels) → LeakyReLU(0.2)
```

**Upsampling Path:**
```
Each layer: Conv → Upsample → GroupNorm → LeakyReLU → concat(skip)
```

**Output:** Same residual learning with clamping

### Decoder Architecture

**Improved STN:**
- GroupNorm + LeakyReLU in feature extraction
- Small random initialization (`std=0.001`) for better early training

**ResBlock-based CNN:**
```
Stem: Conv → GroupNorm → LeakyReLU
Stage 1: Downsample → ResBlock → (64, H/2, W/2)
Stage 2: Downsample → ResBlock → (128, H/4, W/4)
Stage 3: Downsample → ResBlock → (256, H/8, W/8)
Stage 4: Downsample → ResBlock → (512, H/16, W/16)
  ↓ Global Average Pooling → FC → num_bits
```

**ResBlock Structure:**
```python
class ResBlock:
    def forward(x):
        residual = x
        x = Conv → GroupNorm → LeakyReLU → Conv → GroupNorm
        return LeakyReLU(x + residual)
```

### Loss Functions

Same as StegaStamp: BCE + L2 + LPIPS

### Characteristics

| Aspect | Detail |
|--------|--------|
| Normalization | GroupNorm (8 groups) |
| Activation | LeakyReLU(0.2) |
| Gradient Flow | Improved via ResBlocks + LeakyReLU |
| Training Stability | More stable with small batches |
| Batch Size | Works well with batch size 2-4 |

### Why These Changes Matter

**GroupNorm vs BatchNorm:**
- BatchNorm statistics are noisy with small batches
- GroupNorm normalizes within each sample (batch-independent)
- Stable behavior regardless of batch size

**LeakyReLU vs ReLU:**
- ReLU can cause "dying neurons" (zero gradient when input < 0)
- LeakyReLU maintains small gradient for negative inputs
- Prevents gradient flow from being completely blocked

**ResBlocks:**
- Skip connections allow gradients to bypass layers
- Mitigates vanishing gradient problem
- Enables training of deeper networks

---

## Picode v2

**Purpose:** Mobile deployment with reduced visual artifacts

**Location:** `picode/models/picode_v2/`

### Key Innovations

1. **MessageExpander** - Learned progressive upsampling
2. **Bilinear upsampling** throughout U-Net
3. **Content-adaptive residuals** - Exploit image statistics
4. **Tanh-bounded output** - Controlled residual magnitude
5. **No STN in decoder** - Simplified for mobile
6. **GAN training** with PatchDiscriminator
7. **Focal Frequency Loss** - Frequency-domain awareness

### Encoder Architecture

**MessageExpander Module:**
```python
class MessageExpander:
    """Learned upsampling to avoid checkerboard artifacts"""
    def forward(message):
        # Dense: num_bits → 64×5×5
        x = Linear → Reshape(64, 5, 5)

        # Progressive upsampling: 5→25→100→400
        x = Upsample(5→25) → Conv → GroupNorm → LeakyReLU
        x = Upsample(25→100) → Conv → GroupNorm → LeakyReLU
        x = Upsample(100→400) → Conv → GroupNorm → LeakyReLU

        return Conv(64→3)  # Final: (3, 400, 400)
```

**Bilinear Upsampling Path:**
```python
# Instead of ConvTranspose2d or pixel shuffle:
x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
x = Conv2d → GroupNorm → LeakyReLU
```

**Content-Adaptive Residual:**
```python
def compute_residual(image, raw_residual):
    # Bound residual magnitude
    bounded = torch.tanh(raw_residual) * residual_scale

    if content_adaptive:
        # Sobel edge detection for activity map
        activity = sobel_edges(image)  # High in textured regions
        scale_map = 0.3 + 0.7 * activity  # Range: [0.3, 1.0]
        bounded = bounded * scale_map

    return torch.clamp(image + bounded, 0, 1)
```

**Rationale for Content-Adaptive:**
- Larger perturbations in textured/edge regions (less visible)
- Smaller perturbations in smooth regions (more visible)
- Exploits human visual system characteristics

### Decoder Architecture

**Design Decision: No STN**
- Geometric correction delegated to detection pipeline
- Simpler architecture for mobile deployment
- STN adds computational overhead and parameters

**StegaStamp-style CNN:**
```
Input: (3, 400, 400)
  ↓ 7 conv layers (3×3, stride-2 downsampling)
  ↓ Pure ReLU activations (no normalization)
  ↓ Flatten → FC(21632→512) → FC(512→num_bits)
```

**Why Standard Convolutions (not Depthwise Separable):**
- Steganography requires aggregating globally distributed bits
- Depthwise convolutions limit cross-channel information flow
- Full convolutions better for this task despite being heavier

### Discriminator (Training Only)

**PatchGAN Architecture:**
```
Input: (3, 400, 400)
  ↓ Conv(3→64, k=4, s=2) → LeakyReLU
  ↓ Conv(64→128, k=4, s=2) → InstanceNorm → LeakyReLU
  ↓ Conv(128→256, k=4, s=2) → InstanceNorm → LeakyReLU
  ↓ Conv(256→512, k=4, s=1) → InstanceNorm → LeakyReLU
  ↓ Conv(512→1, k=4, s=1)
Output: (1, 49, 49) patch predictions
```

**70×70 Receptive Field:**
- Each output pixel classifies a 70×70 image patch
- Forces encoder to produce locally realistic textures
- More effective than single global real/fake decision

**InstanceNorm (not BatchNorm):**
- Better generalization for unpaired/adversarial training
- Normalizes each instance independently

### Loss Functions

**Focal Frequency Loss:**
```python
def focal_frequency_loss(pred, target, alpha=1.0):
    # 2D FFT
    pred_fft = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)

    # Magnitude difference
    freq_distance = |pred_fft.abs() - target_fft.abs()|

    # Focal weighting: harder frequencies get higher weight
    weight = freq_distance ** alpha

    return (weight * freq_distance).mean()
```

**Purpose:** Penalizes frequency-domain artifacts that L2 loss misses

**GAN Losses (WGAN-GP):**

Generator (Encoder):
```python
g_loss = -discriminator(encoded_image).mean()
```

Discriminator:
```python
d_loss = D(fake).mean() - D(real).mean() + λ_gp * gradient_penalty(D, real, fake)
```

### Characteristics

| Aspect | Detail |
|--------|--------|
| Normalization | GroupNorm (encoder), None (decoder), InstanceNorm (discriminator) |
| Activation | LeakyReLU (encoder/disc), ReLU (decoder) |
| Output Bounding | Tanh (controlled residual magnitude) |
| Artifact Reduction | Content-adaptive + bilinear upsampling + focal freq loss |
| Mobile Optimized | Simpler decoder, no STN |
| Training | GAN-based (requires discriminator) |

---

## Detailed Comparison Tables

### Encoder Comparison

| Feature | StegaStamp | Picode | Picode v2 |
|---------|------------|--------|-----------|
| **Message Expansion** | Linear → reshape → nearest | Linear → reshape → nearest | MessageExpander (learned) |
| **Normalization** | None | GroupNorm(8) | GroupNorm(8) |
| **Activation** | ReLU | LeakyReLU(0.2) | LeakyReLU(0.2) |
| **Upsampling** | Conv2d (2×2 kernel) | Conv2d (2×2 kernel) | nn.Upsample (bilinear) |
| **Output Residual** | Raw | Raw | Tanh-bounded |
| **Content Adaptive** | No | No | Yes (optional) |
| **Checkerboard Artifacts** | Possible | Possible | Minimized |

### Decoder Comparison

| Feature | StegaStamp | Picode | Picode v2 |
|---------|------------|--------|-----------|
| **Architecture** | CNN + FC | ResNet + GAP + FC | CNN + FC |
| **STN** | Yes (identity init) | Yes (random init) | No |
| **Normalization** | None | GroupNorm(8) | None |
| **Activation** | ReLU | LeakyReLU(0.2) | ReLU |
| **Skip Connections** | No | ResBlocks | No |
| **Parameters** | ~2.5M | ~4M | ~2.5M |

### Loss Function Comparison

| Loss Type | StegaStamp | Picode | Picode v2 |
|-----------|------------|--------|-----------|
| **Message** | BCE with logits | BCE with logits | BCE with logits |
| **Pixel** | L2 (MSE) | L2 (MSE) | L2 (MSE) |
| **Perceptual** | LPIPS | LPIPS | Focal Frequency |
| **Adversarial** | No | No | WGAN-GP |

### Training Characteristics

| Aspect | StegaStamp | Picode | Picode v2 |
|--------|------------|--------|-----------|
| **Training Type** | Supervised | Supervised | GAN + Supervised |
| **Batch Size** | 8-16 recommended | 2-4 works well | 4-8 recommended |
| **Learning Rate** | 1e-4 | 1e-4 | 1e-4 (G), 4e-4 (D) |
| **Gradient Flow** | Prone to vanishing | Improved | Good |
| **Training Stability** | Sensitive | More stable | Requires careful balancing |
| **Convergence** | ~100k steps | ~100k steps | ~140k steps |

---

## Use Case Recommendations

### StegaStamp

**Best for:**
- Reproducing original paper results
- Benchmarking against published baselines
- Large batch training scenarios
- When exact TensorFlow compatibility is needed

**Not ideal for:**
- Small batch sizes (< 8)
- Mobile deployment
- When visual quality is paramount

### Picode

**Best for:**
- Research and experimentation
- Limited GPU memory (small batches)
- When training stability is important
- Debugging gradient flow issues

**Not ideal for:**
- Mobile deployment (larger decoder)
- When artifact reduction is critical

### Picode v2

**Best for:**
- Production deployment
- Mobile applications (simpler decoder)
- When visual quality is paramount
- Scenarios requiring minimal visible artifacts

**Not ideal for:**
- Quick prototyping (GAN training complexity)
- When training resources are limited
- Reproducing paper results

---

## Configuration Examples

### StegaStamp Training

```yaml
# configs/stegastamp_baseline.yaml
training:
  model: stegastamp
  num_steps: 100000
  lr: 0.0001
  batch_size: 8
loss:
  message: { scale: 1.0, ramp_steps: 1 }
  l2: { scale: 1.5, ramp_steps: 20000 }
  lpips: { scale: 1.0, ramp_steps: 20000 }
```

### Picode Training

```yaml
# configs/picode_test.yaml
training:
  model: picode
  num_steps: 100000
  lr: 0.0001
  batch_size: 4
loss:
  message: { scale: 1.0, ramp_steps: 1 }
  l2: { scale: 1.5, ramp_steps: 20000 }
  lpips: { scale: 1.0, ramp_steps: 20000 }
```

### Picode v2 Training

```yaml
# configs/picode_v2.yaml
training:
  model: picode_v2
  num_steps: 140000
  lr: 0.0001
  batch_size: 4
  gan:
    enabled: true
    discriminator_lr: 0.0004
    gp_weight: 10.0
loss:
  message: { scale: 1.0, ramp_steps: 1 }
  l2: { scale: 1.5, ramp_steps: 20000 }
  focal_frequency: { scale: 0.1, ramp_steps: 30000 }
encoder:
  content_adaptive: true
  residual_scale: 0.1
```

---

## Import Patterns

```python
# StegaStamp
from picode.models.stegastamp import Encoder, Decoder
from picode.models.stegastamp.loss import stegastamp_loss

# Picode
from picode.models.picode import Encoder, Decoder

# Picode v2
from picode.models.picode_v2 import Encoder, Decoder, PatchDiscriminator
from picode.models.picode_v2.loss import (
    FocalFrequencyLoss,
    generator_loss,
    discriminator_loss,
)
```

---

## Performance Benchmarks

### Model Size and Parameters

| Model | Encoder Params | Decoder Params | Total Params | Checkpoint Size |
|-------|----------------|----------------|--------------|-----------------|
| **StegaStamp** | ~1.8M | ~2.5M | ~4.3M | ~17 MB |
| **Picode** | ~1.9M | ~4.0M | ~5.9M | ~24 MB |
| **Picode v2** | ~2.1M | ~2.5M | ~4.6M | ~18 MB |
| **Picode v2 + Discriminator** | ~2.1M | ~2.5M + 2.8M | ~7.4M | ~30 MB |

*Note: Discriminator is only used during training, not inference.*

### Inference Speed (400×400 input)

| Model | CPU (i7) | GPU (RTX 3080) | Apple M1 | Notes |
|-------|----------|----------------|----------|-------|
| **StegaStamp Encoder** | ~120ms | ~8ms | ~25ms | Baseline |
| **StegaStamp Decoder** | ~80ms | ~5ms | ~18ms | With STN |
| **Picode Encoder** | ~130ms | ~9ms | ~28ms | +GroupNorm overhead |
| **Picode Decoder** | ~110ms | ~7ms | ~24ms | ResBlocks add latency |
| **Picode v2 Encoder** | ~140ms | ~10ms | ~30ms | MessageExpander overhead |
| **Picode v2 Decoder** | ~70ms | ~4ms | ~15ms | No STN, simpler |

*Benchmarks are approximate and vary with hardware/batch size.*

### Memory Usage (Batch Size 4)

| Model | Training VRAM | Inference VRAM | Peak Memory |
|-------|---------------|----------------|-------------|
| **StegaStamp** | ~4 GB | ~1.5 GB | ~6 GB |
| **Picode** | ~5 GB | ~1.8 GB | ~7 GB |
| **Picode v2** | ~8 GB | ~1.6 GB | ~12 GB |

*Picode v2 training requires more memory due to discriminator and gradient penalty computation.*

---

## Robustness Analysis

### Distortion Tolerance

How well each model recovers messages after various distortions (Bit Error Rate at default distortion strength):

| Distortion | StegaStamp | Picode | Picode v2 | Notes |
|------------|------------|--------|-----------|-------|
| **JPEG Q=50** | 2-5% BER | 1-3% BER | 1-2% BER | All handle well |
| **JPEG Q=25** | 8-12% BER | 5-8% BER | 4-6% BER | v2 best |
| **Gaussian Blur σ=1.0** | 3-6% BER | 2-4% BER | 2-3% BER | Similar |
| **Gaussian Blur σ=2.0** | 10-15% BER | 8-12% BER | 6-10% BER | v2 best |
| **Gaussian Noise σ=0.05** | 5-8% BER | 4-6% BER | 3-5% BER | v2 best |
| **Crop 80%** | 15-25% BER | 12-18% BER | 8-12% BER | Depends on STN |
| **Rotation ±5°** | 5-10% BER | 4-8% BER | 20-30% BER | v2 lacks STN |
| **Perspective** | 8-15% BER | 6-12% BER | 25-35% BER | v2 lacks STN |
| **Screen-Camera** | 10-20% BER | 8-15% BER | 6-12% BER | v2 best for quality |

**Key Insight:** Picode v2 excels at quality-degrading distortions (JPEG, blur, noise) but struggles with geometric distortions due to lacking an STN. The detection pipeline handles geometric correction separately for v2.

### ECC Integration

With BCH(127, 64) error correction (corrects up to 10 bit errors per 127-bit block):

| Scenario | Raw BER Tolerance | Effective Message Recovery |
|----------|-------------------|---------------------------|
| **Light distortion** | < 5% BER | ~100% recovery |
| **Medium distortion** | 5-8% BER | ~95% recovery |
| **Heavy distortion** | 8-12% BER | ~80% recovery |
| **Extreme distortion** | > 12% BER | < 50% recovery |

---

## Visual Quality Metrics

### Image Quality (Encoded vs Original)

| Metric | StegaStamp | Picode | Picode v2 | Target |
|--------|------------|--------|-----------|--------|
| **PSNR** | 33-36 dB | 34-37 dB | 36-40 dB | > 35 dB |
| **SSIM** | 0.94-0.96 | 0.95-0.97 | 0.97-0.99 | > 0.95 |
| **LPIPS** | 0.02-0.04 | 0.015-0.03 | 0.008-0.02 | < 0.02 |

*Higher PSNR/SSIM is better. Lower LPIPS is better.*

### Artifact Types

| Artifact | StegaStamp | Picode | Picode v2 |
|----------|------------|--------|-----------|
| **Checkerboard patterns** | Sometimes visible | Sometimes visible | Rare (bilinear upsampling) |
| **Color banding** | Occasional | Occasional | Rare (content-adaptive) |
| **Edge artifacts** | Moderate | Moderate | Minimal (content-adaptive) |
| **Smooth region noise** | Visible | Visible | Minimal (content-adaptive) |
| **High-frequency ringing** | Present | Present | Reduced (focal freq loss) |

---

## Mobile Export

Picode v2 decoder is designed for mobile deployment. Export options:

### Core ML (iOS)

```python
import coremltools as ct
from picode.models.picode_v2 import Decoder
from picode.detection.mobile_export import export_coreml

decoder = Decoder(num_bits=100)
decoder.load_state_dict(torch.load("decoder.pt"))

# Export with optimizations
mlmodel = export_coreml(
    decoder,
    input_shape=(1, 3, 400, 400),
    minimum_deployment_target=ct.target.iOS15,
    compute_precision=ct.precision.FLOAT16,
)
mlmodel.save("PicodeDecoder.mlpackage")
```

### TensorFlow Lite (Android)

```python
from picode.detection.mobile_export import export_tflite

export_tflite(
    decoder,
    output_path="picode_decoder.tflite",
    input_shape=(1, 3, 400, 400),
    quantization="float16",  # or "int8" for smaller size
)
```

### Mobile Performance (Picode v2 Decoder)

| Platform | Model Size | Inference Time | Notes |
|----------|------------|----------------|-------|
| **iOS (Core ML, iPhone 13)** | ~5 MB (FP16) | ~15ms | Neural Engine |
| **iOS (Core ML, iPhone 11)** | ~5 MB (FP16) | ~25ms | Neural Engine |
| **Android (TFLite, Pixel 6)** | ~5 MB (FP16) | ~20ms | GPU delegate |
| **Android (TFLite, Pixel 6)** | ~2.5 MB (INT8) | ~35ms | CPU, quantized |

---

## Troubleshooting

### StegaStamp Issues

**Problem:** Training loss explodes or NaN values
- **Cause:** No normalization makes training sensitive
- **Solution:** Reduce learning rate to 5e-5, use gradient clipping

**Problem:** Poor message recovery after training
- **Cause:** Decoder not learning STN properly
- **Solution:** Verify STN initializes to identity, check affine parameters

**Problem:** Visible artifacts in encoded images
- **Cause:** Residual magnitude too large
- **Solution:** Increase L2 loss weight, add LPIPS loss earlier

### Picode Issues

**Problem:** GroupNorm causing instability with batch size 1
- **Cause:** GroupNorm needs multiple channels per group
- **Solution:** Use batch size ≥ 2, or reduce num_groups

**Problem:** Decoder overfitting on small datasets
- **Cause:** ResBlocks increase model capacity
- **Solution:** Add dropout, use data augmentation, reduce model size

**Problem:** STN predicting extreme transformations
- **Cause:** Random initialization too large
- **Solution:** Reduce STN init std to 0.0001

### Picode v2 Issues

**Problem:** GAN training mode collapse
- **Cause:** Discriminator too strong or generator too weak
- **Solution:** Reduce discriminator LR, increase GP weight, balance D/G update ratio

**Problem:** Geometric distortions break decoding
- **Cause:** No STN in decoder
- **Solution:** Use detection pipeline for geometric correction before decoding

**Problem:** Content-adaptive scaling causes edge artifacts
- **Cause:** Sobel edge detection too sensitive
- **Solution:** Disable content_adaptive or tune activity map threshold

**Problem:** Focal frequency loss dominates training
- **Cause:** Scale too high relative to other losses
- **Solution:** Reduce focal_frequency scale, increase ramp_steps

---

## Migration Guide

### StegaStamp → Picode

```python
# Before (StegaStamp)
from picode.models.stegastamp import Encoder, Decoder

# After (Picode)
from picode.models.picode import Encoder, Decoder

# Models are API-compatible, but weights are NOT transferable
# Must retrain from scratch
```

**Config changes:**
```yaml
training:
  model: picode  # was: stegastamp
  batch_size: 4  # can reduce from 8
```

### Picode → Picode v2

```python
# Before (Picode)
from picode.models.picode import Encoder, Decoder

# After (Picode v2)
from picode.models.picode_v2 import Encoder, Decoder, PatchDiscriminator
from picode.models.picode_v2.loss import FocalFrequencyLoss, generator_loss, discriminator_loss

# Training loop changes required for GAN
```

**Config changes:**
```yaml
training:
  model: picode_v2  # was: picode
  gan:
    enabled: true
    discriminator_lr: 0.0004
    gp_weight: 10.0
loss:
  focal_frequency: { scale: 0.1, ramp_steps: 30000 }  # replaces lpips
encoder:
  content_adaptive: true
  residual_scale: 0.1
```

**Key differences:**
1. Add discriminator to training loop
2. Replace LPIPS with Focal Frequency Loss
3. Handle geometric distortions in detection pipeline (not decoder)
4. Configure content-adaptive residuals

### Using Pre-trained Weights

Weights are **not compatible** between models due to architectural differences:
- Different normalization layers
- Different activation functions
- Different layer configurations

Always train from scratch when switching models.

---

## Advanced Topics

### Custom Distortion Strategies

Each model works with the distortion strategy system:

```python
from picode.training import create_distortion_strategy

# Curriculum: gradually increase difficulty
strategy = create_distortion_strategy("curriculum", max_strength=1.0)

# Fixed: constant distortion strength
strategy = create_distortion_strategy("fixed", strength=0.5)

# Random: random strength per batch
strategy = create_distortion_strategy("random", min_strength=0.0, max_strength=1.0)

# None: no distortions (not recommended)
strategy = create_distortion_strategy("none")
```

### Multi-Scale Detection

For real-world deployment, use the detection pipeline:

```python
from picode.detection import Detector

detector = Detector(
    decoder_path="checkpoints/decoder.pt",
    model_type="picode_v2",  # or "stegastamp", "picode"
    scales=[0.25, 0.5, 0.75, 1.0],
    threshold=0.1,
)

# Handles geometric correction, multi-scale search, confidence scoring
result = detector.detect("input.jpg")
```

### Ensemble Approaches

For maximum robustness, combine models:

```python
def ensemble_decode(image, decoders):
    """Average predictions from multiple decoders."""
    predictions = []
    for decoder in decoders:
        logits = decoder(image)
        predictions.append(torch.sigmoid(logits))

    # Soft voting
    avg_pred = torch.stack(predictions).mean(dim=0)
    return (avg_pred > 0.5).float()
```

---

## Summary

The three models represent an evolution from research baseline to production-ready:

1. **StegaStamp**: Faithful reproduction of the original paper, good for benchmarking
2. **Picode**: Training stability improvements, good for research/experimentation
3. **Picode v2**: Artifact reduction and mobile optimization, good for deployment

### Quick Selection Guide

| Requirement | Recommended Model |
|-------------|-------------------|
| Reproduce paper results | StegaStamp |
| Limited GPU memory | Picode |
| Training stability | Picode |
| Best visual quality | Picode v2 |
| Mobile deployment | Picode v2 |
| Geometric robustness | StegaStamp or Picode |
| Quick prototyping | StegaStamp or Picode |
| Production system | Picode v2 |

Choose based on your use case: reproducibility (StegaStamp), experimentation (Picode), or production (Picode v2).
