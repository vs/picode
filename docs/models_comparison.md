# Models Comparison

This document compares the four steganography model architectures in the Picode framework: **StegaStamp**, **PicodeLite**, **PicodeFrame**, and **PicoTrust**.

## Overview

| Model | Purpose | Key Innovation | Status |
|-------|---------|----------------|--------|
| **StegaStamp** | Baseline implementation | Original U-Net encoder with STN decoder (CVPR 2020) | Foundation model |
| **PicodeLite** | Lightweight experiment | No STN, flexible input sizes, BCH(63,36) | Experimental |
| **PicodeFrame** | Frame-border encoding | Message in border only, center pristine | Specialized |
| **PicoTrust** | Best overall model | E_post refinement, compact STN, grayscale residual | **Recommended** |

## Architecture Evolution

```
StegaStamp (CVPR 2020, Tancik et al.)
    |
    |--- PicodeLite [Lightweight experiment: no STN, flexible sizes]
    |
    |--- PicodeFrame [Frame-border variant: hard mask, border-pooling decoder]
    |
    └--- PicoTrust v1 [E_post refinement, MSE loss, compact STN, 256x256]
              |
              └--- PicoTrust v2 [512->256, grayscale residual, strength annealing, WGAN]
```

---

## StegaStamp

**Reference:** Tancik et al., "StegaStamp: Invisible Hyperlinks in Physical Photographs", CVPR 2020

**Location:** `picode/models/stegastamp/`

### Encoder Architecture

The StegaStamp encoder uses a U-Net architecture to embed binary messages into images.

**Message Preparation:**
- Linear expansion: `num_bits -> 7500 -> (3, 50, 50)`
- Nearest-neighbor upsampling: `(3, 50, 50) -> (3, 400, 400)`
- Concatenated with input image (6 channels total)

**Downsampling Path (5 layers):**
```
Input: (6, 400, 400)
  | Conv(6->32, k=3, s=1)    -> (32, 400, 400)
  | Conv(32->32, k=3, s=2)   -> (32, 200, 200)
  | Conv(32->64, k=3, s=2)   -> (64, 100, 100)
  | Conv(64->128, k=3, s=2)  -> (128, 50, 50)
  | Conv(128->256, k=3, s=2) -> (256, 25, 25)
```
- ReLU activations
- No normalization layers

**Upsampling Path (with skip connections):**
```
(256, 25, 25)
  | Conv(256->128, 2x2) + Upsample -> concat(skip4) -> (256, 50, 50)
  | Conv(256->128, 3x3)  -> (128, 50, 50)
  | Conv(128->64, 2x2) + Upsample -> concat(skip3) -> (128, 100, 100)
  | Conv(128->64, 3x3)  -> (64, 100, 100)
  | Conv(64->32, 2x2)  + Upsample -> concat(skip2) -> (64, 200, 200)
  | Conv(64->32, 3x3)   -> (32, 200, 200)
  | Conv(32->32, 2x2)  + Upsample -> concat(skip1, input) -> (70, 400, 400)
  | Conv(70->32, 3x3)   -> (32, 400, 400)
  | Conv(32->3, 1x1)    -> (3, 400, 400)  [residual]
```

**Output:** Unclamped residual addition (allows gradient flow beyond [0,1] during training):
```python
encoded = image + residual  # No clamping
```

### Decoder Architecture

**Spatial Transformer Network (STN):**
- 3 stride-2 conv layers -> flatten(128*50*50) -> FC(128) -> affine (2x3)
- Identity-initialized (zero weight, identity bias)
- Optional freeze/unfreeze for training stability

**Main CNN Path:**
```
Input: (3, 400, 400)
  | 5 stride-2 + 2 stride-1 conv layers
  | Spatial: 400->200->200->100->100->50->25->13
  | Channels: 3->32->32->64->64->64->128->128
  | Flatten -> FC(128*13*13=21632, 512) -> FC(512, num_bits)
```
- Output: Raw logits (no sigmoid)
- ReLU activations, no normalization

### Discriminator

**WGAN Discriminator** (default, `discriminator.py`):
- 5-layer conv stack: Conv(3->8->16->32->64->1), stride-2, ReLU
- No normalization (WGAN requirement)
- Weight clipping at 0.01
- Output: scalar score per image via global average pooling

**PatchGAN Discriminator** (alternative, `patchgan.py`):
- 5-layer conv stack with InstanceNorm, LeakyReLU(0.2)
- 70x70 receptive field, spatial score map output
- Gaussian weight init (std=0.02)

### Loss Functions

| Loss | Purpose | Default Weight | Ramp Steps |
|------|---------|----------------|------------|
| BCE (with logits) | Message accuracy | 1.0 | 1 |
| L2 (MSE) | Pixel fidelity | 1.5 | 15,000 |
| LPIPS | Perceptual similarity | 1.0 | 15,000 |
| Border falloff | Edge penalty | 10.0 gain | delay 60k |
| WGAN G loss | Adversarial realism | 1.0 | 15,000 |

### Characteristics

| Aspect | Detail |
|--------|--------|
| Default size | 400x400 |
| Num bits | 100 |
| Normalization | None |
| Activation | ReLU |
| STN | Yes (large: flatten 128*50*50) |
| Encoder params | ~1.75M |
| Decoder params | ~54M (dominated by STN FC layer) |

---

## PicodeLite

**Purpose:** Lightweight variant for flexible input sizes and mobile inference.

**Location:** `picode/models/picodelite/`

### Key Differences from StegaStamp

1. **No STN** in decoder -- simpler, fewer parameters
2. **Flexible input sizes** -- any dimension divisible by 16 (U-Net skip connections)
3. **63 bits** default (BCH(63,36) error correction)
4. **Message expansion** adapts to input size: `Linear -> (3, 32, 32) -> interpolate to target`

### Encoder Architecture

Same U-Net depth as StegaStamp (5 levels) with minor differences:
- Message preparation uses `Linear(num_bits, 3*32*32)` then nearest interpolation to target size
- No unused `conv10` layer (cleaner residual path)
- Default size: 512x512

### Decoder Architecture

**StegaStamp-style CNN without STN:**
```
Input: (3, H, W)
  | 5 stride-2 + 2 stride-1 conv layers
  | Channels: 3->32->32->64->64->64->128->128
  | Flatten -> FC(128*(H/32)^2, 512) -> FC(512, num_bits)
```
- At 512x512: flatten size = 128*16*16 = 32,768
- No STN, no normalization

### Configuration

```yaml
# configs/picodelite.yaml
model:
  type: picodelite
  encoder_size: 512
  decoder_size: 512
training:
  num_bits: 63
  image_size: 512
loss:
  message: { scale: 1.0, ramp_steps: 1 }
  l2: { scale: 2.0, ramp_steps: 10000 }
  lpips: { scale: 1.5, ramp_steps: 10000 }
  message_loss_type: bce
  yuv_weights: [1.0, 100.0, 100.0]  # Heavy chrominance penalty
  gan_config: { enabled: false }
```

### Characteristics

| Aspect | Detail |
|--------|--------|
| Default size | 512x512 |
| Num bits | 63 |
| STN | No |
| Status | Experimental, not the recommended model |

---

## PicodeFrame

**Purpose:** Encode messages only in a narrow frame border around the image, keeping the center pixels completely untouched.

**Location:** `picode/models/picodeframe/`

### Key Innovation

A hard mask guarantees the original center pixels are never modified. The encoder generates a natural-looking frame via reflection-padded outpainting while encoding the message into the border pixels only.

### Encoder Architecture

Based on StegaStamp U-Net with key modifications:
- **7 input channels**: 3 (padded image) + 1 (center mask) + 3 (message spatial)
- **Single-channel residual**: `Conv(32->1, 1x1)` broadcast to RGB -- no color artifacts by construction
- **Hard mask output**: `framed = image * mask + (image + residual) * (1 - mask)`
- Optional `max_residual_amplitude` with tanh bounding

### Decoder Architecture

**Hybrid CNN + border-pooling architecture:**

The 7-layer strided CNN compresses the 400x400 image to 13x13, which reduces a 32px border to ~1px in the feature map. To solve this, a border-pooling branch provides a direct path for border information.

**CNN Branch (4 input channels):**
- 3 RGB + 1 border indicator channel
- Standard 7-layer strided CNN -> flatten(128*13*13)

**Border-Pooling Branch:**
- Strip-pools each border side along narrow dimension (frame_width -> 1)
- Preserves long dimension (400 pixels)
- 4 strips * 3 channels * height = 4800 features -> FC(256)

**Combined:** `concat(CNN[21632], Border[256]) -> FC(512) -> FC(num_bits)`

Both branches operate on STN-corrected images.

### Frame-Specific Loss Functions

| Loss | Purpose | Default Weight |
|------|---------|----------------|
| BCE (message) | Message accuracy | 7.0 |
| Frame L2 | Pixel fidelity in border only | 1.0 |
| Frame LPIPS | Perceptual quality of border | 0.5 |
| Frame color | Penalize cross-channel variance | 0.0 (disabled; greyscale residual handles this) |
| STN reg | Keep STN near identity | 0.1 |

### Configuration

```yaml
# configs/picodeframe_baseline.yaml
model:
  type: picodeframe
  encoder_size: 400
  decoder_size: 400
training:
  num_bits: 127       # BCH(127,64)
  image_size: 400
  warmup_steps: 5000  # Long warmup: message-only (frame signal is thin)
  no_im_loss_steps: 10000
frame:
  min_frame_pct: 0.02
  max_frame_pct: 0.05
  fixed_frame_steps: 20000  # Max frame for first N steps, then randomize
  frame_l2_scale: 1.0
  frame_lpips_scale: 0.5
  stn_reg_scale: 0.1
```

### Characteristics

| Aspect | Detail |
|--------|--------|
| Default size | 400x400 |
| Num bits | 127 (BCH(127,64)) |
| STN | Yes (full StegaStamp STN in decoder) |
| Center modification | None (hard mask guarantee) |
| Residual channels | 1 (greyscale) |
| Best for | Scenarios requiring pristine center image |

---

## PicoTrust

**Purpose:** Best overall model. Combines StegaStamp's proven U-Net with targeted TrustMark enhancements.

**Location:** `picode/models/picotrust/`

### Key Innovations

1. **E_post refinement block** -- replaces StegaStamp's single residual conv with 3-layer post-processing
2. **Grayscale residual** -- 1-channel E_post output broadcast to RGB = zero colour shifts by construction
3. **Compact STN** -- AdaptiveAvgPool2d instead of flatten+FC, eliminating ~67M parameters
4. **MSE message loss** -- avoids BCE's trivial 0.5 equilibrium
5. **Softsign amplitude bounding** (v2) -- `strength * x / (1 + |x|)`, gradient never reaches zero
6. **Strength annealing** (v2) -- 1.0 -> 0.03 over training for high PSNR

### Encoder Architecture

**StegaStamp U-Net Backbone** (parameterized size, default 256x256 for v1, 512x512 for v2):
- Same architecture as StegaStamp encoder (5-level U-Net with skip connections)
- No normalization, ReLU activations

**E_post Refinement Block** (replaces StegaStamp's `conv10 + residual` layers):
```python
e_post = nn.Sequential(
    nn.Conv2d(32, 32, 3, padding=1),  # Spatial refinement
    nn.ReLU(),
    nn.Conv2d(32, 16, 1),             # Channel reduction
    nn.SiLU(),
    nn.Conv2d(16, 1, 1),              # Grayscale residual (no activation)
)
```
- Outputs 1 channel -> `expand(-1, 3, -1, -1)` for zero colour shift
- Last layer zero-initialized so residual starts at zero

**Amplitude Bounding (v2 only):**
```python
# Softsign-like: gradient 1/(1+|x|)^2 never reaches zero
residual = strength * raw_residual / (1.0 + raw_residual.abs())
```

**Optional learned spatial mask** (`use_mask=True`):
```python
mask_head = Conv(32->16, 3x3) + ReLU + Conv(16->1, 1x1) + Sigmoid
residual = residual * mask  # Focus encoding on textured regions
```

### Decoder Architecture

**Compact STN** (resolution-independent):
```
3 stride-2 convs -> AdaptiveAvgPool2d(1) -> FC(128, 128) -> affine (2x3)
```
- AdaptiveAvgPool2d replaces StegaStamp's flatten(128*50*50), eliminating ~40M params
- Works at any input resolution

**Main CNN** (StegaStamp-style, preserves spatial structure):
```
5 stride-2 + 2 stride-1 convs -> flatten(spatial^2) -> FC(512) -> FC(num_bits)
```
- At 256x256: spatial = 8, flatten = 128*8*8 = 8,192
- At 512x512: spatial = 16, flatten = 128*16*16 = 32,768

### Training Configuration

**v1 (256x256):**
```yaml
# configs/picotrust_baseline.yaml
model:
  type: picotrust
  encoder_size: 256
  decoder_size: 256
training:
  num_bits: 100
  num_steps: 200000
  lr: 0.0001
  stn_lr_scale: 0.01
loss:
  message: { scale: 5.0, ramp_steps: 1 }
  l2: { scale: 1.5, ramp_steps: 20000 }
  lpips: { scale: 1.0, ramp_steps: 20000 }
  ffl: { scale: 1.0, ramp_steps: 20000, delay_steps: 20000 }
  message_loss_type: mse
  gan_config:
    enabled: true
    discriminator_lr: 0.00001
```

**v2 (512 encoder -> 256 decoder):**
```yaml
# configs/picotrust_v2.yaml
model:
  type: picotrust
  encoder_size: 512
  decoder_size: 256  # Trainer downsamples 512->256 for decoder
training:
  num_bits: 100
  num_steps: 200000
  lr: 0.0001
  no_im_loss_steps: 10000    # 10k steps pure message
  residual_strength: 1.0     # Start unbounded
  residual_strength_anneal_target: 0.03  # Target for ~38+ dB PSNR
  residual_strength_anneal_start: 10000
  residual_strength_anneal_steps: 60000
  phase2_step: 60000
  phase2_decoder_lr_scale: 0.1
loss:
  message: { scale: 5.0, ramp_steps: 1 }
  l2: { scale: 1.5, ramp_steps: 50000 }
  lpips: { scale: 1.0, ramp_steps: 50000 }
  ffl: { scale: 1.0, ramp_steps: 50000, delay_steps: 30000 }
  message_loss_type: mse
  gan_config:
    enabled: true
    discriminator_lr: 0.00001
```

### Characteristics

| Aspect | Detail |
|--------|--------|
| Default size | 256x256 (v1), 512->256 (v2) |
| Num bits | 100 |
| Normalization | None |
| Activation | ReLU (U-Net), SiLU (E_post middle layer) |
| STN | Compact (AdaptiveAvgPool2d) |
| Residual channels | 1 (greyscale, broadcast to 3) |
| Encoder params | ~1.75M |
| Decoder params | ~4.68M (at 256x256) |
| Total params | ~6.4M |
| Message loss | MSE (not BCE) |

---

## Detailed Comparison Tables

### Encoder Comparison

| Feature | StegaStamp | PicodeLite | PicodeFrame | PicoTrust |
|---------|------------|------------|-------------|-----------|
| **Backbone** | U-Net (5 levels) | U-Net (5 levels) | U-Net (5 levels) | U-Net (5 levels) |
| **Message Prep** | Linear->7500->(3,50,50)->nearest | Linear->(3,32,32)->nearest | Linear->7500->(3,50,50)->nearest | Linear->7500->(3,50,50)->nearest |
| **Input Channels** | 6 (img+msg) | 6 (img+msg) | 7 (img+mask+msg) | 6 (img+msg) |
| **Post-processing** | Single conv(32->3) | Single conv(32->3) | Single conv(32->1) | E_post 3-layer block |
| **Residual Channels** | 3 (RGB) | 3 (RGB) | 1 (greyscale) | 1 (greyscale) |
| **Residual Bounding** | None (unclamped) | None (unclamped) | Optional tanh | Softsign + strength annealing (v2) |
| **Hard Mask** | No | No | Yes (center preserved) | No |
| **Normalization** | None | None | None | None |
| **Default Size** | 400x400 | 512x512 | 400x400 | 256 (v1) / 512 (v2) |

### Decoder Comparison

| Feature | StegaStamp | PicodeLite | PicodeFrame | PicoTrust |
|---------|------------|------------|-------------|-----------|
| **Architecture** | CNN + FC | CNN + FC | Hybrid CNN + border-pooling | CNN + FC |
| **STN** | Yes (large) | No | Yes (large) | Yes (compact, AdaptiveAvgPool) |
| **STN FC size** | 128*50*50=320K | N/A | 128*50*50=320K | 128 (via GAP) |
| **Border Branch** | No | No | Yes (strip-pooling) | No |
| **CNN Input** | 3 channels | 3 channels | 4 channels (RGB + border mask) | 3 channels |
| **Normalization** | None | None | None | None |
| **Output** | Raw logits | Raw logits | Raw logits | Raw logits |
| **Default Size** | 400x400 | 512x512 | 400x400 | 256x256 |

### Loss Function Comparison

| Loss Type | StegaStamp | PicodeLite | PicodeFrame | PicoTrust |
|-----------|------------|------------|-------------|-----------|
| **Message** | BCE | BCE | BCE | MSE |
| **Message Scale** | 1.0 | 1.0 | 7.0 | 5.0 |
| **Pixel** | L2 (MSE) | L2 (MSE) | Frame L2 | L2 (MSE) |
| **Perceptual** | LPIPS | LPIPS | Frame LPIPS | LPIPS |
| **Frequency** | No | No | No | FFL (delayed) |
| **Adversarial** | WGAN | No | No | WGAN |
| **Border Falloff** | Yes | No | N/A (frame mask) | Yes |
| **STN Reg** | No | N/A | Yes | No |
| **YUV Weights** | [1, 1, 1] | [1, 100, 100] | N/A | [1, 1, 1] |

### Training Characteristics

| Aspect | StegaStamp | PicodeLite | PicodeFrame | PicoTrust |
|--------|------------|------------|-------------|-----------|
| **Default Steps** | 140,000 | 140,000 | 140,000 | 200,000 |
| **Learning Rate** | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| **Batch Size** | 4 | 4 | 4 | 8 (v1) / 4 (v2) |
| **Warmup Steps** | 500 | 2,000 | 5,000 | 500 |
| **Message-only Steps** | 500 | 2,000 | 10,000 | 500 (v1) / 10,000 (v2) |
| **GAN** | WGAN | None | None | WGAN |
| **Distortion Strategy** | Curriculum | Curriculum | Curriculum | Curriculum |
| **Grad Clip** | 0.25 | -- | -- | 0.25 |

---

## Performance Results

### PicoTrust v1 (256x256) -- Measured

Trained to 200K steps on COCO dataset. Checkpoint: `checkpoints/picotrust_baseline/best.pt`

| Metric | Value |
|--------|-------|
| **Bit Accuracy** | 99.8% |
| **PSNR** | 26.54 dB |

**Robustness sweep:**

| Distortion | Strength | Bit Accuracy |
|------------|----------|--------------|
| None | -- | 99.8% |
| JPEG Q10 | Extreme | 99.4% |
| JPEG Q50 | Medium | 100% |
| Noise 0.1 | High | 98.8% |
| Blur sigma=3.0 | Heavy | 98.8% |
| Brightness +/-0.5 | Strong | 96.8% |
| Contrast 0.5 | Strong | 100% |

### PicoTrust v2 (512->256, grayscale residual) -- Measured

| Metric | Value |
|--------|-------|
| **Bit Accuracy** | 98.4% |
| **PSNR** | 32.82 dB |
| **JPEG Q10** | 98.6% |

### Comparison with TrustMark (from literature)

| Model | PSNR | JPEG Q10 Robustness | Notes |
|-------|------|---------------------|-------|
| **TrustMark-P** | ~49 dB | Fails | High PSNR but no compression robustness |
| **TrustMark-Q** | ~42 dB | Fails | Better but still fails at Q10 |
| **PicoTrust v1** | 26.54 dB | 99.4% | Lower PSNR, far better robustness |
| **PicoTrust v2** | 32.82 dB | 98.6% | Better PSNR than v1, strong robustness |

**Key insight:** PicoTrust trades absolute PSNR for real-world robustness. TrustMark's high PSNR comes from barely-perceptible residuals that are destroyed by JPEG compression. PicoTrust's curriculum distortion training produces residuals that survive aggressive compression.

### Other Models -- No Published Results

StegaStamp, PicodeLite, and PicodeFrame do not have final benchmark results available. StegaStamp is the baseline used for comparison; PicodeLite was an experiment; PicodeFrame is specialized for frame-border use cases and requires separate evaluation methodology.

### Model Size

| Model | Encoder Params | Decoder Params | Total | Notes |
|-------|----------------|----------------|-------|-------|
| **StegaStamp** | ~1.75M | ~54M | ~56M | STN FC layer dominates |
| **PicodeLite** | ~1.6M | ~0.6M (at 512) | ~2.2M | No STN |
| **PicodeFrame** | ~1.75M | ~54M + border FC | ~56M | Full STN + border branch |
| **PicoTrust** | ~1.75M | ~4.68M | ~6.4M | Compact STN saves ~50M params |

*Note: StegaStamp and PicodeFrame decoder params are dominated by the STN's `Linear(128*50*50, 128)` = ~40M params. PicoTrust's compact STN uses AdaptiveAvgPool2d(1) -> `Linear(128, 128)` = ~16K params.*

---

## Key Architectural Lessons

### ResNet50 Decoder Does NOT Work for Steganography

Attempted during PicoTrust development (Runs 1-2). `AdaptiveAvgPool2d(1)` reduces feature maps to a single 2048-dim vector, destroying all spatial information. Steganography requires spatial awareness to detect subtle per-pixel perturbations. StegaStamp's flatten-from-8x8 approach preserves spatial structure.

### BCE Has a Trivial 0.5 Equilibrium

With BCE loss, the decoder can output sigmoid(0) = 0.5 for all bits, achieving loss = ln(2) = 0.693. This is a stable equilibrium that training cannot escape. MSE loss (`mse_loss(sigmoid(logits), targets)`) has no such equilibrium -- the gradient always pushes away from 0.5.

### Message Scale Must Dominate

PicoTrust uses `message.scale = 5.0` against total image losses of ~3.5 at full ramp. The encoder-decoder must establish communication before image quality losses are applied, otherwise the encoder learns to produce zero residual (trivially good image quality but no message).

### Grayscale Residual Eliminates Colour Artifacts

Both PicodeFrame and PicoTrust output 1-channel residuals broadcast to RGB. This guarantees zero colour shift by construction, eliminating the need for chrominance penalties or YUV weighting.

### Compact STN via AdaptiveAvgPool2d

StegaStamp's STN uses `flatten(128*50*50) -> Linear(320000, 128)` = ~40M params. PicoTrust replaces this with `AdaptiveAvgPool2d(1) -> Linear(128, 128)` = ~16K params. The STN only needs to predict 6 affine parameters -- global average pooling provides sufficient information.

---

## Use Case Recommendations

| Requirement | Recommended Model |
|-------------|-------------------|
| Best overall accuracy + robustness | **PicoTrust v2** |
| Highest compression robustness | **PicoTrust v1** (99.4% at JPEG Q10) |
| Best PSNR with robustness | **PicoTrust v2** (32.82 dB) |
| Center image must be pristine | **PicodeFrame** |
| Reproduce StegaStamp paper | **StegaStamp** |
| Fewest parameters | **PicodeLite** (~2.2M) |
| Quick prototyping / baseline | **StegaStamp** |

---

## Import Patterns

```python
# StegaStamp
from picode.models.stegastamp import Encoder, Decoder
from picode.models.stegastamp.loss import compute_loss, message_loss, image_loss
from picode.models.stegastamp import Discriminator

# PicodeLite
from picode.models.picodelite import Encoder, Decoder

# PicodeFrame
from picode.models.picodeframe import Encoder, Decoder
from picode.models.picodeframe.loss import (
    compute_picodeframe_loss,
    message_loss,
    frame_l2_loss,
    frame_lpips_loss,
    frame_color_loss,
    stn_scale_loss,
)

# PicoTrust
from picode.models.picotrust import Encoder, Decoder

# Factory (model-agnostic)
from picode.models.factory import create_encoder, create_decoder
```

---

## Configuration Reference

Available configs in `picode-model/configs/`:

| Config File | Model | Description |
|-------------|-------|-------------|
| `stegastamp_baseline.yaml` | StegaStamp | Original defaults (400x400, 100 bits) |
| `stegastamp_original.yaml` | StegaStamp | Strict original TF reproduction |
| `picodelite.yaml` | PicodeLite | Default (512x512, 63 bits) |
| `picodelite_kaggle.yaml` | PicodeLite | Kaggle GPU config |
| `picodelite_256bit_kaggle.yaml` | PicodeLite | 256-bit variant |
| `picodeframe_baseline.yaml` | PicodeFrame | Default (400x400, 127 bits) |
| `picodeframe_kaggle.yaml` | PicodeFrame | Kaggle GPU config |
| `picotrust_baseline.yaml` | PicoTrust | v1 (256x256, 100 bits) |
| `picotrust_v2.yaml` | PicoTrust | v2 (512->256, strength annealing) |
| `modal_training.yaml` | StegaStamp | Modal cloud training |
| `detection_training.yaml` | -- | FastDetector training |
| `gradient_test.yaml` | -- | Gradient debugging |

---

## Troubleshooting

### All Models

**Problem:** Training loss stuck at ln(2) = 0.693 (BCE) or 0.25 (MSE)
- **Cause:** Decoder collapsed to trivial solution (output 0.5 for all bits)
- **Solution:** Use MSE loss instead of BCE. Increase `message.scale`. Ensure `no_im_loss_steps` gives the encoder-decoder time to bootstrap communication before image losses push residual to zero.

**Problem:** Decoder prob_std drops to ~0.02 then recovers
- **Cause:** Normal behavior during distortion ramp-up. Decoder temporarily loses confidence as distortions increase.
- **Solution:** No action needed. If it stays low (< 0.01) for many steps, check loss balance.

### PicoTrust-Specific

**Problem:** PicoTrust v2 training collapse with tanh+strength residual
- **Cause:** `tanh(x) * strength` with initial `strength=0.1` kills encoder gradient flow, preventing bootstrap
- **Solution:** Start with `residual_strength=1.0` (effectively unconstrained), anneal down over training. The softsign bound `x / (1 + |x|)` also helps because its gradient never reaches zero.

**Problem:** PatchGAN with high LR causes collapse
- **Cause:** PatchGAN with 20x discriminator LR (0.0002 vs 0.00001) overpowers the generator
- **Solution:** Use WGAN with `discriminator_lr=1e-5` (matching v1 settings)

### PicodeFrame-Specific

**Problem:** Decoder cannot learn from fresh init (bit accuracy stuck at 50%)
- **Cause:** 7-layer strided CNN compresses 32px border to ~1px in feature map, drowning signal
- **Solution:** The border-pooling branch provides a direct path. Ensure `warmup_steps` is long enough (5000+) and `message.scale` is high (7.0).

**Problem:** STN learns to zoom in, cropping the frame
- **Cause:** STN affine parameters diverge from identity
- **Solution:** Enable `stn_reg_scale` (default 0.1) to regularize toward identity transform.
