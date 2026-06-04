# Model Improvements: Reducing Visual Artifacts in Picode

> **Note:** All file paths in this document are relative to `picode-model/` unless otherwise specified.

## Implementation Status (as of 2026-06-04)

Several of the proposed improvements were implemented during PicoTrust v2 training. Others were solved through alternative approaches that proved more effective.

| Improvement | Status | Notes |
|-------------|--------|-------|
| 4.1 Improved Message Expansion | **Not implemented** | Still uses Linear→7500→reshape(3,50,50)→nearest-neighbor. Current results are good; future work. |
| 4.2 Focal Frequency Loss | **DONE** | Used in v2 with scale 1.0, ramp 50k, delay 30k. |
| 4.3 Adversarial Training | **DONE** | WGAN with LR 1e-5, g_scale 1.0, ramp 20k. PatchGAN discriminator. |
| 4.4 Content-Adaptive Residual Scaling | **Solved differently** | Grayscale residual (1-ch E_post broadcast to RGB) + softsign bounding with strength annealing + border falloff loss. |
| 4.5 JND Masking | **Not implemented** | Strength annealing achieves similar PSNR control. |
| 4.6 YUV Color Space | **Not implemented** | Grayscale residual eliminates colour shifts by construction, making YUV unnecessary. |
| 4.7 Bilinear Upsampling | **Not done in message expansion** | Still uses nearest-neighbor for 50→image_size. Architecture works well regardless. |

### Results: PicoTrust v1 → v2

| Metric | v1 | v2 | Target | Status |
|--------|-----|-----|--------|--------|
| PSNR | 26.54 dB | 32.82 dB | >38 dB | Improved +6.3 dB, not yet at target |
| Bit accuracy (clean) | 99.8% | 98.4% | 99%+ | Slight trade-off for imperceptibility |
| Bit accuracy (JPEG Q10) | 99.4% | 98.6% | >90% | Exceeds target |
| Colour shifts | Visible | None | None | Solved via grayscale residual |

**Key architectural insight:** The colour shift problem (Section 2.1 "Color channel imbalance") was solved not by YUV embedding or content-adaptive scaling, but by making E_post output a single grayscale channel that is broadcast to RGB. This eliminates colour artifacts by construction.

---

## Executive Summary

Analysis of encoded images from the current model reveals **highly visible structured artifacts** that compromise imperceptibility despite achieving reliable message decoding. This document proposes architectural and training improvements to make watermarks visually undetectable while maintaining decoding robustness.

---

## 1. Mobile Deployment Constraint

**Critical requirement**: The decoder must run efficiently on mobile devices. Per the [Mobile Deployment Guide](./mobile_deployment.md), this imposes hard constraints on architectural choices:

| Constraint | Requirement | Impact on Improvements |
|------------|-------------|----------------------|
| **No GroupNorm** | Use BatchNorm only (fuses with conv) | Affects encoder/decoder architecture |
| **No LeakyReLU** | Use ReLU6 (hardware accelerated) | Affects all proposed modules |
| **No STN** | Train robustness instead | Already planned for mobile decoder |
| **Decoder < 500K params** | Depthwise separable convolutions | Limits decoder complexity |
| **Inference < 100ms** | Minimize computational overhead | JND/activity maps add latency |

**Implication**: All improvements below must have **two variants**:
1. **Training/Server variant**: Full quality, can use any operations
2. **Mobile-compatible variant**: Constrained to mobile-friendly ops

The encoder runs server-side (no constraints), but decoder improvements must be mobile-compatible or provide a mobile fallback.

---

## 2. Artifact Analysis

### 2.1 Observed Artifacts

| Artifact Type | Description | Severity |
|--------------|-------------|----------|
| **Structured wave patterns** | Diagonal stripe/wave patterns visible across entire image | Critical |
| **Border/frame effects** | Rectangular frame visible at image edges | High |
| **Color channel imbalance** | Purple/green tint in residuals (R, G, B treated differently) | Medium |
| **Smooth region visibility** | Artifacts most visible in uniform areas (sky, walls, grass) | High |
| **Content-agnostic embedding** | Same pattern regardless of image content | Critical |

### 2.2 Residual Pattern Analysis

The residuals show:
- **Regular, repeating diagonal stripes** (~45° angle)
- **8x8 or similar block-like structure** from nearest-neighbor upsampling
- **High magnitude** relative to Just Noticeable Difference (JND)
- **No adaptation** to local image texture or edges

---

## 3. Root Cause Analysis

### 3.1 Message Expansion Architecture

**Current implementation** (`prepare_message`):
```python
x = F.relu(self.secret_dense(message))  # (B, 7500)
x = x.view(-1, 3, 50, 50)               # (B, 3, 50, 50)
x = F.interpolate(x, scale_factor=8, mode="nearest")  # (B, 3, 400, 400)
```

**Problems:**
1. **Nearest-neighbor creates blocky 8x8 artifacts** - Each 50x50 "message pixel" becomes an 8x8 block
2. **Only 3 channels** - Limited capacity forces high-amplitude perturbations
3. **No spatial adaptation** - Message pattern doesn't adapt to image content

### 3.2 Loss Function Limitations

**Current losses:**
- L2 (MSE): Treats all pixels equally, doesn't penalize structured noise
- LPIPS: Good for perceptual quality but doesn't target frequency artifacts
- Edge loss: Only activated late in training (step 60k+)

**Missing:**
- **Frequency-domain loss** - To penalize visible frequency components
- **Adversarial loss** - To make encoded images indistinguishable from originals
- **Texture-adaptive loss** - Higher tolerance in textured regions

### 3.3 Encoder Output Stage

**Current:**
```python
residual = self.residual(x)  # (B, 3, 400, 400) - unbounded
encoded = image + residual
encoded = torch.clamp(encoded, 0, 1)
```

**Problems:**
1. **Unbounded residual** - No constraint on perturbation magnitude
2. **No content-adaptive scaling** - Same perturbation in smooth and textured regions
3. **Hard clamp** - Introduces non-differentiable boundaries

---

## 4. Proposed Improvements

### 4.1 Improved Message Expansion (Priority: Critical)

**Problem:** Nearest-neighbor upsampling creates visible 8x8 block artifacts.

**Solution:** Multi-scale learned upsampling with bilinear interpolation:

```python
class ImprovedMessageExpander(nn.Module):
    """Content-adaptive message expansion with smooth upsampling."""

    def __init__(self, num_bits: int = 100, hidden_ch: int = 64):
        super().__init__()

        # Dense expansion to more channels
        self.dense = nn.Linear(num_bits, hidden_ch * 25 * 25)  # 5x5 base

        # Progressive upsampling with convolutions
        # 5x5 -> 25x25 -> 100x100 -> 400x400
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_ch, hidden_ch, 4, stride=5, padding=0),
            nn.GroupNorm(8, hidden_ch),
            nn.LeakyReLU(0.2),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_ch, 32, 4, stride=4, padding=0),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=4, padding=0),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.2),
        )
        # Final refinement
        self.refine = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, message: Tensor) -> Tensor:
        x = F.leaky_relu(self.dense(message), 0.2)
        x = x.view(-1, 64, 5, 5)
        x = self.up1(x)  # -> 25x25
        x = self.up2(x)  # -> 100x100
        x = self.up3(x)  # -> 400x400
        return self.refine(x)
```

**Expected improvement:** Eliminates blocky 8x8 artifacts from nearest-neighbor upsampling.

> **Mobile compatibility:** ⚠️ **Encoder-only** - This improvement is encoder-side only (runs on server). The mobile decoder does not use message expansion. However, note the use of GroupNorm and LeakyReLU - if a mobile encoder is ever needed, replace with BatchNorm and ReLU6.

### 4.2 Focal Frequency Loss (Priority: High) — DONE

> **Implemented in PicoTrust v2:** scale 1.0, ramp 50k steps, delay 30k steps, alpha 1.0.

**Problem:** L2/LPIPS don't penalize structured frequency artifacts.

**Solution:** Add Focal Frequency Loss (FFL) to target visible frequency components:

```python
class FocalFrequencyLoss(nn.Module):
    """Penalize differences in frequency domain, focusing on hard frequencies.

    Reference: Jiang et al., ICCV 2021 "Focal Frequency Loss"
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha  # Focal weight exponent

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # 2D FFT
        pred_freq = torch.fft.fft2(pred, norm='ortho')
        target_freq = torch.fft.fft2(target, norm='ortho')

        # Frequency distance (magnitude)
        pred_mag = torch.abs(pred_freq)
        target_mag = torch.abs(target_freq)

        # Per-frequency weight: higher weight for frequencies with larger error
        freq_distance = (pred_mag - target_mag) ** 2
        weight = freq_distance ** self.alpha  # Focal weighting
        weight = weight / (weight.sum() + 1e-8)  # Normalize

        # Weighted frequency loss
        loss = (weight * freq_distance).sum()
        return loss
```

**Training integration:**
```yaml
loss:
  ffl:
    scale: 0.1
    ramp_steps: 10000
    alpha: 1.0  # Focal weight
```

**Expected improvement:** Reduces structured wave/stripe patterns by directly penalizing frequency-domain artifacts.

> **Mobile compatibility:** ✅ **Training-only** - FFL is a loss function used only during training. No runtime impact on mobile inference.

### 4.3 Adversarial Training with Discriminator (Priority: High) — DONE

> **Implemented in PicoTrust v2:** WGAN with gradient penalty, discriminator LR 1e-5, generator loss scale 1.0, ramp 20k steps. PatchGAN architecture as proposed below.

**Problem:** No mechanism to make encoded images indistinguishable from originals.

**Solution:** Add a discriminator that tries to distinguish encoded from original images:

```python
class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for real/encoded classification.

    Operates on 70x70 patches to detect local artifacts.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        def block(in_ch, out_ch, stride=2, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, 64, norm=False),  # 200x200
            block(64, 128),                       # 100x100
            block(128, 256),                      # 50x50
            block(256, 512, stride=1),            # 50x50
            nn.Conv2d(512, 1, 4, padding=1),      # 49x49
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
```

**GAN loss (WGAN-GP recommended for stability):**
```python
def discriminator_loss(real_pred, fake_pred, real, fake, lambda_gp=10.0):
    """WGAN-GP discriminator loss."""
    # Wasserstein distance
    d_loss = fake_pred.mean() - real_pred.mean()

    # Gradient penalty
    alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)
    d_interp = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interp, inputs=interpolates,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return d_loss + lambda_gp * gradient_penalty

def generator_loss(fake_pred):
    """Generator tries to fool discriminator."""
    return -fake_pred.mean()
```

**Expected improvement:** Forces encoder to produce perturbations that are statistically indistinguishable from natural image variation.

> **Mobile compatibility:** ✅ **Training-only** - The discriminator is only used during training. Note: uses InstanceNorm and LeakyReLU but this doesn't affect deployment since discriminator is discarded after training.

### 4.4 Content-Adaptive Residual Scaling (Priority: High) — Solved Differently

> **Not implemented as proposed.** Instead, three alternative techniques addressed the same problems:
> 1. **Grayscale residual** — E_post outputs 1 channel broadcast to RGB, eliminating colour shifts architecturally.
> 2. **Softsign bounding with strength annealing** — Controls residual magnitude without destroying gradient flow (tanh+strength constraint was found to kill encoder gradient flow).
> 3. **Border falloff in loss function** — Reduces edge/frame artifacts (Section 2.1 "Border/frame effects").
>
> These proved more effective than a runtime activity map because they address the root causes (unbounded residual, colour imbalance, border effects) rather than masking symptoms.

**Problem:** Same perturbation magnitude in smooth and textured regions.

**Original proposed solution:** Scale residual based on local image activity:

```python
class ContentAdaptiveEncoder(nn.Module):
    """Encoder with texture-adaptive residual scaling."""

    def compute_activity_map(self, image: Tensor) -> Tensor:
        """Compute local image activity (texture/edge density)."""
        # Convert to grayscale
        gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        gray = gray.unsqueeze(1)

        # Local variance (activity measure)
        kernel_size = 7
        mean = F.avg_pool2d(
            F.pad(gray, (3, 3, 3, 3), mode='reflect'),
            kernel_size, stride=1
        )
        mean_sq = F.avg_pool2d(
            F.pad(gray ** 2, (3, 3, 3, 3), mode='reflect'),
            kernel_size, stride=1
        )
        variance = mean_sq - mean ** 2

        # Normalize to [0.1, 1.0] - always allow some embedding
        activity = torch.clamp(variance / (variance.max() + 1e-8), 0.1, 1.0)
        return activity

    def forward(self, image: Tensor, message: Tensor) -> Tensor:
        # ... existing encoder forward pass ...
        residual = self.residual(x)

        # Scale residual by local activity
        activity_map = self.compute_activity_map(image)
        scaled_residual = residual * activity_map

        # Optional: also apply global magnitude constraint
        max_perturbation = 0.05  # Max 5% change per channel
        scaled_residual = torch.tanh(scaled_residual) * max_perturbation

        encoded = image + scaled_residual
        return torch.clamp(encoded, 0, 1)
```

**Expected improvement:** Hides perturbations in textured regions where they're less visible, reduces artifacts in smooth areas.

> **Mobile compatibility:** ⚠️ **Encoder-only** - Content-adaptive scaling runs in the encoder (server-side). The decoder does not need to know how the watermark was scaled. If mobile encoding is needed, the `compute_activity_map` uses standard convolutions and can be made mobile-friendly.

### 4.5 Just Noticeable Difference (JND) Masking (Priority: Medium) — Not Implemented

> **Decided against:** Strength annealing (softsign bounding with a schedule that tightens the residual magnitude over training) achieves similar PSNR control without the complexity of computing per-pixel JND thresholds. May revisit if pushing beyond 38 dB PSNR.

**Problem:** Perturbations exceed human visual perception thresholds in some regions.

**Solution:** Compute JND mask and constrain perturbations:

```python
class JNDMask(nn.Module):
    """Just Noticeable Difference mask for perceptually-guided embedding."""

    def __init__(self):
        super().__init__()
        # Luminance adaptation factor
        self.T0 = 17  # Base threshold

    def compute_luminance_mask(self, image: Tensor) -> Tensor:
        """Compute luminance-based JND threshold."""
        # Convert to grayscale (0-255 scale for standard JND formulas)
        lum = 255 * (0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2])

        # Luminance masking (Weber-Fechner law based)
        # Higher threshold in bright and dark regions
        lum_mask = torch.where(
            lum <= 127,
            self.T0 * (1 - (lum / 127) ** 0.5) + 3,
            (lum - 127) * 0.01 + 3
        )
        return lum_mask / 255  # Normalize back to [0, 1] scale

    def compute_contrast_mask(self, image: Tensor) -> Tensor:
        """Compute contrast/texture masking."""
        gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        gray = gray.unsqueeze(1)

        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)

        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        gradient_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        # Higher contrast = higher tolerance
        return gradient_mag.squeeze(1)

    def forward(self, image: Tensor) -> Tensor:
        """Compute combined JND mask."""
        lum_mask = self.compute_luminance_mask(image)
        contrast_mask = self.compute_contrast_mask(image)

        # Combined mask (multiplicative)
        jnd = (lum_mask + 0.1) * (1 + 2 * contrast_mask)
        return jnd.unsqueeze(1).expand(-1, 3, -1, -1)
```

**Usage in encoder:**
```python
jnd_mask = self.jnd.forward(image)
constrained_residual = torch.tanh(residual) * jnd_mask * 0.1
```

**Expected improvement:** Mathematically constrains perturbations to be below human perception thresholds.

> **Mobile compatibility:** ⚠️ **Encoder-only, but affects decoder if YUV mode used** - JND masking is encoder-side. However, if combined with YUV embedding (4.6), the decoder must also work in YUV space. The JND computation itself uses only standard ops (conv2d, basic math) and is mobile-compatible if needed.

### 4.6 YUV/LAB Color Space Embedding (Priority: Medium) — Not Implemented

> **Decided against:** The grayscale residual approach (4.4) eliminates colour shifts by construction, making YUV embedding unnecessary. Since E_post outputs a single channel broadcast to all three RGB channels, there is no colour channel imbalance to solve.

**Problem:** RGB channels treated equally, but human vision is more sensitive to luminance.

**Solution:** Embed primarily in chrominance channels:

```python
def rgb_to_yuv(rgb: Tensor) -> Tensor:
    """Convert RGB to YUV color space."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.147 * r - 0.289 * g + 0.436 * b
    v = 0.615 * r - 0.515 * g - 0.100 * b
    return torch.stack([y, u, v], dim=1)

def yuv_to_rgb(yuv: Tensor) -> Tensor:
    """Convert YUV back to RGB."""
    y, u, v = yuv[:, 0], yuv[:, 1], yuv[:, 2]
    r = y + 1.140 * v
    g = y - 0.395 * u - 0.581 * v
    b = y + 2.032 * u
    return torch.stack([r, g, b], dim=1)

class YUVEncoder(nn.Module):
    """Encoder that works in YUV space with channel-specific weights."""

    def forward(self, image: Tensor, message: Tensor) -> Tensor:
        # Convert to YUV
        yuv = rgb_to_yuv(image)

        # ... encoder produces residual in YUV space ...
        residual_yuv = self.encoder_network(yuv, message)

        # Apply different scaling per channel
        # Y (luminance): minimal change, U/V (chrominance): more tolerance
        channel_weights = torch.tensor([0.3, 1.0, 1.0], device=image.device)
        scaled_residual = residual_yuv * channel_weights.view(1, 3, 1, 1)

        # Add and convert back
        encoded_yuv = yuv + scaled_residual
        return torch.clamp(yuv_to_rgb(encoded_yuv), 0, 1)
```

**Expected improvement:** Reduces luminance artifacts (most visible) while embedding more in chrominance (less visible).

> **Mobile compatibility:** ⚠️ **Affects both encoder AND decoder** - If the encoder embeds in YUV space, the decoder must also convert to YUV before decoding. RGB↔YUV conversion is simple matrix math and mobile-friendly. **Recommendation:** Train separate RGB and YUV decoder variants, or pre-convert images in the mobile app before feeding to decoder.

### 4.7 Bilinear/Bicubic Upsampling Throughout (Priority: Medium) — Partially Done

> **Not done in message expansion** (still uses nearest-neighbor for 50→image_size), but U-Net decoder upsampling in the encoder uses bilinear. The architecture works well regardless; message expansion artifacts are absorbed by the U-Net.

**Problem:** Multiple `mode="nearest"` upsampling operations create blocky patterns.

**Solution:** Replace all nearest-neighbor with bilinear or learned upsampling:

```python
# Before (creates block artifacts)
x = F.interpolate(x, scale_factor=2, mode="nearest")

# After (smooth upsampling)
x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

# Or learned upsampling (best quality)
self.upsample = nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    nn.Conv2d(ch, ch, 3, padding=1),
    nn.GroupNorm(8, ch),
    nn.LeakyReLU(0.2),
)
```

**Expected improvement:** Eliminates blocky artifacts throughout the encoder pathway.

> **Mobile compatibility:** ✅ **Fully compatible** - Bilinear interpolation (`F.interpolate(..., mode='bilinear')`) is well-supported on all mobile frameworks (Core ML, TFLite, ONNX). This is a pure win with no mobile penalty.

---

## 5. Training Strategy Improvements

### 5.1 Progressive Perceptual Training

**Current:** L2 and LPIPS ramp together over 20k steps.

**Improved schedule:**
```yaml
loss:
  # Phase 1 (0-10k): Focus on message embedding
  message:
    scale: 1.0
    ramp_steps: 1

  # Phase 2 (0-30k): Add L2 for basic image fidelity
  l2:
    scale: 2.0
    ramp_steps: 30000

  # Phase 3 (10k-50k): Add LPIPS for perceptual quality
  lpips:
    scale: 1.5
    ramp_steps: 40000
    delay_steps: 10000

  # Phase 4 (20k-60k): Add FFL for frequency artifacts
  ffl:
    scale: 0.1
    ramp_steps: 40000
    delay_steps: 20000
    alpha: 1.0

  # Phase 5 (30k+): Add adversarial loss
  gan:
    scale: 0.01
    ramp_steps: 50000
    delay_steps: 30000
```

### 5.2 Stronger Regularization

```yaml
training:
  # Gradient clipping for stability with GAN
  grad_clip: 1.0

  # Spectral normalization for discriminator
  spectral_norm: true

  # EMA for stable generator updates
  ema_decay: 0.999
```

### 5.3 Augmentation During Training

Add augmentations that teach the encoder to avoid detectable patterns:

```python
class EncoderAugmentation:
    """Augmentations applied to encoded images during training."""

    def __call__(self, encoded: Tensor) -> Tensor:
        # Random JPEG compression (teaches frequency robustness)
        if random.random() < 0.5:
            encoded = jpeg_compress(encoded, quality=random.randint(50, 95))

        # Random Gaussian blur (teaches spatial robustness)
        if random.random() < 0.3:
            encoded = gaussian_blur(encoded, sigma=random.uniform(0.5, 2.0))

        # Random noise (teaches noise robustness)
        if random.random() < 0.3:
            encoded = encoded + torch.randn_like(encoded) * 0.02

        return encoded
```

---

## 6. Mobile Compatibility Summary

| Improvement | Encoder | Decoder | Mobile Status | Implementation |
|-------------|---------|---------|---------------|----------------|
| 4.1 Improved Message Expansion | ✓ | - | ⚠️ Encoder-only | Not done (future work) |
| 4.2 Focal Frequency Loss | Training | Training | ✅ Training-only | **DONE** |
| 4.3 Adversarial Training | Training | Training | ✅ Training-only | **DONE** (WGAN) |
| 4.4 Content-Adaptive Scaling | ✓ | - | ⚠️ Encoder-only | **Solved differently** (grayscale residual) |
| 4.5 JND Masking | ✓ | - | ⚠️ Encoder-only | Not done (strength annealing suffices) |
| 4.6 YUV Color Space | ✓ | ✓ | ⚠️ Both affected | Not done (grayscale residual makes unnecessary) |
| 4.7 Bilinear Upsampling | ✓ | - | ✅ Fully compatible | Partially done (U-Net only) |

**Key insight (validated):** As predicted, the most impactful improvements were training-only (FFL, WGAN) and encoder-side (grayscale residual, softsign bounding). No mobile decoder changes were required. YUV embedding (4.6) turned out to be unnecessary because the grayscale residual approach eliminates colour shifts by construction.

---

## 7. Implementation Roadmap

### Phase 1: Quick Wins (1-2 training runs)
1. **Replace nearest-neighbor with bilinear** in all `F.interpolate` calls — Partially done (U-Net decoder, not message expansion)
2. **Add Focal Frequency Loss** to existing training — **DONE** (scale 1.0, ramp 50k, delay 30k)
3. **Reduce residual magnitude** with tanh scaling — **Solved differently** (softsign bounding + strength annealing)

### Phase 2: Architectural Improvements (2-3 training runs)
4. **Implement ImprovedMessageExpander** with learned upsampling — Not done (future work)
5. **Add content-adaptive residual scaling** — **Solved differently** (grayscale residual + border falloff)
6. **Train in YUV space** with channel-specific weights — Not implemented (grayscale residual makes this unnecessary)

### Phase 3: Adversarial Training (3-5 training runs)
7. **Add PatchDiscriminator** with WGAN-GP — **DONE** (WGAN, LR 1e-5, g_scale 1.0, ramp 20k)
8. **Implement progressive training schedule** — **DONE** (phased loss ramp with delays)
9. **Add JND masking** as constraint — Not implemented (strength annealing suffices for now)

### Phase 4: Validation & Refinement
10. **A/B testing** of each improvement — Ongoing
11. **Perceptual user study** for artifact visibility — Not yet done
12. **Robustness testing** to ensure decoding accuracy maintained — **DONE** (98.6% at JPEG Q10)

### Phase 5: Mobile Validation (Required)
13. **Export improved decoder to Core ML/TFLite** - Verify all ops supported
14. **Benchmark mobile inference** - Target < 100ms on iPhone 12+
15. **Validate bit accuracy on mobile** - Target < 2% degradation vs. server
16. ~~**Test YUV decoder variant** if Phase 2.6 implemented~~ — No longer needed

---

## 8. Expected Outcomes vs. Actual Results

### Quality Metrics

| Metric | v1 Baseline | Target | v2 Actual | Status |
|--------|-------------|--------|-----------|--------|
| PSNR | 26.54 dB | >38 dB | 32.82 dB | +6.3 dB improvement, not yet at target |
| SSIM | ~0.92 | >0.97 | — | Not measured yet |
| LPIPS | ~0.08 | <0.03 | — | Not measured yet |
| Bit accuracy (clean) | 99.8% | 99%+ | 98.4% | Slight trade-off for imperceptibility |
| Bit accuracy (JPEG Q10) | 99.4% | >90% | 98.6% | Exceeds target |
| Colour shifts | Visible | None | None | Solved via grayscale residual |
| Visual artifact score | Obvious | Imperceptible | Greatly reduced | Border falloff + strength annealing |

**Analysis:** The +6.3 dB PSNR gain comes from the combination of FFL, WGAN adversarial training, grayscale residual, softsign bounding with strength annealing, and border falloff. The remaining gap to 38 dB may require further strength annealing tightening, improved message expansion (4.1), or JND masking (4.5).

### Mobile Deployment Metrics

| Metric | Target (iPhone 12+) | Target (Android Flagship) | Notes |
|--------|---------------------|--------------------------|-------|
| Decoder params | < 500K | < 500K | See [mobile_deployment.md](./mobile_deployment.md) |
| Model size | < 2 MB (INT8) | < 2 MB (INT8) | Post-quantization |
| Inference time | < 100ms | < 150ms | 400x400 input |
| Bit accuracy (mobile) | > 97% | > 97% | < 2% degradation vs. server |
| Mobile ops compatibility | 100% | 100% | No unsupported ops |

---

## 9. References

### Papers
- [Focal Frequency Loss for Image Reconstruction and Synthesis](https://github.com/EndlessSora/focal-frequency-loss) - Jiang et al., ICCV 2021
- [InvisMark: Invisible and Robust Watermarking for AI-generated Image Provenance](https://arxiv.org/html/2411.07795v1) - Uses FFL + GAN for imperceptibility
- [FreqMark: Invisible Image Watermarking via Frequency Based Optimization](https://arxiv.org/abs/2410.20824) - NeurIPS 2024
- [HiDDeN: Hiding Data With Deep Networks](https://openaccess.thecvf.com/content_ECCV_2018/papers/Jiren_Zhu_HiDDeN_Hiding_Data_ECCV_2018_paper.pdf) - Zhu et al., ECCV 2018
- [NeurIPS 2024 Invisible Watermark Challenge](https://arxiv.org/abs/2508.21072) - Insights on watermark robustness

### Related Project Documents
- [Decoder Architecture Analysis](./stegastamp-decoder-architecture-analysis.md) - Why decoder simplicity is justified
- [Mobile Deployment Guide](./mobile_deployment.md) - Constraints for mobile-optimized models

---

## 10. Appendix: Quick Test Script

```python
"""Quick visual comparison of improvement techniques."""
import torch
from picode.models.stegastamp import Encoder, Decoder

def visualize_residual(encoder, image, message):
    """Generate and visualize encoding residual."""
    encoded = encoder(image, message)
    residual = encoded - image

    # Amplify for visualization
    residual_vis = (residual - residual.min()) / (residual.max() - residual.min())

    # Compute frequency spectrum
    residual_gray = residual.mean(dim=1, keepdim=True)
    spectrum = torch.fft.fftshift(torch.fft.fft2(residual_gray))
    spectrum_mag = torch.log(torch.abs(spectrum) + 1e-8)

    return {
        'encoded': encoded,
        'residual': residual_vis,
        'spectrum': spectrum_mag,
        'psnr': 10 * torch.log10(1 / ((encoded - image) ** 2).mean()),
    }
```

---

*Document created: 2026-03-03*
*Updated: 2026-03-04 - Added mobile deployment constraints and compatibility analysis*
*Updated: 2026-06-04 - Added implementation status, actual results from PicoTrust v2 training*
*Status: Partially implemented. FFL (4.2) and WGAN (4.3) done. Colour/residual issues (4.4, 4.6) solved via grayscale residual + softsign bounding. Remaining items (4.1, 4.5) are future work.*
