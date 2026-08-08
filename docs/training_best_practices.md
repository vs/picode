# PicoTrust Training Best Practices

Distilled from 12 model versions and 27 training lessons. This is the recipe for building a production-quality steganography model from scratch.

> **Note:** this recipe is the blur-based v12 approach. The current production model
> (b72s20m85) replaces decoder/perceptual blur with a Sobel texture mask in training — see
> [sobel_adaptive.md](sobel_adaptive.md) and `picode-model/configs/picotrust_b72s20m85.yaml`.

## Recommended Architecture

| Component | Value | Why |
|-----------|-------|-----|
| Encoder | 512×512, U-Net + dilated E_post | Bilinear message upsampling, 1-channel grayscale residual, zero-init E_post |
| Decoder | 416×416 | Sweet spot: enough resolution for content-adaptive decoding, mild low-pass from downsampling |
| Bits | 64 | Good capacity for short links/IDs. 32 bits bootstraps easier but less payload. 80+ bits is harder to bootstrap |
| Residual bound | Softsign: `strength × r/(1+|r|)` | Never saturates, always has gradient (unlike tanh) |
| Discriminator | WGAN PatchGAN | LR = 0.15× encoder LR (conservative to prevent collapse) |

## Recommended Training Config

```yaml
training:
  num_steps: 200000
  lr: 0.0001
  num_bits: 64
  image_size: 512
  residual_strength: 1.0
  residual_strength_anneal_target: 0.014
  residual_strength_anneal_start: 10000
  residual_strength_anneal_steps: 120000
  anneal_schedule: exponential        # not linear
  decoder_blur_sigma: 0.8             # decoder-side only, not encoder
  no_im_loss_steps: 10000
  phase2_step: 60000
  phase2_decoder_lr_scale: 0.1

loss:
  message: { scale: 5.0, ramp_steps: 1 }
  l2: { scale: 1.5, ramp_steps: 50000 }
  lpips: { scale: 1.5, ramp_steps: 50000 }
  ffl: { scale: 1.0, ramp_steps: 50000, delay_steps: 0 }
  message_loss_type: mse
  gan_config:
    enabled: true
    discriminator_lr: 0.000015
    g_loss_scale: 1.5
    g_loss_ramp_steps: 20000
```

## Training Phases

### 1. Bootstrap (steps 0-10k)
- Strength = 1.0 (unbounded), message loss only, no image losses
- Encoder-decoder establish communication
- **No blur** during bootstrap — blur prevents bootstrapping
- Monitor `decoder_prob_std`: must rise above 0.05 by step 2000
- If collapsed (`prob_std < 0.02`), kill and restart with new random seed
- Bootstrap success rate: ~50% at 32 bits, ~9% at 64 bits. Use auto-retry loop

### 2. Squeeze (steps 10k-130k)
- Exponential strength annealing: `strength = 1.0 × (0.014)^progress`
- Image losses (L2, LPIPS, FFL, GAN) ramp in over 50k steps
- Decoder-side blur (σ=0.8) active — teaches decoder to use low-frequency patterns
- LPIPS and GAN receive blurred encoded image for content-adaptive gradients
- L2 and FFL receive the real encoded image for pixel-level quality
- Phase 2 at step 60k: decoder LR drops to 0.1× (stabilizes decoder)

### 3. Fine-tuning (steps 130k-200k)
- Strength fixed at target (0.014)
- All losses active, model converges
- Accuracy recovers from squeeze dip

### 4. De-annealing (optional, steps 200k+)
- Resume training with lower target strength (e.g., 0.014→0.010)
- Linear anneal, ~15k steps per 0.001 reduction
- Trades accuracy for PSNR: each 0.001 costs ~1-2% accuracy, gains ~0.5-1.0 dB
- Can push to 40+ dB PSNR at 64 bits

## Key Design Decisions

### Decoder-side blur + blurred LPIPS (content-adaptive clean output)

The central innovation of v12. Three separate uses of Gaussian blur (σ=0.8):

1. **Decoder input**: blurred during training so decoder learns low-frequency decoding
2. **LPIPS input**: blurred so LPIPS sees large-scale patterns and penalizes them more in smooth image regions → content-adaptive gradients
3. **GAN input**: blurred so discriminator judges at the same scale as LPIPS

The encoder's actual output is never blurred — clean, sharp images at inference.

### Exponential annealing

`strength = initial × (target/initial)^progress`

For 1.0→0.014, linear annealing is at 0.507 halfway through. Exponential is at 0.118 — much closer to the target, spending more time in the critical low-strength regime where the encoder needs to learn efficient encoding.

### Loss balance

Message loss (5.0) must dominate total image losses (~4.5). This ensures the encoder prioritizes message capacity. If image losses dominate, the encoder minimizes residual at the expense of accuracy.

LPIPS and GAN at 1.5× (not 1.0×) enables content-adaptive encoding. The higher perceptual weights teach the encoder to concentrate residuals in textured regions.

### MSE message loss (not BCE)

BCE with logits has a trivial equilibrium at 0.5 (all logits = 0). MSE has stronger gradients away from 0.5 and no stable trivial solution.

## Production Inference

### Adaptive strength per image

The model supports any strength at inference via the softsign bound. Recipe:

1. Encode at trained strength (0.014)
2. Decode and check accuracy
3. If accuracy > 98%: re-encode at lower strength (0.010-0.012) for better PSNR
4. If accuracy < 90%: re-encode at higher strength (0.016-0.020) for reliability

### Texture masking

Post-processing that attenuates residual in smooth regions:

```bash
picode -c checkpoint.pt encode input.jpg output.png -m "message" --texture-mask --mask-floor 0.5
```

Gains +1.4 dB PSNR at floor=0.5. Pair with LDPC soft decoding to tolerate the accuracy drop.

### Error correction

Use LDPC soft decoding with decoder logit probabilities. At 64 raw bits with LDPC, expect ~32-40 payload bits with near-perfect recovery on clean images and strong robustness to JPEG/blur.

## What NOT to Do

- Don't use encoder-side blur — prevents bootstrap
- Don't use linear annealing for large strength ranges — wastes training time
- Don't use BCE message loss — trivial equilibrium at 0.5
- Don't use 256 decoder if you want content-adaptive encoding — too coarse
- Don't use 512 decoder — allows HF artifacts, worse JPEG robustness
- Don't set GAN LR > encoder LR — causes training collapse
- Don't use Laplacian loss — fades to irrelevance under strength annealing
- Don't skip bootstrap monitoring — 50-90% of runs collapse silently
