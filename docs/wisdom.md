# Steganography Training Wisdom

Hard-learned lessons from training steganography models with PyTorch.

---

## Training Dynamics

### StegaStamp Convergence Path
```
Step 0-500:      loss_msg decreases rapidly (decoder learns early patterns)
Step 500-20k:    loss_l2 ramps, encoder refines residual patterns
Step 20k-140k:   loss_lpips ramps, fine-tuning imperceptibility
Result:          Stable encoder/decoder pair
```

---

## Training Bugs and Pitfalls

### LPIPS Loss Gradient Bug

The LPIPS loss was wrapped in `torch.no_grad()`:

```python
# BROKEN - no gradients flow through LPIPS
with torch.no_grad():
    loss = lpips_fn(orig_scaled, enc_scaled)
```

This completely blocked gradients from flowing back to the encoder through the perceptual loss. The LPIPS value was added to `total_loss`, but during backpropagation, the encoder received zero gradient signal from LPIPS.

**The fix:** Remove `torch.no_grad()`. The LPIPS network weights won't update anyway (it's in eval mode and not in the optimizer) - only the encoder receives gradients, which is what we want.

### Encoder Learns Degenerate Solutions

Without proper loss balance, the encoder learns global shortcuts instead of spatially structured patterns:

```
Step 0:   residual_mean = 0.008  (random initialization)
Step 100: residual_mean = 0.432  (encoder brightens everything)
Step 200: residual_mean = -0.171 (encoder darkens everything)
```

The encoder discovers it can change global brightness, which the decoder learns to detect. This achieves ~65% accuracy but destroys the image.

**The tension:**
- Without L2 loss: Encoder produces degenerate patterns (global brightness)
- With L2 loss: Encoder produces near-zero residuals (nothing to decode)

The original StegaStamp balances this with a specific training schedule:
1. First 500 steps: message loss only (encoder learns to embed)
2. After 500 steps: add L2 loss with slow ramp (encoder refines while preserving image)

### Trivial Solution Detection

Watch for `loss_msg` stuck at these values:

| Loss Type | Trivial Value | Meaning |
|-----------|---------------|---------|
| BCE | 0.693 (log 2) | Decoder outputs 0.5 for all bits |
| MSE | 0.250 | Decoder outputs 0.5 for all bits |

If `loss_msg` plateaus at these values early in training and doesn't decrease, the encoder-decoder pair has collapsed to the trivial solution.

### Gradient Imbalance

The encoder receives ~70× weaker gradients than the decoder because it's farther from the loss in the computational graph:

```
Encoder mean gradient: 0.0016
Decoder mean gradient: 0.1087
```

This can cause the decoder to learn faster than the encoder can adapt, leading to unstable training.

**Potential fixes:**
- Use `encoder_lr_scale` to give encoder higher learning rate
- Use gradient clipping with different thresholds per network
- Detach encoder output periodically to let decoder "catch up"

---

## Summary

**Key lessons:**
- The STN with trainable linear parameters causes trivial solution collapse in PyTorch - freeze STN parameters early in training
- LPIPS must allow gradients through (no `torch.no_grad()`)
- Don't clamp encoder output - let loss functions control residual magnitude
- Watch for trivial solutions (loss_msg stuck at 0.693 for BCE or 0.25 for MSE)
- Balance message and image losses carefully to avoid degenerate encoder solutions

---

## TensorFlow vs PyTorch: The Critical Differences

Extensive debugging revealed why the PyTorch implementation collapsed to trivial solutions while the original TensorFlow implementation worked. The TensorFlow code was verified to train successfully with just 100 images, reaching 95% bit accuracy within 500 steps.

### The STN (Spatial Transformer Network) Problem

**Root cause:** The STN's trainable linear parameters (`stn_fc_weight` and `stn_fc_bias`) destabilize training when learned from scratch.

**Evidence from experiments:**

| Configuration | Bit Accuracy @ 500 steps | Works? |
|--------------|-------------------------|--------|
| Full STN trainable | ~50% (random) | ❌ No |
| STN linear frozen (W, b) | ~95% | ✅ Yes |
| Entire STN frozen | ~94% | ✅ Yes |
| No STN at all | ~96% | ✅ Yes |

**What happens with trainable STN:**
1. STN starts with identity transform (W=zeros, b=[1,0,0,0,1,0])
2. During backprop, large gradients flow through `grid_sample`
3. These gradients update W and b, causing non-identity transforms
4. Non-identity transforms confuse the decoder early in training
5. Decoder learns to output 0.5 for all bits (safe prediction = trivial solution)
6. Once collapsed, the network cannot recover

**Why TensorFlow doesn't have this problem:**
TensorFlow's spatial transformer implementation has different gradient characteristics. The TF `stn_transformer` we implemented uses a different coordinate rescaling formula in `bilinear_sampler`:

```python
# TF STN bilinear_sampler
x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
```

PyTorch's `F.grid_sample` with `align_corners=False` uses a different coordinate system. This affects gradient magnitudes through the spatial transform.

**The fix:**
Freeze the STN linear parameters during early training:
```python
decoder.stn_fc_weight.requires_grad = False
decoder.stn_fc_bias.requires_grad = False
```

Or use a much smaller learning rate for STN parameters.

### Encoder Output Clamping

**Issue:** PyTorch encoder clamped output to [0, 1], but TensorFlow does not.

```python
# PyTorch (WRONG)
encoded = image + residual
encoded = torch.clamp(encoded, 0, 1)  # Blocks gradients at boundaries

# TensorFlow (CORRECT)
residual_warped = encoder((secret_input, input_warped))
encoded_warped = residual_warped + input_warped  # No clamping!
```

**Why this matters:**
- Clamping blocks gradients when values hit 0 or 1
- Early in training, encoder may need to overshoot temporarily
- The L2 loss naturally penalizes out-of-range values without blocking gradients

**The fix:** Remove `torch.clamp()` from encoder. Let loss functions control the residual magnitude.

### Decoder Weight Initialization

**TensorFlow:** Uses `glorot_uniform` (Xavier) for decoder conv layers by default

**PyTorch (original):** Used `kaiming_normal` (He) for ALL layers including decoder

The decoder doesn't have ReLU at every layer (the STN params path has its own ReLU pattern), so He initialization may not be optimal. Xavier/Glorot is more appropriate for mixed architectures.

However, experiments showed this initialization difference alone doesn't cause the trivial solution - the STN is the main culprit.

### Training Loop Differences

**TensorFlow no_im_loss_steps:**
```python
if no_im_loss:
    # Train ONLY on secret_loss_op (unscaled)
    sess.run([train_secret_op, loss_op, global_step_tensor], feed_dict)
```

**PyTorch warmup_steps:**
```python
if self.global_step < warmup_steps:
    # Train on scaled message loss: 1.5 * loss_msg
    total_loss = msg_scale * losses["loss_msg"]
```

The scaling factor (1.5x vs 1.0x) shouldn't matter for optimization, but it changes the gradient magnitude.

### Summary of Required Fixes

1. **STN linear layer:** Freeze `stn_fc_weight` and `stn_fc_bias` for first ~5000 steps, or use 10x lower learning rate
2. **Encoder clamping:** Remove `torch.clamp(encoded, 0, 1)` from encoder forward pass
3. **Decoder initialization:** Consider using Xavier/Glorot instead of He for decoder layers

### Diagnostic Metrics to Watch

When debugging trivial solution collapse:

| Metric | Healthy | Collapsed |
|--------|---------|-----------|
| `decoder_prob_std` | > 0.1 | < 0.01 |
| `bit_accuracy` | Improving | ~50% |
| `loss_msg` | Decreasing | ~0.693 |
| `residual_mean` | Near 0 | Drifting |

The `decoder_prob_std` is the earliest indicator - if it collapses to near-zero within the first 10 steps, the network is heading for trivial solution.

### Full Comparison Table

| Aspect | TensorFlow Original | PyTorch (Fixed) |
|--------|-------------------|-----------------|
| Framework | TF 1.x compat mode | PyTorch 2.x |
| Encoder clamp | No | No (removed) |
| STN linear | Trainable | Frozen early |
| Decoder init | Glorot (default) | Kaiming (He) |
| Loss scaling | l2=1.5, lpips=1, msg=1 | Configurable |
| no_im_loss_steps | 500 (default) | Configurable |
| Result @ 500 steps | ~95% accuracy | ~95% accuracy (with fixes) |

---

## Message Spatial Expansion Artifacts

### The 32x32 Grid Problem (PicodeLite)

PicodeLite's visible blocky artifacts are **NOT** caused by bit-to-region mapping. Each message bit contributes to every spatial position through the linear layer. The artifacts come from **nearest-neighbor upsampling** of the 32x32 intermediate tensor to 512x512 — a 16x stretch that creates a visible grid pattern in the encoded residual.

---

## PicodeLite & PicodeFrame Training Techniques

### Ranked by Impact

| Technique | Impact | Source |
|-----------|--------|--------|
| STN LR scaling (0.01x) | Critical | PicodeLite STN collapse debugging |
| Single-channel residual (greyscale) | Critical | PicodeFrame color artifact fix |
| Distortion warmup (skip during first N steps) | High | PicodeLite trainer |
| no_im_loss_steps (message-only phase) | High | StegaStamp original |
| Loss ramping (L2/LPIPS over 15K-20K steps) | High | StegaStamp original |
| Flatten + FC decoder (not global avg pooling) | High | PicodeFrame decoder rewrite |
| Message scale >> image scale for frame models | High | PicodeFrame 82% plateau fix |
| Decoder bootstrap phase (warmup before warmup) | High | PicodeFrame training schedule |
| STN identity regularization loss | Medium | PicodeFrame |
| Encoder LR scaling | Medium | PicodeLite |
| Curriculum distortion strategy | Medium | StegaStamp |
| `_filter_compatible()` for arch migration | Medium | PicodeFrame checkpoint resume |

### STN Parameter Group Separation

Any model with STN (StegaStamp, PicodeFrame, PicoTrust) **must** have STN parameters (`stn_fc_weight`, `stn_fc_bias`) in a separate optimizer param group with ~100x lower learning rate. The Trainer checks `config.model.type in ("stegastamp", "picodeframe", "picotrust")` for this. Forgetting to add a new model type here will cause STN instability.

### Trainer Image Size Configuration

Models that use `ModelConfig.encoder_size` / `decoder_size` (PicodeLite, PicoTrust) must be handled in the Trainer's image size selection logic — the `else` branch uses `training.image_size` which may differ from the model config, causing shape mismatches at decode time.

---

## Diagnostic Quick Reference

### Trivial Solution Indicators

| Metric | Healthy | Collapsed |
|--------|---------|-----------|
| `decoder_prob_std` | > 0.1 | < 0.01 |
| `bit_accuracy` | Improving | ~50% |
| `loss_msg` (BCE) | Decreasing | ~0.693 |
| `loss_msg` (MSE) | Decreasing | ~0.250 |
| `residual_mean` | Near 0 | Drifting |

### Grid Artifact Diagnostic

To check if a model has grid artifacts, generate example images and inspect the residual (10x amplified):
```bash
picode -c checkpoint.pt encode input.jpg out.png -m "test" --save-residual residual.png
```
If the residual shows a regular grid pattern, the model is using spatial message expansion with nearest-neighbor upsampling.

---

## PicodeFrame Greyscale Residual Architecture

### Color Artifact Problem

The original 3-channel residual encoder (`Conv2d(32, 3, 1)`) produced visible color artifacts in the frame border — shifts in hue/saturation that were perceptible even at narrow frame widths. This happened because the network learned to use all three color channels independently, creating chromatic patterns that look unnatural.

### Single-Channel Residual Fix

**Architectural constraint beats loss tuning.** Instead of adding stronger color loss penalties (which only push the problem down, never eliminate it), output a single-channel residual and broadcast to RGB:

```python
# Output layers — single-channel residual (greyscale only)
self.residual = nn.Conv2d(32, 1, 1)  # Was Conv2d(32, 3, 1)

# Forward pass — broadcast to 3 channels
residual = self.residual(x).expand(-1, 3, -1, -1)  # (B,1,H,W) -> (B,3,H,W)
```

This **architecturally guarantees** greyscale-only modifications — no color loss term needed (set `frame_color_scale: 0.0`). The encoder can only change luminance, never hue or saturation.

**Capacity impact:** At 400×400 with 5% frame border, there are ~30,400 border pixels. With 127-bit messages, that's ~239 pixels per bit — more than enough redundancy even with single-channel encoding.

### BCH(127, 64) for 64-bit Payload

For URL shortener IDs, 64 bits provides 2^64 ≈ 1.8×10^19 unique IDs — sufficient for any practical use. BCH(127, 64) adds 63 parity bits for error correction:

- **Codeword length:** 127 bits (model's `num_bits`)
- **Payload:** 64 bits
- **Error correction:** t=10 (corrects up to 10 bit errors, i.e., 7.9% error rate)
- **Required raw bit accuracy:** ~92% for reliable ECC decoding

---

## PicodeFrame Training Plateau & Loss Rebalancing

### The 82% Plateau

With the greyscale residual architecture, training reached ~82% bit accuracy by step 60K and plateaued through step 92K. BCH(127,64) needs ~92% accuracy, so the model was stuck well below the usability threshold.

**Diagnosis:** Image losses (L2 and LPIPS) were suppressing the encoder's ability to create strong-enough patterns in the frame border. The frame region is only ~10% of total pixels, so even small L2/LPIPS weights create strong pressure to minimize the residual — directly competing with the message loss.

### Loss Rebalancing That Worked

| Parameter | Before (plateau) | After (rebalanced) |
|-----------|-------------------|---------------------|
| `message.scale` | 10.0 | 15.0 |
| `frame_l2_scale` | 1.0 | 0.5 |
| `frame_lpips_scale` | 0.5 | 0.25 |
| `num_steps` | 140,000 | 230,000 |

**Principle:** For frame-based models where the encoding region is small relative to the image, message loss must dominate. Image quality losses should be relaxed — the hard mask already guarantees zero modification of the center image, so frame quality is a secondary concern.

### Multi-Phase Training Schedule (PicodeFrame)

```
Steps 0-3K:       decoder_warmup — message loss only, no frame losses
                   (bootstrap decoder to recognize patterns)
Steps 3K-5K:      warmup — message loss only, no distortions
                   (encoder + decoder co-adapt on clean images)
Steps 5K-10K:     distortions ramp in, still no image losses
                   (model learns robustness before being penalized for quality)
Steps 10K+:       image losses (L2, LPIPS) start ramping in
                   (refine frame appearance after message path is established)
```

**Key insight:** The `decoder_warmup_steps` phase (before `warmup_steps`) trains only the decoder on message recovery. This bootstraps the decoder to recognize encoder patterns before the encoder starts adapting to distortions.

---

## Checkpoint Architecture Migration

### `strict=False` Doesn't Handle Shape Mismatches

PyTorch's `load_state_dict(state_dict, strict=False)` only handles **missing keys** and **unexpected keys**. If a key exists in both the checkpoint and the model but with **different shapes**, it still throws a `RuntimeError`.

This means you cannot use `strict=False` alone to migrate between architectures (e.g., 3-channel → 1-channel residual layer).

### `_filter_compatible()` Pattern

Pre-filter the state dict before loading to drop shape-mismatched keys:

```python
def _filter_compatible(
    state_dict: dict[str, Tensor], model: nn.Module,
) -> dict[str, Tensor]:
    model_state = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        elif k in model_state:
            print(f"  Skipping {k}: checkpoint {v.shape} != model {model_state[k].shape}")
    return filtered

# Usage: preserves all compatible weights, freshly initializes mismatched layers
model.load_state_dict(_filter_compatible(ckpt_state, model), strict=False)
```

This allows resuming training from an old architecture checkpoint — all compatible weights (U-Net layers, decoder CNN, STN) are restored, while only the changed layers (e.g., `residual` conv) are freshly initialized.

---

## Kaggle Training Workflow

### Dataset Caching Gotcha

Kaggle caches dataset versions aggressively. After uploading a new version of a dataset (e.g., `picode-source`), the kernel may still use the old cached version for several minutes. Symptoms: error tracebacks reference line numbers from old code.

**Workaround:** Wait 3-5 minutes between `kaggle datasets version` and `kaggle kernels push`. There is no explicit cache invalidation API.

### Kaggle Scripts

Use `scripts/kaggle_setup.sh` for all Kaggle operations:
```bash
./scripts/kaggle_setup.sh --model picodeframe upload-code   # Upload source dataset
./scripts/kaggle_setup.sh --model picodeframe push          # Push kernel
./scripts/kaggle_setup.sh --model picodeframe status        # Check kernel status
./scripts/kaggle_setup.sh --model picodeframe output        # Get kernel output/logs
./scripts/kaggle_setup.sh --model picodeframe resume        # Resume from checkpoint
```

### Checkpoint Resume Flow

1. Stop running kernel (from Kaggle UI — no CLI stop command)
2. Download latest checkpoint: `./scripts/kaggle_setup.sh --model picodeframe download-ckpts`
3. Upload checkpoint to dataset: update `kaggle_ckpts_clean/` and `kaggle datasets version`
4. Update config (e.g., increase `num_steps`, adjust loss weights)
5. Upload new code: `upload-code`
6. Push new kernel: `push`
