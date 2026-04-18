# Steganography Training Wisdom

Hard-learned lessons from training picode models. This document explains why StegaStamp trains successfully while picode variants failed to reduce message loss.

---

## The Core Problem

Multiple training runs with picode v1 and v2 models failed to achieve meaningful `loss_msg` reduction. The models would plateau early while StegaStamp converged normally. The root cause: **architectural complexity killed the gradient signal**.

---

## Why StegaStamp Works

StegaStamp succeeds because of deliberate simplicity:

| Component | StegaStamp Choice | Why It Works |
|-----------|-------------------|--------------|
| Normalization | None | Message signal passes through unfiltered |
| Activation | ReLU | Predictable gradient flow with He init |
| Message expansion | Linear → nearest-neighbor | Simple, no learnable parameters |
| Residual output | Unbounded | Full magnitude available for embedding |
| Decoder architecture | Sequential convolutions | Forces learning of message patterns |
| STN initialization | Identity matrix | Stable starting point |
| Weight init | He normal (fan_in) | Matches ReLU expectations |

### The Signal Path

```
Encoder embeds raw residual → Decoder sees unfiltered signal → Learns pattern
```

The decoder receives exactly what the encoder produces. No normalization, no dampening, no shortcuts.

---

## Why Picode v1 Failed

### GroupNorm: The Hidden Culprit

Picode v1 added GroupNorm throughout encoder and decoder for "training stability." This backfired catastrophically.

**What GroupNorm does:**
- Normalizes activations per-group toward mean=0, std=1
- Adaptive to batch content but independent of batch statistics

**Why this kills steganography:**
```
Encoder output residual (before norm): [0.05, 0.02, -0.01, 0.03, ...]
After GroupNorm in decoder:            [normalized noise with buried message bits]
```

The subtle perturbations that encode message bits get normalized away. The decoder sees noise instead of signal.

### ResBlocks in Decoder

Picode v1 added ResBlocks after each downsample for "gradient stability." This created a second problem:

- Skip connections allow the decoder to ignore message information
- Gradient flows through shortcuts instead of learning message patterns
- The decoder can achieve low loss without actually detecting the message

### LeakyReLU + GroupNorm Interaction

The combination of LeakyReLU (instead of ReLU) with GroupNorm created unstable gradient flow for weak message patterns. The negative slope of LeakyReLU combined with normalization produced noisy gradients.

---

## Why Picode v2 Failed

Picode v2 attempted to fix v1's problems but introduced new ones:

### tanh-Bounded Residuals

```python
bounded_residual = torch.tanh(raw_residual) * residual_scale
```

With tanh, output range is constrained to `[-1, 1]`. This:
- Constrains the embedding space too tightly
- Forces message into micro-perturbations
- Distortions (JPEG, blur, noise) destroy these subtle signals

Even with `residual_scale=1.0`, the constraint was too tight.

### Learned MessageExpander

Instead of simple nearest-neighbor upsampling:
```
5×5 → 25×25 → 100×100 → 400×400 (learned bilinear)
```

This added trainable parameters that must learn spatial distribution of message bits. The complexity didn't generalize - the expander learned patterns specific to training data.

### Loss Weight Wasn't Enough

Message loss weight was increased from 1.0 to 7.0, but this couldn't overcome architectural handicaps. You can't gradient-descend your way out of a signal that's been normalized away.

---

## Architecture Comparison

| Component | StegaStamp | Picode v1 | Picode v2 | Outcome |
|-----------|-----------|----------|----------|---------|
| Encoder norm | None | GroupNorm | GroupNorm | GroupNorm kills signal |
| Encoder activation | ReLU | LeakyReLU | LeakyReLU | ReLU is simpler |
| Message expansion | Simple | Simple | Learned | Simple works |
| Residual bounding | None | None | tanh | Unbounded works |
| Decoder STN | Identity init | Random init | None | Identity init works |
| Decoder ResBlocks | No | Yes | No | No ResBlocks works |
| Decoder norm | None | GroupNorm | None | No norm works |

---

## Training Dynamics

### StegaStamp Convergence Path
```
Step 0-500:      loss_msg decreases rapidly (decoder learns early patterns)
Step 500-20k:    loss_l2 ramps, encoder refines residual patterns
Step 20k-140k:   loss_lpips ramps, fine-tuning imperceptibility
Result:          Stable encoder/decoder pair
```

### Picode Convergence Failure
```
Step 0:          GroupNorm normalizes random features
Step 1-1000:     loss_msg decreases (lucky gradient alignment)
Step 1000-10k:   PLATEAU - decoder gradient becomes noisy:
                 - GroupNorm dampens message signal
                 - ResBlocks provide shortcuts that skip message
                 - LeakyReLU dead zones with low signal
Step 10000+:     No improvement, loss_msg stalls at ~0.5
```

---

## Key Insights

### 1. Steganography Requires Simplicity

Every "improvement" we tried (GroupNorm, ResBlocks, learned upsampling, bounded outputs) backfired. Steganography needs:
- Direct signal path from encoder to decoder
- No normalization layers
- No skip connections in decoder
- Simple, deterministic message expansion

### 2. Normalization Is Poison

Normalization layers (BatchNorm, GroupNorm, LayerNorm) are designed to stabilize training by controlling activation statistics. In steganography, **the message IS the activation statistics**. Normalizing them away destroys the signal.

### 3. Bounded Outputs Constrain Too Much

tanh or sigmoid bounded residuals seem reasonable for controlling image perturbation magnitude. But they force the message into a tiny signal range that distortions destroy. Let the loss functions (L2, LPIPS) control magnitude instead.

### 4. ResBlocks Enable Cheating

Skip connections are great for classification - they help gradient flow. In steganography decoders, they let the network ignore the message by routing gradients through shortcuts. Force sequential processing.

### 5. Loss Weight Can't Fix Architecture

Increasing message loss weight from 1.0 to 7.0 didn't help because the signal was normalized away before the decoder could learn from it. Architecture trumps loss tuning.

---

## The Fix: Match StegaStamp

Picode v3 was designed to match StegaStamp's architecture:
- No GroupNorm
- No tanh bounding
- Simple nearest-neighbor message expansion
- ReLU activations
- No ResBlocks in decoder
- Identity-initialized STN

This is the correct approach. Steganography is a solved problem architecturally - StegaStamp/HiDDeN works. Innovation should focus on:
- Better distortion robustness (training curriculum)
- Better imperceptibility (loss functions)
- Faster inference (model pruning/quantization)

Not on changing the fundamental encoder/decoder signal path.

---

## Training Bugs and Pitfalls (January 2026)

### LPIPS Loss Gradient Bug

The LPIPS loss was wrapped in `torch.no_grad()`:

```python
# BROKEN - no gradients flow through LPIPS
with torch.no_grad():
    loss = lpips_fn(orig_scaled, enc_scaled)
```

This completely blocked gradients from flowing back to the encoder through the perceptual loss. The LPIPS value was added to `total_loss`, but during backpropagation, the encoder received zero gradient signal from LPIPS.

**The fix:** Remove `torch.no_grad()`. The LPIPS network weights won't update anyway (it's in eval mode and not in the optimizer) - only the encoder receives gradients, which is what we want.

### Bit Capacity Limitation

The StegaStamp decoder architecture has a fundamental limitation: **it can reliably learn ~10-20 bits, but fails at 100 bits**.

**Why:** The decoder compresses 400×400 input to 13×13 feature maps before the final linear layer. With 100 bits in a 10×10 spatial grid, each bit region becomes ~1 pixel at the final conv layer - not enough spatial resolution to distinguish patterns.

| Bits | Spatial Grid | Patch Size | Final Feature Size | Result |
|------|-------------|------------|-------------------|--------|
| 10 | ~3×3 | ~130×130 | ~4×4 per patch | Works |
| 20 | ~5×4 | ~80×100 | ~2.5×3 per patch | Marginal |
| 100 | 10×10 | 40×40 | ~1×1 per patch | Fails |

**The original StegaStamp uses only 20 bits**, not 100. This is why their architecture works.

**Solutions:**
1. Reduce `num_bits` to 20-30 (like original)
2. Use error correction (BCH/LDPC) to encode fewer robust bits
3. Modify decoder to preserve more spatial resolution

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

**Why StegaStamp works:** Simple architecture with unobstructed signal flow from encoder to decoder.

**Why Picode failed:** "Improvements" (GroupNorm, ResBlocks, learned expansion, bounded outputs) all dampened or destroyed the message signal the decoder needs to learn.

**The lesson:** In steganography, simplicity wins. The message must flow from encoder to decoder without normalization, shortcuts, or constraints. Trust the proven architecture.

**Additional lessons (2026):**
- LPIPS must allow gradients through (no `torch.no_grad()`)
- Bit capacity is limited by decoder spatial resolution (~20 bits for 400×400 images)
- Watch for trivial solutions (loss_msg stuck at 0.693 for BCE or 0.25 for MSE)
- Balance message and image losses carefully to avoid degenerate encoder solutions

---

## TensorFlow vs PyTorch: The Critical Differences (February 2026)

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
