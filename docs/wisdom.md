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

## PicoTrust Training Lessons (2026-05-26 — 2026-06-04)

PicoTrust is a 512x512 encoder (U-Net + E_post refinement) with bilinear downsample to 256x256 decoder. These lessons come from two months of training iterations across v1 and v2.

### Lesson 1: Bounded Residuals Kill Bootstrap

Applying tanh or softsign bounds from step 0 prevents the encoder-decoder pair from bootstrapping. The encoder needs large residuals initially to create patterns distinguishable from decoder noise — bounding forces small residuals that look like random noise to an untrained decoder.

**Fix:** Strength annealing — start with strength=1.0 (effectively unbounded), anneal to target (0.03) after bootstrap:
```yaml
residual_strength: 1.0
residual_strength_anneal_target: 0.03
residual_strength_anneal_start: 10000
residual_strength_anneal_steps: 60000
```

### Lesson 2: Grayscale Residual Eliminates Colour Shifts

3-channel E_post output creates R!=G!=B residuals causing visible colour shifts even at small amplitudes. Chroma loss doesn't fully fix this — it reduces shifts but doesn't eliminate them.

**Fix:** E_post final layer outputs 1 channel (`Conv(16, 1, 1)`), broadcast to 3 channels. R=G=B by construction. No colour loss term needed.

### Lesson 3: Softsign > Tanh for Residual Bounding

Tanh gradient approaches 0 for large |x|. Once weights grow during training, gradients vanish and the encoder stops learning.

**Fix:** Use softsign: `strength * x / (1 + |x|)`. Gradient `1/(1+|x|)^2` is small but never zero — the encoder always receives some learning signal.

### Lesson 4: Zero-Init E_post Last Layer

Kaiming-init E_post produces O(1) activations at step 0, pushing into saturation immediately. This means the residual starts large and random, fighting the image losses from step 0.

**Fix:** Zero-init final Conv2d weight and bias. Residual starts at exactly zero, then grows as the network learns.

### Lesson 5: Checkpoint Stores Config, Not Runtime Values

Checkpoint saves `residual_strength: 1.0` (initial config), not the annealed value at the saved step. Resuming with the saved config value resets annealing.

**Fix:** Recompute annealed strength from step: `t = min((step - start) / steps, 1.0); strength = initial + t * (target - initial)`.

### Lesson 6: ResNet50 Decoder Cannot Do Steganography

`AdaptiveAvgPool2d(1)` destroys ALL spatial information. Steganographic signals are per-pixel — the decoder needs spatial awareness. StegaStamp CNN decoder with flatten from spatial feature maps (8x8) learned in 300 steps what ResNet50 couldn't in 114K.

**Takeaway:** Never use global average pooling in a steganography decoder. Use flatten from spatial feature maps.

### Lesson 7: MSE > BCE for Message Loss

BCE has a trivial equilibrium at 0.5 (logit=0 for all bits). The decoder can sit at this stable point and never move. MSE has stronger gradients away from 0.5 and no stable trivial solution — the gradient is always proportional to the error.

### Lesson 8: Learned Spatial Masks Collapse Under Regularization

`mask_reg` penalty pushes the learned mask toward 0, reducing effective residual to near-zero. At `mask_mean=0.03`, effective signal is ~0.003 — the decoder can't learn from such a faint signal.

**Fix:** Don't use learned masks with bounded residuals. The bound itself is the quality guarantee — adding a learned mask on top creates redundant and conflicting constraints.

### Lesson 9: message.scale Must Dominate Total Image Losses

Working ratio: `message.scale=5.0` vs ~3.5 total image losses. If image losses dominate, the encoder minimizes the residual to satisfy L2/LPIPS, and the decoder starves.

### Lesson 10: Black Border Inflates residual_abs_max in Logs

`residual = encoded - images` includes border regions where `encoded=0` and `images=original`, giving `max=1.0`. This is misleading — the border is not part of the encoded content.

**Fix:** Check `residual_mean` and `residual_std` instead of `residual_abs_max` for meaningful residual magnitude diagnostics.

### Lesson 11: Conservative GAN LR

PatchGAN discriminator LR of 0.0002 (20x encoder LR) causes training collapse. The discriminator overpowers the encoder, and the encoder can't embed messages while also fooling the discriminator.

**Fix:** Use LR 1e-5 (0.1x encoder LR) for stable adversarial training. The discriminator should provide gentle guidance, not dominate.

### PicoTrust Best Results

| Metric | v1 | v2 |
|--------|-----|-----|
| PSNR | 26.54 dB | 32.82 dB |
| Bit Accuracy | 99.8% | 98.4% |
| JPEG Q10 | 99.4% | 98.6% |
| Colour Shifts | Yes | None |

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

---

## PicoGrain: Why Amplitude-Modulated Noise Failed (2026-07-29)

PicoGrain encoded the message in a smooth amplitude envelope multiplied by a
freshly-resampled zero-mean Gaussian noise carrier, and reused the PicoTrust
CNN+STN decoder unchanged. It collapsed at step 2000 on Modal:
`decoder_prob_std=0.0036`, bit accuracy at chance, `loss_msg` pinned at 0.2499
(exactly the MSE of predicting 0.5 for every bit).

The design was abandoned. The code remains in the tree as a documented dead
end: `picode/models/picograin/`, `picode/training/grain_losses.py`,
`picode/training/grain_discriminator.py`, `configs/picograin_b127.yaml`.

### Lesson: Linear convolutions cannot estimate the amplitude of a zero-mean random field

The message lived in `|envelope|`; the carrier was `randn_like(envelope)`,
resampled every forward pass. Convolution is linear, so over any receptive
field `Σ wᵢ·envelopeᵢ·noiseᵢ` has expectation ≈ 0 no matter how large the
envelope is — the noise is zero-mean and independent of the weights.
Averaging zero-mean noise yields zero. The first conv layer's response carried
almost no information about the envelope, so the decoder's best strategy was to
predict the prior, and it did.

Recovering amplitude requires a **rectifying nonlinearity before spatial
pooling** — `x²`, `|x|`, or a structure tensor — to convert amplitude into
something a linear filter can average. ReLU does not count: it sits after the
first convolution, by which point linear mixing has already destroyed the
signal.

**Takeaway:** Never modulate a random carrier and expect a conv stack to read
the modulation. Either give the decoder an explicit energy front-end, or make
the carrier deterministic so decoding becomes correlation against a known
pattern (a linear, easily learned operation).

The original spec anticipated the symptom and prescribed the wrong cure —
"consider adding initial low-pass filtering layers to help envelope
extraction". Low-pass filtering zero-mean noise drives it toward zero. It has
to be rectify *then* smooth.

### Lesson: Check carrier frequency against the channel before training

Fine, pixel-scale grain does not survive print-then-photograph: halftone
screening and reduced capture resolution destroy texture at that scale. A
20-minute CPU experiment (synthesise grain, push it through a simulated print
chain, correlate the recovered band against the original field) measured
carrier survival by clump size:

| Grain clump | portrait | product | car |
|-------------|----------|---------|-----|
| 1 px        | 0.10     | 0.08    | 0.07 |
| 2 px        | 0.16     | 0.11    | 0.11 |
| 4 px        | 0.45     | 0.34    | 0.32 |
| 8 px        | 0.53     | 0.43    | 0.36 |
| 12 px       | 0.51     | 0.44    | 0.35 |

Sharp knee between 2px and 4px, plateau by 8px. The shipped design used
pixel-scale grain — around 8% carrier survival, i.e. noise.

Coarse grain is not a free fix: at 8px and above the texture stops reading as
film grain and becomes visible blotching, which looks like damage on skin.
4px was the only scale that was both survivable and attractive.

**Takeaway:** Carrier frequency versus channel bandwidth is cheap to measure
and expensive to get wrong. Run the numbers before spending GPU time. This
design had two independent fatal flaws — undecodable carrier, and a carrier
that would not survive the target channel anyway — and fixing the first would
only have exposed the second.

### Lesson: The bootstrap collapse detector pays for itself

`modal_train.py` halts training at step 2000 when `prob_std` falls below 0.02,
after a soft recovery attempt at 1500 (restore EMA weights, halve LR). It
caught this run in ~15 minutes for about $0.30 instead of burning the full
5.5-hour schedule. Keep it, and watch `decoder_prob_std` as the leading
indicator — it collapses well before bit accuracy makes the problem obvious.

---

## BCH vs LDPC at Short Block Lengths (2026-07-31)

Measured on `picotrust_b31_compositing` (step 140000, MIR Flickr, s=0.020,
Sobel mask floor 0.40). Both codes decode the **same** decoder output on the
same 50 images, so the only variable is the error-correction strategy.

- **BCH(31,16)** — hard decision, t=3, all 31 channel bits
- **LDPC(30,17)** — soft decision, 30 bits, multi-SNR ladder [2,3,5,8,10,12],
  accepting the first parity-passing decode

LDPC carries one *more* payload bit, so it is if anything favoured.

| Distortion | Raw acc | BCH(31,16) | LDPC(30,17) |
|------------|---------|------------|-------------|
| clean      | 99.1%   | 50/50      | 49/50 |
| jpeg50     | 99.4%   | 50/50      | 50/50 |
| jpeg25     | 98.3%   | 48/50      | 50/50 |
| jpeg10     | 87.7%   | 31/50      | 30/50 |
| blur 1.0   | 99.3%   | 50/50      | 49/50 |
| blur 2.0   | 98.5%   | 48/50      | 48/50 |
| noise 0.05 | 95.1%   | 45/50      | 48/50 |
| rescale 0.5| 99.1%   | 50/50      | 49/50 |
| combo      | 97.8%   | 48/50      | 50/50 |
| **total**  |         | **420/450**| **423/450** |

### Lesson: LDPC needs long codewords — use BCH at short block lengths

A 3-image difference out of 450 is a tie. The soft-decision advantage does not
materialise at n≈30, because belief propagation needs a long block to work:
short parity graphs have tight cycles and BP gets too few rounds of genuine
information mixing. LDPC wants codeword lengths in the hundreds.

Note `b72`'s LDPC(72,38) is also short by LDPC standards and may be leaving
performance on the table — worth measuring the same way.

Practical note: **pyldpc cannot construct n=31 at all.** `d_c` must divide `n`,
and 31 is prime. Any LDPC at this block length needs n=30 or n=32.

**Takeaway:** "Soft decision beats hard decision" is a statement about long
codes. At short block lengths, prefer BCH — it is simpler, faster (no SNR
ladder, no iteration), and more predictable.

### Lesson: LDPC's confident-but-wrong failure survives the SNR ladder

LDPC lost one image on **clean** input at 99.1% raw bit accuracy — a codeword
hard-decision BCH recovers trivially. The multi-SNR sweep mitigates the
pathology recorded in the LDPC SNR notes but does not eliminate it. BP can
still talk itself out of a nearly-perfect codeword.

The win/loss pattern is consistent with this: LDPC is ahead only in the middle
band (jpeg25, noise, combo: +7 combined) where errors are numerous but soft
values still carry information, and behind on easy cases (-4) where there is
nothing to gain and only the pathology to lose.

### Lesson: Separate channel limits from ECC limits before optimising the code

JPEG Q10 fails for both codes (31/50 and 30/50) because raw accuracy drops to
87.7% — a 12.3% BER. BCH t=3 covers 9.7%; LDPC(30,17) at rate 0.567 cannot
reliably cover 12.3% either. Both fail on the same images for the same reason:
not enough information survives in the signal.

No decoder change fixes this. It needs a stronger model or a lower-rate code
(fewer payload bits, more redundancy). Compute the BER your channel produces
and compare it to the code's correction capacity *before* assuming a smarter
decoder will help.

---

## PicoTrust Bootstrap Collapse Is Stochastic — Retry, Don't Redesign (2026-07-31)

`picotrust_b31_coco_print` collapsed at step 2000 on the first Modal attempt
(`prob_std=0.0029`). Relaunching the **identical config** produced a healthy run
that hit 99% bit accuracy by step 1900. Nothing was changed between the two
launches except the random initialisation.

Bootstrap collapse is a real and not-rare failure mode for this architecture.
That is why `scripts/modal_train_autoretry.sh` exists and defaults to 10
attempts. The correct response to a step-2000 collapse is to relaunch, not to
re-engineer the config.

### The step-2000 check is trustworthy — the separation is two orders of magnitude

Same config, same phase, both runs still inside `no_im_loss_steps=10000`:

| Step | Collapsed run | Healthy run |
|------|---------------|-------------|
| 1800 | acc 0.540, prob_std 0.0022, residual_mean 0.351 | acc 0.984, prob_std 0.497, residual_mean 0.048 |
| 1900 | acc 0.427, prob_std 0.0021, residual_mean 0.337 | acc 1.000, prob_std 0.498, residual_mean 0.116 |
| 2000 | acc ~0.5,  prob_std 0.0029, residual_mean ~0.34 | acc 0.992, prob_std 0.497, residual_mean -0.011 |

A healthy PicoTrust run reaches near-perfect accuracy **long before** image
losses engage at step 10000. `prob_std` separates the two cases by ~200x
against a 0.02 threshold, so the check is cheap, early, and unambiguous.

**Do not move the bootstrap check later to "give the model more time".** It was
briefly changed to fire at `max(warmup_steps, no_im_loss_steps) + 1000` on the
theory that step 2000 sat too early in the unrestrained phase. The theory was
wrong, and the change only made failures take 31 minutes to detect instead of 6.
Reverted.

### Residual growth during no_im_loss is not by itself a collapse signal

Image losses are weighted to zero while `step < no_im_loss_steps`, so the
encoder is unrestrained and `residual_mean` wanders in both runs. The healthy
run fluctuated around 0.05-0.12 and re-centred; the collapsed run climbed to
0.35 and stayed. Magnitude alone does not discriminate — **watch
`decoder_prob_std`**, which is unambiguous.

### Meta-lesson: a consistent story is not a verified diagnosis

The collapse was initially explained by combining two true facts — the encoder
is unrestrained during `no_im_loss_steps`, and `Trainer.fit`'s own detector
deliberately waits until `max(warmup_steps, no_im_loss_steps)` before judging —
into a conclusion that the Modal gate was mis-calibrated. Every ingredient was
real and the story was coherent. It was still wrong.

The discriminating measurement was trivial and was simply never taken: *what
does a healthy run look like at step 2000?* One grep against the next run
answered it and demolished the theory. When a failure has a plausible
mechanism, find the observation that separates it from the alternatives before
acting on it — especially before changing shared infrastructure.

(The related observation that every earlier PicoTrust model was trained on GCE,
which has no such gate, is factually true and worth knowing, but it was not the
cause of this failure.)

---

## Print-to-Photo Distortions Are Redundant (2026-07-31)

`picotrust_b31_coco_print` was trained to survive mild print-then-photograph:
COCO train2017, strength 0.020, compositing, plus the five print-to-photo
distortions (resolution loss, shot noise, barrel, vignetting, chromatic
aberration) enabled for the first time. 140k steps, ~6.6h on an A10G.

**It came out worse than the model it was meant to beat.** Head-to-head against
`picotrust_b31_compositing` on the same 50 images, same Sobel mask, same
BCH(31,16), at the same strength:

| Gate | MIR Flickr (no print training) | COCO + print chain |
|------|-------------------------------|--------------------|
| clean | 49/50 | 49/50 |
| jpeg50 | 50/50 | 47/50 |
| jpeg25 | 49/50 | 47/50 |
| jpeg10 | 22/50 | 12/50 |
| blur 1.0 | 50/50 | 49/50 |
| blur 2.0 | 49/50 | 46/50 |
| noise 0.05 | 47/50 | 43/50 |
| rescale 0.5 | 50/50 | 49/50 |
| combo | 49/50 | 47/50 |
| resolution_loss | 50/50 | 49/50 |
| vignetting | 49/50 | 48/50 |
| barrel | 49/50 | 49/50 |
| chromatic_ab | 50/50 | 49/50 |
| print_chain | 49/50 | 47/50 |
| print+jpeg30 | 46/50 | 43/50 |
| **total** | **708/750** | **674/750** |

PSNR was identical (40.32 vs 40.35 dB); TRC slightly worse (0.600 vs 0.584).

### Lesson: the existing curriculum already covers mild print-to-photo

Read the MIR Flickr column on the print gates. That model **never saw a single
print distortion in training**, yet scores resolution_loss 50/50, chromatic_ab
50/50, vignetting 49/50, and the full print_chain 49/50.

Perspective, blur, JPEG, noise, brightness and rescale already confer the
invariances mild print-then-photograph needs. `resolution_loss` largely
duplicates rescale; `shot_noise` duplicates gaussian noise. The five additions
bought no new robustness while enlarging the distortion space the model had to
cover on the same capacity and step budget — over-augmentation, paid for out of
general accuracy. JPEG Q10 nearly halved (22/50 -> 12/50).

**Takeaway:** do not enable the print-to-photo distortions expecting a gain. If
you need mild print robustness, `picotrust_b31_compositing` already delivers it
at 49/50 through the print chain and 46/50 through print+JPEG30. Before adding
augmentation, first measure whether the existing model already handles the
target channel — augmentation is only worth its cost if there is a measured gap.

### Lesson: change one variable per training run

This run altered the dataset (MIR Flickr -> COCO) **and** the distortion set at
the same time. The regression is real and reproducible, but it cannot be
attributed to either cause. The clean design would have been to enable the
print chain while holding the dataset fixed.

A ~$7 run that produces an unattributable result is a poor trade against a ~$7
run that isolates one variable. Decide what the run is meant to prove, then
check that the config changes only one thing that could affect it.

(Ironically, the finding above — that the print distortions are redundant —
suggests the dataset may be the real cause of the regression. Still unknown.)
