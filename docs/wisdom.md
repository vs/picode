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

## Summary

**Why StegaStamp works:** Simple architecture with unobstructed signal flow from encoder to decoder.

**Why Picode failed:** "Improvements" (GroupNorm, ResBlocks, learned expansion, bounded outputs) all dampened or destroyed the message signal the decoder needs to learn.

**The lesson:** In steganography, simplicity wins. The message must flow from encoder to decoder without normalization, shortcuts, or constraints. Trust the proven architecture.
