# Model Architecture Fixes

This document describes issues found in the original encoder/decoder implementation and the fixes applied.

## Problem: loss_msg Not Decreasing

During training, `loss_msg` stayed stuck at ~0.693 (random guessing) regardless of training steps. This indicated the model wasn't learning to encode/decode messages.

## Root Causes

### 1. Vanishing Gradients Through Deep Network

The encoder-decoder pipeline has ~17 layers total:
- Encoder: 5 conv layers down + 4 up blocks + 2 output layers
- Decoder: 7 conv layers + 2 FC layers

With plain ReLU activations and no normalization, gradients vanished:

| Component | Gradient Sum |
|-----------|--------------|
| Encoder   | 0.0015       |
| Decoder   | 45.45        |

The encoder received almost no gradient signal, so it couldn't learn.

**Fix:** Added BatchNorm after every conv layer in both encoder and decoder.

```python
# Before (encoder ConvBlock)
def forward(self, x):
    return F.relu(self.conv(x))

# After
def forward(self, x):
    return F.relu(self.bn(self.conv(x)))
```

### 2. torch.clamp Kills Gradients

The encoder output used `torch.clamp(image + residual, 0.0, 1.0)`. When pixels hit 0 or 1, `clamp` has zero gradient, blocking backpropagation.

With random initialization, ~9% of pixels were clamped, creating "dead zones" where no learning could happen.

**Fix:** Replace clamp with bounded tanh:

```python
# Before
encoded = torch.clamp(image + residual, 0.0, 1.0)

# After
residual = 0.3 * torch.tanh(residual)  # Bounded to [-0.3, 0.3]
encoded = image + residual
```

This keeps residuals in a valid range while maintaining gradient flow.

### 3. Incorrect Loss Weighting

The original configuration used equal weights for message and image losses:
- `message_scale = 1.0`
- `l2_scale = 2.0`

This caused the encoder to minimize L2 loss by outputting the original image unchanged (trivial solution), rather than actually encoding the message.

**Fix:** StegaStamp uses ~7x higher message loss weight:

```python
# Before (config.py defaults)
message: LossRamp = field(default_factory=lambda: LossRamp(1.0, 1))
l2: LossRamp = field(default_factory=lambda: LossRamp(2.0, 20000))

# After
message: LossRamp = field(default_factory=lambda: LossRamp(7.0, 1))
l2: LossRamp = field(default_factory=lambda: LossRamp(1.0, 20000))
```

The high message loss weight forces the encoder to prioritize message encoding. The L2 loss then encourages making the modifications imperceptible, but only after the message is encoded.

## Results After Fixes

| Metric | Before | After |
|--------|--------|-------|
| Encoder gradient | 0.0015 | 0.044 |
| Gradient flow | Blocked at clamp | Full backprop |
| Loss balance | L2 dominates | Message prioritized |

## Training Requirements

Even with these fixes, proper training requires:

1. **GPU** - CPU training with 400x400 images is ~100x slower
2. **Many steps** - StegaStamp uses 140,000 steps, not 500
3. **Proper learning rate** - 1e-4 to 1e-3 with gradient clipping
4. **Curriculum learning** - Start without distortions, add them gradually

## Files Modified

- `picode/models/stegastamp/encoder.py` - Added BatchNorm, changed to tanh residual
- `picode/models/stegastamp/decoder.py` - Added BatchNorm to all conv layers
- `picode/training/config.py` - Updated default loss weights (message=7.0, l2=1.0)

## References

- StegaStamp paper: Tancik et al., "StegaStamp: Invisible Hyperlinks in Physical Photographs", CVPR 2020
- Loss weighting based on StegaStamp official implementation
