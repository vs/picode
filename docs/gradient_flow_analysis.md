# Gradient Flow Analysis

> **Note:** All file paths in this document are relative to `picode-model/`.

Investigation into gradient flow through the Picode steganography pipeline (encoder → distortions → decoder).

## What's Working

1. **All distortions are differentiable** - Every distortion (including JPEG with straight-through estimator) properly passes gradients
2. **Gradients flow through full pipeline** - From message loss → decoder → distortions → encoder
3. **Encoder and decoder both receive gradients** - Both networks are being updated
4. **STN initialization is intentional** - Zero weights create identity transform initially, then adapts after first gradient step

## Key Issues

### 1. Massive Gradient Attenuation Through Decoder (76x reduction)

```
decoded_logits: 0.002439 → after decoder backward: 0.000032
```

The decoder's deep architecture (5 conv layers + 2 FC) attenuates gradients significantly before they reach the distortion chain.

### 2. Encoder Gets Only 6% of Decoder Gradients

Even with NO distortions:

```
Enc/Dec gradient ratio: 0.062 (about 1/16)
```

This is an architectural issue, not distortion-related. The encoder-decoder architecture inherently has imbalanced gradients.

### 3. Training Duration

200 steps is nowhere near enough for convergence:

- StegaStamp paper trains for **140,000 steps**
- With random messages each step + stochastic distortions, convergence requires many iterations
- The 77% accuracy seen in short tests is early-stage learning, not a plateau

## Why Short Training Shows Stuck Loss

Training output showed:
```
step=0   | loss_msg=0.6990  (random guessing, BCE = log(2))
step=190 | loss_msg=0.6927  (still ~random)
```

This is because:
1. Only 200 steps with batch_size=2 = 400 examples total
2. Random messages each step makes learning harder
3. Multiple stochastic distortions compound variance
4. Encoder updates 16x slower than decoder due to gradient ratio

## Gradient Magnitude Through Pipeline

```
Layer                Grad Mean       Grad Max        Activation Mean
----------------------------------------------------------------------
decoded_logits       0.002439        0.003778        0.340531
perspective          0.000032        0.000306        0.800156
jpeg                 0.000021        0.000229        0.805982
contrast             0.000018        0.000229        0.812383
saturation           0.000017        0.000221        0.819742
brightness_hue       0.000016        0.000214        0.820025
noise                0.000011        0.000214        0.815668
encoded              0.000010        0.000214        0.817513
```

## Distortion Strength vs Gradient Ratio

```
Strength     Enc Grad Mean      Dec Grad Mean      Enc/Dec Ratio
---------------------------------------------------------------
0.0          0.061713           0.992936           0.062152
0.1          0.038267           1.886317           0.020287
0.3          0.035278           3.135886           0.011250
0.5          0.097568           3.488907           0.027965
1.0          0.056600           3.480100           0.016264
```

Stronger distortions generally worsen the gradient ratio, but even without distortions the encoder only receives ~6% of decoder gradients.

## Recommendations

1. **Train for 10,000+ steps** to see meaningful convergence
2. **Consider separate learning rates**: encoder LR = 10-30x decoder LR to compensate for gradient imbalance
3. **Use curriculum learning**: start with no/weak distortions, ramp up gradually
4. **Use GPU** for practical training (400x400 images are slow on CPU)

## Verification Scripts

The following scripts were created to analyze gradient flow:

- `scripts/verify_gradients.py` - Tests gradient flow through individual distortions and full pipeline
- `scripts/debug_stn_gradients.py` - Investigates STN gradient flow issue
- `scripts/debug_encoder_gradients.py` - Analyzes encoder gradient magnitude
- `scripts/test_longer_training.py` - Tests convergence with more training steps
