# Distortions Implementation Comparison

This document compares the distortion implementations between the original **StegaStamp** project (TensorFlow 1.x) and this project's **PyTorch** implementation. Each section explains the mathematical concepts, shows code from both projects, and suggests modern library alternatives.

## Table of Contents

1. [Overview](#overview)
2. [Blur Distortions](#blur-distortions)
3. [Noise Distortions](#noise-distortions)
4. [Color Distortions](#color-distortions)
5. [JPEG Compression](#jpeg-compression)
6. [Geometric Transforms](#geometric-transforms)
7. [Summary](#summary)

---

## Overview

### Why Distortions Matter for Training

When training a steganography model (hiding information in images), we need the decoder to recover hidden data even after the image has been modified. During training, we apply random distortions to simulate real-world conditions:

- **Printing and scanning** an image
- **Compression** when sharing on social media
- **Color adjustments** from different displays
- **Geometric changes** from camera angles

The key requirement: **all distortions must be differentiable** so gradients can flow backward through them during training.

### Framework Differences

| Aspect | StegaStamp (TensorFlow 1.x) | This Project (PyTorch) |
|--------|----------------------------|------------------------|
| Graph construction | Static (define-then-run) | Dynamic (define-by-run) |
| Tensor format | NHWC (batch, height, width, channels) | NCHW (batch, channels, height, width) |
| Convolution | `tf.nn.conv2d` | `F.conv2d` |
| Randomness | `tf.random.*` (graph ops) | `torch.rand*` (immediate) |

---

## Blur Distortions

Blur simulates loss of sharpness from camera motion, printing, or focus issues.

### Concept

A blur operation replaces each pixel with a weighted average of its neighbors. The **kernel** (a small matrix) defines these weights:

- **Identity kernel**: No blur (center pixel = 1, rest = 0)
- **Gaussian kernel**: Weights follow a bell curve, creating smooth blur
- **Motion/line kernel**: Blur along a specific direction

### StegaStamp Implementation

Located in `utils.py:8-43`:

```python
def random_blur_kernel(probs, N_blur, sigrange_gauss, sigrange_line, wmin_line):
    N = N_blur
    # Create coordinate grid centered at (0, 0)
    # Each position gets (x, y) coordinates relative to center
    coords = tf.to_float(tf.stack(tf.meshgrid(
        tf.range(N_blur), tf.range(N_blur), indexing='ij'), -1)) - (.5 * (N-1))

    # Manhattan distance from center (|x| + |y|)
    # Used to identify the center pixel and limit blur extent
    manhat = tf.reduce_sum(tf.abs(coords), -1)

    # IDENTITY KERNEL: Only center pixel has value 1
    # manhat < 0.5 is True only for the center pixel (0,0)
    vals_nothing = tf.to_float(manhat < .5)

    # GAUSSIAN KERNEL: Values decrease with distance from center
    # sigma controls the "spread" - larger sigma = more blur
    sig_gauss = tf.random.uniform([], sigrange_gauss[0], sigrange_gauss[1])
    # Gaussian formula: exp(-(x^2 + y^2) / (2 * sigma^2))
    vals_gauss = tf.exp(-tf.reduce_sum(coords**2, -1)/2./sig_gauss**2)

    # MOTION/LINE KERNEL: Blur along a random direction
    # theta is a random angle (0 to 2*pi radians)
    theta = tf.random_uniform([], 0, 2.*np.pi)
    # v is the unit vector in that direction
    v = tf.convert_to_tensor([tf.cos(theta), tf.sin(theta)])
    # dists measures distance perpendicular to blur direction
    dists = tf.reduce_sum(coords * v, -1)

    sig_line = tf.random.uniform([], sigrange_line[0], sigrange_line[1])
    w_line = tf.random.uniform([], wmin_line, .5 * (N-1) + .1)
    # Gaussian falloff perpendicular to line, limited by w_line
    vals_line = tf.exp(-dists**2/2./sig_line**2) * tf.to_float(manhat < w_line)

    # PROBABILISTIC SELECTION between the three kernel types
    t = tf.random_uniform([])
    vals = vals_nothing  # Start with identity
    # If t < probs[0] + probs[1], use line blur
    vals = tf.cond(t < probs[0]+probs[1], lambda: vals_line, lambda: vals)
    # If t < probs[0], use gaussian blur (overrides line)
    vals = tf.cond(t < probs[0], lambda: vals_gauss, lambda: vals)

    # Normalize so kernel sums to 1 (preserves image brightness)
    v = vals / tf.reduce_sum(vals)

    # Create 3-channel kernel for RGB (each channel blurred identically)
    z = tf.zeros_like(v)
    f = tf.reshape(tf.stack([v,z,z, z,v,z, z,z,v],-1), [N,N,3,3])
    return f
```

Applied in `models.py:139-142`:

```python
# Create random blur kernel (25% gaussian, 25% line, 50% identity)
f = utils.random_blur_kernel(probs=[.25,.25], N_blur=7,
                       sigrange_gauss=[1.,3.], sigrange_line=[.25,1.], wmin_line=3)
# Apply via convolution (SAME padding preserves image size)
encoded_image = tf.nn.conv2d(encoded_image, f, [1,1,1,1], padding='SAME')
```

### This Project's Implementation

Located in `distortions/blur.py`:

```python
class RandomBlur(Distortion):
    """Probabilistic blur matching StegaStamp's random_blur_kernel."""

    def _create_blur_kernel(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        N = self.kernel_size

        # Create coordinate grid centered at origin
        # Same as StegaStamp but using PyTorch's meshgrid
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(N, device=device, dtype=dtype),
                torch.arange(N, device=device, dtype=dtype),
                indexing="ij",
            ),
            dim=-1,
        ) - (0.5 * (N - 1))

        # Manhattan distance (same logic as StegaStamp)
        manhat = coords.abs().sum(dim=-1)

        # Identity kernel
        vals_nothing = (manhat < 0.5).float()

        # Gaussian kernel with random sigma in range [1.0, 3.0]
        sig_gauss = (
            self.sigma_gauss_range[0]
            + torch.rand(1, device=device).item()
            * (self.sigma_gauss_range[1] - self.sigma_gauss_range[0])
        )
        vals_gauss = torch.exp(-(coords**2).sum(dim=-1) / (2.0 * sig_gauss**2))

        # Motion blur kernel with random angle
        theta = torch.rand(1, device=device).item() * 2.0 * math.pi
        v = torch.tensor([math.cos(theta), math.sin(theta)], device=device, dtype=dtype)
        dists = (coords * v).sum(dim=-1)

        sig_line = (
            self.sigma_line_range[0]
            + torch.rand(1, device=device).item()
            * (self.sigma_line_range[1] - self.sigma_line_range[0])
        )
        w_line = 3.0 + torch.rand(1, device=device).item() * (0.5 * (N - 1) - 3.0 + 0.1)
        vals_line = torch.exp(-dists**2 / (2.0 * sig_line**2)) * (manhat < w_line).float()

        # Probabilistic selection (uses Python if/else instead of tf.cond)
        t = torch.rand(1, device=device).item()
        if t < self.prob_gauss:
            vals = vals_gauss
        elif t < self.prob_gauss + self.prob_line:
            vals = vals_line
        else:
            vals = vals_nothing

        vals = vals / vals.sum()

        # Shape for PyTorch conv2d with groups=3
        # (out_channels=3, in_channels=1, H, W)
        kernel = vals.unsqueeze(0).unsqueeze(0).expand(3, 1, N, N)
        return kernel

    def forward(self, x: Tensor) -> Tensor:
        kernel = self._create_blur_kernel(x.device, x.dtype)

        # SAME padding equivalent (reflect mode for better edges)
        pad = self.kernel_size // 2
        x_padded = F.pad(x, [pad, pad, pad, pad], mode="reflect")

        # groups=3 applies each kernel slice to corresponding channel
        blurred = F.conv2d(x_padded, kernel, groups=3)
        return blurred
```

### Key Differences

| Aspect | StegaStamp | This Project |
|--------|------------|--------------|
| Kernel shape | `[N, N, 3, 3]` (TF format) | `[3, 1, N, N]` (PyTorch groups) |
| Random selection | `tf.cond` (graph branching) | Python `if/else` (dynamic) |
| Padding | `padding='SAME'` (implicit) | `F.pad` with reflect mode |
| Per-image kernel | Static per batch | Dynamic per forward pass |

### Modern Library Alternative: Kornia

[Kornia](https://kornia.github.io/) provides GPU-accelerated, differentiable image operations:

```python
import kornia.filters as KF

# GAUSSIAN BLUR with random sigma
# Kornia handles kernel creation, padding, and convolution
class ModernGaussianBlur(nn.Module):
    def __init__(self, kernel_size=7, sigma_range=(1.0, 3.0)):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size)
        self.sigma_range = sigma_range

    def forward(self, x):
        # Sample random sigma for each batch
        sigma = torch.empty(x.size(0)).uniform_(*self.sigma_range)
        sigma = sigma.to(x.device)

        # Kornia's gaussian_blur2d is fully differentiable
        return KF.gaussian_blur2d(
            x,
            kernel_size=self.kernel_size,
            sigma=(sigma, sigma)  # Isotropic blur
        )

# MOTION BLUR with random angle
class ModernMotionBlur(nn.Module):
    def __init__(self, kernel_size=7, angle_range=(0, 360)):
        super().__init__()
        self.kernel_size = kernel_size
        self.angle_range = angle_range

    def forward(self, x):
        # Kornia's motion_blur is differentiable
        angle = torch.empty(x.size(0)).uniform_(*self.angle_range)
        direction = torch.zeros(x.size(0))  # No perpendicular motion

        return KF.motion_blur(
            x,
            kernel_size=self.kernel_size,
            angle=angle.to(x.device),
            direction=direction.to(x.device)
        )
```

**Why Kornia is better for training:**
1. **Tested and optimized**: Battle-tested implementations with proper gradient flow
2. **Cleaner API**: No manual kernel construction or padding calculations
3. **Batched parameters**: Can apply different blur to each image in a batch
4. **More blur types**: Includes median blur, box blur, bilateral filter, etc.

---

## Noise Distortions

Noise simulates sensor noise, compression artifacts, or transmission errors.

### Concept

Gaussian noise adds random values drawn from a normal distribution to each pixel. The **standard deviation (std)** controls noise intensity.

### StegaStamp Implementation

Located in `models.py:144-146`:

```python
# rnd_noise is ramped up during training (starts small, increases)
rnd_noise = tf.random.uniform([]) * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise

# Generate noise tensor with same shape as image
# mean=0 centers the noise around the original pixel values
# stddev controls how much the noise can deviate
noise = tf.random_normal(shape=tf.shape(encoded_image), mean=0.0, stddev=rnd_noise)

# Add noise to image
encoded_image = encoded_image + noise

# Clip to valid range [0, 1] (pixels can't be negative or > 1)
encoded_image = tf.clip_by_value(encoded_image, 0, 1)
```

### This Project's Implementation

Located in `distortions/noise.py`:

```python
class GaussianNoise(Distortion):
    """Add Gaussian noise matching StegaStamp implementation."""

    def __init__(self, intensity: float = 0.5, std: float = 0.02):
        super().__init__(intensity)
        self.std = std  # Default 0.02 matches StegaStamp's --rnd_noise

    def forward(self, x: Tensor) -> Tensor:
        if self.intensity == 0.0:
            return x

        # Scale std by intensity (like StegaStamp's ramp function)
        # This allows gradual increase during training
        effective_std = self.std * self.intensity

        # Generate noise: torch.randn_like creates same-shape tensor
        # from standard normal distribution (mean=0, std=1)
        # Multiply by effective_std to scale to desired range
        noise = torch.randn_like(x) * effective_std

        # Add noise and clamp (same as tf.clip_by_value)
        return torch.clamp(x + noise, 0.0, 1.0)
```

### Key Differences

| Aspect | StegaStamp | This Project |
|--------|------------|--------------|
| Std calculation | `random_uniform * ramp * base_std` | `base_std * intensity` |
| Random multiplier | Random per forward pass | Controlled by intensity |
| Implementation | TF ops | PyTorch ops |

The implementations are mathematically equivalent. The main difference is that StegaStamp adds a random multiplier (`tf.random.uniform([])`) before the base std, making noise variance doubly random. This project uses a deterministic intensity-based scaling for more controlled augmentation.

### Modern Library Alternative: Kornia

```python
import kornia.augmentation as K

class ModernGaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=0.02):
        super().__init__()
        # Kornia's RandomGaussianNoise handles everything
        self.noise = K.RandomGaussianNoise(
            mean=mean,
            std=std,
            p=1.0  # Always apply (probability = 1)
        )

    def forward(self, x):
        return self.noise(x)

# For training with variable noise levels:
class ModernAdaptiveNoise(nn.Module):
    def __init__(self, std_range=(0.0, 0.02)):
        super().__init__()
        self.std_range = std_range

    def forward(self, x, intensity=1.0):
        # Sample std based on training progress (intensity)
        max_std = self.std_range[1] * intensity
        std = torch.rand(1).item() * max_std

        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, 0.0, 1.0)
```

**Why Kornia is better:**
1. **Consistent API**: Same interface as other augmentations
2. **p parameter**: Built-in probability of applying the transform
3. **Batched operations**: Efficient GPU utilization

---

## Color Distortions

Color distortions simulate display variations, lighting conditions, and camera differences.

### Brightness and Hue (Combined)

#### Concept

- **Brightness**: Add/subtract a constant from all pixels (shift the whole image lighter/darker)
- **Hue shift (in RGB)**: Add different amounts to each color channel, creating color casts

StegaStamp combines these into one operation for efficiency.

#### StegaStamp Implementation

Located in `utils.py:77-80` and applied in `models.py:124-126, 152-153`:

```python
# utils.py - Generate random shifts
def get_rnd_brightness_tf(rnd_bri, rnd_hue, batch_size):
    # Per-channel shift (simulates hue rotation in RGB space)
    # Shape: (batch_size, 1, 1, 3) for broadcasting
    rnd_hue = tf.random.uniform((batch_size, 1, 1, 3), -rnd_hue, rnd_hue)

    # Global brightness shift (same for all channels)
    # Shape: (batch_size, 1, 1, 1) broadcasts to all pixels and channels
    rnd_brightness = tf.random.uniform((batch_size, 1, 1, 1), -rnd_bri, rnd_bri)

    # Combined shift: each pixel gets channel-specific + global adjustment
    return rnd_hue + rnd_brightness

# models.py - Apply the shifts
rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri  # Ramp up during training
rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
rnd_brightness = utils.get_rnd_brightness_tf(rnd_bri, rnd_hue, args.batch_size)
# ...
encoded_image = encoded_image + rnd_brightness  # Add to image
encoded_image = tf.clip_by_value(encoded_image, 0, 1)
```

#### This Project's Implementation

Located in `distortions/color.py`:

```python
class BrightnessHue(Distortion):
    """Adjust brightness and hue together, matching StegaStamp."""

    def __init__(self, intensity: float = 0.5, rnd_bri: float = 0.3, rnd_hue: float = 0.1):
        super().__init__(intensity)
        self.rnd_bri = rnd_bri  # Max brightness shift (default 0.3)
        self.rnd_hue = rnd_hue  # Max per-channel shift (default 0.1)

    def forward(self, x: Tensor) -> Tensor:
        if self.intensity == 0.0:
            return x

        b, c, h, w = x.shape
        device, dtype = x.device, x.dtype

        # Scale by intensity (replaces StegaStamp's ramp function)
        effective_hue = self.rnd_hue * self.intensity
        effective_bri = self.rnd_bri * self.intensity

        # Per-channel hue shift: (B, 3, 1, 1) - note NCHW vs NHWC
        # uniform(-1, 1) * range gives values in [-range, +range]
        rnd_hue = (torch.rand(b, c, 1, 1, device=device, dtype=dtype) * 2 - 1) * effective_hue

        # Global brightness shift: (B, 1, 1, 1)
        rnd_brightness = (torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * effective_bri

        # Add both and clamp
        adjusted = x + rnd_hue + rnd_brightness
        return torch.clamp(adjusted, 0.0, 1.0)
```

### Contrast

#### Concept

Contrast controls the difference between light and dark areas. Mathematically, we multiply pixel values by a scale factor:

- **scale < 1**: Reduces contrast (values compress toward middle gray)
- **scale > 1**: Increases contrast (values spread apart)
- **scale = 1**: No change

#### StegaStamp Implementation

Located in `models.py:148-151`:

```python
# Ramp contrast range during training
contrast_low = 1. - (1. - args.contrast_low) * ramp_fn(args.contrast_ramp)
contrast_high = 1. + (args.contrast_high - 1.) * ramp_fn(args.contrast_ramp)
contrast_params = [contrast_low, contrast_high]

# Sample different contrast for each image in batch
contrast_scale = tf.random_uniform(
    shape=[tf.shape(encoded_image)[0]],  # One value per batch item
    minval=contrast_params[0],
    maxval=contrast_params[1]
)
# Reshape for broadcasting: (batch, 1, 1, 1)
contrast_scale = tf.reshape(contrast_scale, shape=[tf.shape(encoded_image)[0], 1, 1, 1])

# Apply: simply multiply all pixel values
encoded_image = encoded_image * contrast_scale
```

#### This Project's Implementation

```python
class Contrast(Distortion):
    """Adjust image contrast matching StegaStamp."""

    def __init__(self, intensity: float = 0.5, contrast_low: float = 0.5, contrast_high: float = 1.5):
        super().__init__(intensity)
        self.contrast_low = contrast_low   # Minimum contrast (0.5 = half)
        self.contrast_high = contrast_high # Maximum contrast (1.5 = 50% boost)

    def forward(self, x: Tensor) -> Tensor:
        if self.intensity == 0.0:
            return x

        b = x.shape[0]

        # Interpolate range based on intensity (like StegaStamp's ramp)
        # At intensity=0: range is [1.0, 1.0] (no change)
        # At intensity=1: range is [contrast_low, contrast_high]
        effective_low = 1.0 - (1.0 - self.contrast_low) * self.intensity
        effective_high = 1.0 + (self.contrast_high - 1.0) * self.intensity

        # Sample contrast scale per batch element: (B, 1, 1, 1)
        contrast_scale = effective_low + torch.rand(
            b, 1, 1, 1, device=x.device, dtype=x.dtype
        ) * (effective_high - effective_low)

        # Simple multiplication (no clipping here - done after brightness)
        return x * contrast_scale
```

### Saturation

#### Concept

Saturation controls color intensity. The algorithm:

1. Convert image to grayscale (luminance only)
2. Blend between original (colorful) and grayscale

A **desaturation factor** of 0 keeps original colors, 1 makes it fully grayscale.

Note: StegaStamp's saturation is inverted from intuition - higher `rnd_sat` means MORE desaturation.

#### StegaStamp Implementation

Located in `models.py:156-157`:

```python
# Compute luminance (grayscale) using standard weights
# Human eyes are more sensitive to green, less to blue
# Weights: Red=0.3, Green=0.6, Blue=0.1
encoded_image_lum = tf.expand_dims(
    tf.reduce_sum(encoded_image * tf.constant([.3, .6, .1]), axis=3),  # Sum weighted channels
    3  # Add channel dimension back
)

# Lerp (linear interpolation) between color and grayscale
# (1 - rnd_sat) * color + rnd_sat * grayscale
# When rnd_sat=0: keep original color
# When rnd_sat=1: fully grayscale
encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum
```

#### This Project's Implementation

```python
class Saturation(Distortion):
    """Adjust image saturation matching StegaStamp."""

    def __init__(self, intensity: float = 0.5, rnd_sat: float = 1.0):
        super().__init__(intensity)
        self.rnd_sat = rnd_sat  # Max desaturation factor

    def forward(self, x: Tensor) -> Tensor:
        if self.intensity == 0.0:
            return x

        # StegaStamp luminance weights (same values)
        weights = torch.tensor([0.3, 0.6, 0.1], device=x.device, dtype=x.dtype)
        weights = weights.view(1, 3, 1, 1)  # Shape for NCHW format

        # Compute luminance: weighted sum across channels
        lum = (x * weights).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        lum = lum.expand_as(x)  # Broadcast to (B, 3, H, W)

        # Sample desaturation factor
        effective_sat = self.rnd_sat * self.intensity
        rnd_sat = torch.rand(1, device=x.device, dtype=x.dtype).item() * effective_sat

        # Lerp: same formula as StegaStamp
        adjusted = (1 - rnd_sat) * x + rnd_sat * lum
        return adjusted
```

### Key Differences

| Aspect | StegaStamp | This Project |
|--------|------------|--------------|
| Tensor format | NHWC | NCHW |
| Ramp mechanism | Training step based | Intensity parameter |
| Clamping | After all color ops | After each operation |

### Modern Library Alternative: Kornia

```python
import kornia.augmentation as K
import kornia.enhance as KE

class ModernColorDistortions(nn.Module):
    def __init__(self, brightness=0.3, contrast=(0.5, 1.5), saturation=(0.0, 1.0), hue=0.1):
        super().__init__()

        # ColorJitter combines all color augmentations
        self.jitter = K.ColorJitter(
            brightness=brightness,      # Additive shift
            contrast=contrast,          # (min, max) multiplier
            saturation=saturation,      # (min, max) - Kornia uses standard definition
            hue=hue,                    # Shift in [-hue, +hue]
            p=1.0
        )

    def forward(self, x):
        return self.jitter(x)

# Or use individual operations for more control:
class ModernSaturation(nn.Module):
    def forward(self, x, factor=0.5):
        # Kornia's adjust_saturation uses standard definition:
        # factor=0: grayscale, factor=1: original, factor>1: oversaturated
        return KE.adjust_saturation(x, factor)

class ModernContrast(nn.Module):
    def forward(self, x, factor=1.0):
        # Kornia's adjust_contrast centers around mean
        return KE.adjust_contrast(x, factor)
```

**Why Kornia is better:**
1. **Standard definitions**: Saturation uses intuitive 0-1-2 scale
2. **Proper color space handling**: Can work in HSV for true hue rotation
3. **Combined transforms**: ColorJitter applies all in one efficient operation
4. **Consistent API**: Same interface as blur, noise, etc.

---

## JPEG Compression

JPEG compression is crucial for training robust watermarks since images are often saved as JPEG.

### Concept

JPEG compression works in several stages:

1. **Color conversion**: RGB to YCbCr (separates brightness from color)
2. **Block splitting**: Divide image into 8x8 pixel blocks
3. **DCT (Discrete Cosine Transform)**: Convert spatial data to frequency data
4. **Quantization**: Divide by a matrix and round (this causes quality loss!)
5. **Inverse process**: Dequantize, inverse DCT, convert back to RGB

The challenge: **rounding is not differentiable** (gradient is zero everywhere). Both implementations use tricks to approximate gradients through the rounding step.

### StegaStamp Implementation

Located in `utils.py:83-378`:

```python
# RGB to YCbCr conversion (JPEG standard coefficients)
def rgb_to_ycbcr_jpeg(image):
    matrix = np.array(
        [[0.299, 0.587, 0.114],        # Y  (luminance)
         [-0.168736, -0.331264, 0.5],  # Cb (blue difference)
         [0.5, -0.418688, -0.081312]], # Cr (red difference)
        dtype=np.float32).T
    shift = [0., 128., 128.]  # Cb and Cr are centered at 128

    result = tf.tensordot(image, matrix, axes=1) + shift
    return result

# DCT using precomputed tensor (for efficiency)
def dct_8x8(image):
    image = image - 128  # Center around zero

    # Precompute DCT basis functions
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        # DCT formula: cos((2x+1)*u*pi/16) * cos((2y+1)*v*pi/16)
        tensor[x, y, u, v] = np.cos((2*x+1) * u * np.pi / 16) * \
                              np.cos((2*y+1) * v * np.pi / 16)

    # Apply normalization factors
    alpha = np.array([1./np.sqrt(2)] + [1]*7)  # First coefficient scaled
    scale = np.outer(alpha, alpha) * 0.25

    # Tensor contraction applies DCT to all blocks at once
    result = scale * tf.tensordot(image, tensor, axes=2)
    return result

# DIFFERENTIABLE ROUNDING - the key innovation!
def diff_round(x):
    # Forward pass: round(x)
    # Backward pass: gradient flows through x^3 term
    return tf.round(x) + (x - tf.round(x))**3

def round_only_at_0(x):
    # Alternative: only round values close to zero
    # Helps preserve more gradient information
    cond = tf.cast(tf.abs(x) < 0.5, tf.float32)
    return cond * (x ** 3) + (1 - cond) * x

# Standard JPEG quantization tables
y_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    # ... (8x8 matrix for Y channel)
], dtype=np.float32).T

def y_quantize(image, rounding, factor=1):
    # Divide by quantization matrix (larger values = more compression)
    image = image / (y_table * factor)
    # Apply differentiable rounding
    image = rounding(image)
    return image
```

Main compression function (simplified):

```python
def jpeg_compress_decompress(image, downsample_c=True, rounding=diff_round, factor=1):
    image *= 255  # Scale to [0, 255] range

    # 1. RGB to YCbCr
    image = rgb_to_ycbcr_jpeg(image)

    # 2. Split into Y, Cb, Cr components
    if downsample_c:
        y, cb, cr = downsampling_420(image)  # Cb, Cr at half resolution
    else:
        y, cb, cr = tf.split(image, 3, axis=3)

    # 3. For each component: DCT -> Quantize -> Dequantize -> IDCT
    for component in [y, cb, cr]:
        component = image_to_patches(component)  # Split into 8x8 blocks
        component = dct_8x8(component)           # Transform to frequency domain
        component = quantize(component, rounding, factor)  # Compress
        component = dequantize(component, factor)          # Decompress
        component = idct_8x8(component)          # Back to spatial domain
        component = patches_to_image(component)  # Reassemble

    # 4. YCbCr to RGB
    image = ycbcr_to_rgb_jpeg(image)
    image = tf.clip_by_value(image, 0., 255.)
    image /= 255  # Back to [0, 1]

    return image
```

### This Project's Implementation

Located in `distortions/compression.py`:

```python
class JPEGCompression(Distortion):
    """Differentiable JPEG compression using straight-through estimator."""

    def __init__(self, intensity: float = 0.5, quality: int = 50):
        super().__init__(intensity)
        self.quality = max(1, min(100, quality))

        # Pre-register DCT matrix as buffer (not trainable)
        dct_matrix = _create_dct_matrix(8)
        self.register_buffer("dct_matrix", dct_matrix)

        q_matrix = _get_jpeg_quantization_matrix(self.quality)
        self.register_buffer("q_matrix", q_matrix)

    def _rgb_to_ycbcr(self, x: Tensor) -> Tensor:
        """Convert RGB to YCbCr (JPEG standard)."""
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        # Same coefficients as StegaStamp
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5

        return torch.cat([y, cb, cr], dim=1)

    def _dct_2d(self, blocks: Tensor) -> Tensor:
        """Apply 2D DCT using matrix multiplication.

        DCT: D @ x @ D^T
        Einsum efficiently handles batched matrix multiplication.
        """
        dct = self.dct_matrix
        return torch.einsum("ij,bcnmjk,lk->bcnmil", dct, blocks, dct)

    def _quantize(self, dct_blocks: Tensor) -> Tensor:
        """Quantize with STRAIGHT-THROUGH ESTIMATOR.

        The trick: quantized + (rounded - quantized).detach()
        - Forward pass: uses rounded values (actual quantization)
        - Backward pass: gradient flows through original values
        """
        q_matrix = self.q_matrix.view(1, 1, 1, 1, 8, 8)
        divided = dct_blocks / q_matrix

        # Round with straight-through estimator
        rounded = torch.round(divided)
        # .detach() stops gradient through the rounding operation
        # So gradient flows through 'divided' as if no rounding happened
        quantized = divided + (rounded - divided).detach()

        return quantized * q_matrix

    def forward(self, x: Tensor) -> Tensor:
        # Pad to multiple of 8
        B, C, H, W = x.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_padded = x

        # Full pipeline: RGB -> YCbCr -> DCT -> Quantize -> IDCT -> RGB
        ycbcr = self._rgb_to_ycbcr(x_padded)
        ycbcr_scaled = ycbcr * 255.0 - 128.0

        blocks = self._blockify(ycbcr_scaled, 8)
        dct_blocks = self._dct_2d(blocks)
        quantized = self._quantize(dct_blocks)
        reconstructed = self._idct_2d(quantized)

        ycbcr_reconstructed = self._deblockify(reconstructed, H_pad, W_pad, 8)
        ycbcr_reconstructed = (ycbcr_reconstructed + 128.0) / 255.0
        rgb_reconstructed = self._ycbcr_to_rgb(ycbcr_reconstructed)

        # Blend with original based on intensity
        output = x + self.intensity * (rgb_reconstructed - x)
        return torch.clamp(output, 0.0, 1.0)
```

### Key Differences

| Aspect | StegaStamp | This Project |
|--------|------------|--------------|
| Rounding trick | `round(x) + (x - round(x))^3` | Straight-through estimator |
| DCT implementation | Tensor contraction | Einsum matrix multiplication |
| Chroma subsampling | 4:2:0 (Cb/Cr at half res) | Full resolution (simpler) |
| Quality parameter | Runtime via factor | Constructor parameter |

**Gradient approximation comparison:**

```
StegaStamp (diff_round):
  forward:  round(x)
  backward: 1 + 3*(x - round(x))^2

This Project (straight-through):
  forward:  round(x)
  backward: 1 (gradient passes through unchanged)
```

Straight-through is simpler and often works better in practice because it doesn't distort gradients near integer values.

### Modern Library Alternative: DiffJPEG

[DiffJPEG](https://github.com/mlomnitz/DiffJPEG) is a dedicated differentiable JPEG library:

```python
from DiffJPEG import DiffJPEG

class ModernJPEGCompression(nn.Module):
    def __init__(self, quality=50):
        super().__init__()
        # DiffJPEG handles all the complexity
        self.jpeg = DiffJPEG(
            height=400,
            width=400,
            differentiable=True,
            quality=quality
        )

    def forward(self, x):
        # Expects input in [0, 1] range
        return self.jpeg(x)

# For variable quality during training:
class ModernAdaptiveJPEG(nn.Module):
    def __init__(self, quality_range=(25, 100)):
        super().__init__()
        self.quality_range = quality_range

    def forward(self, x, intensity=1.0):
        # Sample quality based on training progress
        # Lower quality = more compression = harder to decode
        min_q = int(100 - (100 - self.quality_range[0]) * intensity)
        quality = torch.randint(min_q, self.quality_range[1] + 1, (1,)).item()

        jpeg = DiffJPEG(
            height=x.size(2),
            width=x.size(3),
            differentiable=True,
            quality=quality
        )
        return jpeg(x)
```

**Why DiffJPEG is better:**
1. **Complete implementation**: Handles all JPEG details including chroma subsampling
2. **Validated gradients**: Extensively tested for training stability
3. **Dynamic quality**: Can change quality per forward pass
4. **Performance**: Optimized CUDA kernels

---

## Geometric Transforms

Geometric transforms simulate perspective changes from different viewing angles.

### Concept

**Perspective transformation** maps a quadrilateral to another quadrilateral. Think of viewing a rectangular sign from an angle - it appears as a trapezoid.

The math uses **homogeneous coordinates** and a 3x3 **homography matrix**:

```
[x']   [h11 h12 h13] [x]
[y'] = [h21 h22 h23] [y]
[w']   [h31 h32 h33] [1]

Output: (x'/w', y'/w')
```

### StegaStamp Implementation

Uses OpenCV for homography computation, TensorFlow's spatial transformer for warping.

Located in `utils.py:46-75`:

```python
def get_rand_transform_matrix(image_size, d, batch_size):
    """Generate random perspective transform matrices.

    Args:
        image_size: Size of square image
        d: Maximum corner displacement in pixels
        batch_size: Number of transforms to generate

    Returns:
        Ms: Array of shape (batch_size, 2, 8) containing forward and inverse transforms
    """
    Ms = np.zeros((batch_size, 2, 8))

    for i in range(batch_size):
        # Random displacements for each corner (in pixels)
        # tl = top-left, tr = top-right, bl = bottom-left, br = bottom-right
        tl_x = random.uniform(-d, d)
        tl_y = random.uniform(-d, d)
        bl_x = random.uniform(-d, d)
        bl_y = random.uniform(-d, d)
        tr_x = random.uniform(-d, d)
        tr_y = random.uniform(-d, d)
        br_x = random.uniform(-d, d)
        br_y = random.uniform(-d, d)

        # Source rectangle (displaced corners)
        rect = np.array([
            [tl_x, tl_y],                           # Top-left
            [tr_x + image_size, tr_y],              # Top-right
            [br_x + image_size, br_y + image_size], # Bottom-right
            [bl_x, bl_y + image_size]               # Bottom-left
        ], dtype="float32")

        # Destination rectangle (standard corners)
        dst = np.array([
            [0, 0],
            [image_size, 0],
            [image_size, image_size],
            [0, image_size]
        ], dtype="float32")

        # OpenCV computes the 3x3 homography matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        M_inv = np.linalg.inv(M)

        # Store first 8 elements (9th is always 1 after normalization)
        Ms[i, 0, :] = M_inv.flatten()[:8]  # Inverse transform
        Ms[i, 1, :] = M.flatten()[:8]      # Forward transform

    return Ms
```

Applied in `models.py:198-204` using TensorFlow's spatial transformer:

```python
# Warp input image using perspective transform
input_warped = tf.contrib.image.transform(
    image_input,
    M[:, 1, :],           # Forward transform matrix
    interpolation='BILINEAR'
)

# Create mask to identify warped regions
mask_warped = tf.contrib.image.transform(
    tf.ones_like(input_warped),
    M[:, 1, :],
    interpolation='BILINEAR'
)

# Fill gaps with original image
input_warped += (1 - mask_warped) * image_input
```

### This Project's Implementation

Located in `distortions/geometric.py`:

```python
class PerspectiveWarp(Distortion):
    """Random perspective transformation using homography."""

    def __init__(self, intensity: float = 0.5, scale: float = 0.1):
        super().__init__(intensity)
        self.scale = scale  # Max displacement as fraction of image size

    def _compute_homography(self, src: Tensor, dst: Tensor) -> Tensor:
        """Compute 3x3 homography using Direct Linear Transform (DLT).

        DLT sets up a system of equations Ah = 0 where h is the
        flattened homography matrix, then solves via SVD.
        """
        A = []
        for i in range(4):
            x, y = dst[i, 0].item(), dst[i, 1].item()
            u, v = src[i, 0].item(), src[i, 1].item()

            # Each point correspondence gives 2 equations
            A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
            A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])

        A = torch.tensor(A, device=src.device, dtype=src.dtype)

        # SVD solution: h is the right singular vector with smallest singular value
        _, _, Vh = torch.linalg.svd(A)
        H = Vh[-1].reshape(3, 3)
        H = H / H[2, 2]  # Normalize so H[2,2] = 1

        return H

    def _apply_homography(self, H: Tensor, height: int, width: int,
                          device: torch.device, dtype: torch.dtype) -> Tensor:
        """Create sampling grid by applying homography to pixel coordinates."""
        # Create grid of destination coordinates
        y_coords = torch.linspace(-1, 1, height, device=device, dtype=dtype)
        x_coords = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Homogeneous coordinates: [x, y, 1]
        ones = torch.ones_like(grid_x)
        coords = torch.stack([grid_x, grid_y, ones], dim=-1)

        # Apply homography: coords @ H^T
        coords_flat = coords.reshape(-1, 3)
        transformed = coords_flat @ H.T

        # Convert from homogeneous: divide by w
        transformed = transformed.reshape(height, width, 3)
        grid = transformed[..., :2] / transformed[..., 2:3].clamp(min=1e-8)

        return grid

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        # Build sampling grids for each batch item
        grids = []
        for b in range(B):
            # Random corner displacements in normalized [-1, 1] coordinates
            effective_scale = self.scale * self.intensity
            displacements = (torch.rand(4, 2, device=x.device) * 2 - 1) * effective_scale

            src_corners = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]],
                                       device=x.device, dtype=x.dtype)
            dst_corners = src_corners + displacements

            H = self._compute_homography(src_corners, dst_corners)
            grid = self._apply_homography(H, H, W, x.device, x.dtype)
            grids.append(grid)

        grid = torch.stack(grids, dim=0)

        # grid_sample performs the actual warping (bilinear interpolation)
        output = F.grid_sample(x, grid, mode="bilinear",
                               padding_mode="zeros", align_corners=True)

        return output.clamp(0.0, 1.0)
```

### Key Differences

| Aspect | StegaStamp | This Project |
|--------|------------|--------------|
| Homography computation | OpenCV (`cv2.getPerspectiveTransform`) | Pure PyTorch DLT |
| Warping | `tf.contrib.image.transform` | `F.grid_sample` |
| Coordinate system | Pixel coordinates (0 to size) | Normalized (-1 to 1) |
| Batch processing | NumPy loop + TF ops | Full PyTorch |
| Border handling | Fill with original image | Zero padding |

### Modern Library Alternative: Kornia

```python
import kornia.geometry.transform as KGT
import kornia.augmentation as K

class ModernPerspectiveWarp(nn.Module):
    def __init__(self, distortion_scale=0.1):
        super().__init__()
        # Kornia's RandomPerspective handles everything
        self.warp = K.RandomPerspective(
            distortion_scale=distortion_scale,
            p=1.0,  # Always apply
            sampling_method='basic'  # or 'area_preserving'
        )

    def forward(self, x):
        return self.warp(x)

# For more control, use low-level functions:
class ModernHomographyWarp(nn.Module):
    def forward(self, x, H):
        """Apply arbitrary homography matrix H (B, 3, 3)."""
        B, C, H, W = x.shape

        # Kornia's warp_perspective is fully differentiable
        return KGT.warp_perspective(
            x,
            H,
            dsize=(H, W),
            mode='bilinear',
            padding_mode='zeros'
        )

# Combined geometric augmentations:
class ModernGeometricAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = K.AugmentationSequential(
            K.RandomPerspective(distortion_scale=0.1),
            K.RandomRotation(degrees=15),
            K.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        )

    def forward(self, x):
        return self.aug(x)
```

**Why Kornia is better:**
1. **Built-in differentiable homography**: No manual DLT implementation needed
2. **Extensive transforms**: Rotation, scale, shear, thin-plate spline, etc.
3. **Proper interpolation**: Handles anti-aliasing and boundary conditions
4. **GPU optimized**: Faster than manual implementations

---

## Summary

### Implementation Comparison Table

| Distortion | StegaStamp | This Project | Recommended Modern Library |
|------------|------------|--------------|---------------------------|
| Gaussian Blur | Custom kernel + conv2d | Custom kernel + conv2d | `kornia.filters.gaussian_blur2d` |
| Motion Blur | Line kernel + conv2d | Line kernel + conv2d | `kornia.filters.motion_blur` |
| Gaussian Noise | `tf.random_normal` | `torch.randn_like` | `kornia.augmentation.RandomGaussianNoise` |
| Brightness/Hue | Additive shift | Additive shift | `kornia.augmentation.ColorJitter` |
| Contrast | Multiplicative scale | Multiplicative scale | `kornia.enhance.adjust_contrast` |
| Saturation | Lerp with luminance | Lerp with luminance | `kornia.enhance.adjust_saturation` |
| JPEG | `diff_round` trick | Straight-through estimator | DiffJPEG |
| Perspective | OpenCV + tf.transform | Pure PyTorch DLT | `kornia.augmentation.RandomPerspective` |

### Key Architectural Differences

1. **Tensor format**: StegaStamp uses NHWC (TensorFlow default), this project uses NCHW (PyTorch default). This affects how kernels and operations are shaped.

2. **Graph vs Eager**: StegaStamp builds a static computation graph with `tf.cond` for branching. PyTorch uses dynamic graphs with Python control flow.

3. **Randomness**: StegaStamp's random ops are graph operations executed at runtime. PyTorch samples immediately during forward pass.

4. **Gradient handling**: Both projects solve the non-differentiable rounding problem, but with different approximations.

### Recommendations for New Projects

1. **Use Kornia** for standard augmentations - it's well-tested, fast, and integrates seamlessly with PyTorch training loops.

2. **Use DiffJPEG** for JPEG compression - it handles all the complexity of proper JPEG simulation.

3. **Keep custom implementations** when you need:
   - Exact compatibility with a reference implementation
   - Specific parameter distributions not supported by libraries
   - Maximum control over gradient flow

4. **Consider torchvision.transforms.v2** for basic augmentations in production - it's part of PyTorch's official ecosystem and well-maintained.
