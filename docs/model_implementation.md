# Model Implementation Comparison

This document compares the encoder and decoder implementations between the original **StegaStamp** project (TensorFlow 1.x/Keras) and this project's **PyTorch** implementation. Each section explains the architectural concepts, shows code from both projects, and highlights key differences.

## Table of Contents

1. [Overview](#overview)
2. [Encoder Architecture](#encoder-architecture)
3. [Decoder Architecture](#decoder-architecture)
4. [Loss Functions](#loss-functions)
5. [Training Pipeline](#training-pipeline)
6. [Summary](#summary)

---

## Overview

### What is StegaStamp?

StegaStamp is a deep learning system for **image steganography** - the art of hiding information within images. Given:
- An **image** (the "cover" image)
- A **message** (a sequence of bits, e.g., 100 bits)

The system produces:
- An **encoded image** that looks nearly identical to the original
- The ability to **decode** the message even after the image has been printed, photographed, compressed, or otherwise distorted

### The Two Networks

| Network | Purpose | Input | Output |
|---------|---------|-------|--------|
| **Encoder** | Hide message in image | Image + Message bits | Encoded image |
| **Decoder** | Extract message from image | (Distorted) encoded image | Message bits |

### Framework Differences

| Aspect | StegaStamp (TensorFlow 1.x) | This Project (PyTorch) |
|--------|----------------------------|------------------------|
| Computation mode | Static graph (define-then-run) | Dynamic graph (define-by-run) |
| Tensor format | NHWC (batch, height, width, channels) | NCHW (batch, channels, height, width) |
| Layer definition | Keras `Layer` subclass | `nn.Module` subclass |
| Image size | Fixed at 400×400 | Fixed at 400×400 |
| Default bits | 20 | 100 |

---

## Encoder Architecture

The encoder is a **U-Net** - a convolutional neural network with an encoder-decoder structure and **skip connections**. Understanding U-Nets is key to understanding this architecture.

### Concept: U-Net Architecture

A U-Net has three parts:

```
Input
  │
  ▼
┌─────────────┐
│  Encoder    │  ← Downsamples image, extracts features
│  (contract) │    Each level halves spatial dimensions
└──────┬──────┘
       │
       │  (bottleneck - smallest spatial size)
       │
       ▼
┌─────────────┐
│  Decoder    │  ← Upsamples back to original size
│  (expand)   │    Each level doubles spatial dimensions
└──────┬──────┘
       │
       ▼
    Output
```

The key innovation of U-Net is **skip connections**: at each level, the encoder's output is concatenated with the decoder's input at the corresponding level. This preserves fine spatial details that would otherwise be lost during downsampling.

```
Encoder Level 1 ──────────────────┐
       │                          │ (skip connection)
       ▼                          ▼
Encoder Level 2 ───────┐    Decoder Level 1 ← concatenate
       │               │          ▲
       ▼               ▼          │
Encoder Level 3   Decoder Level 2 ← concatenate
       │               ▲
       ▼               │
   Bottleneck ─────────┘
```

### Why U-Net for Steganography?

1. **Preserves spatial details**: Skip connections maintain fine image details needed for imperceptible encoding
2. **Multi-scale processing**: Different levels capture features at different scales
3. **Residual output**: The network outputs a small "residual" that's added to the original image

### Message Preparation

Before the image can be processed, the binary message must be transformed into a spatial format that can be concatenated with the image.

#### StegaStamp Implementation

Located in `models.py:11-62`:

```python
class StegaStampEncoder(Layer):
    def __init__(self, height, width):
        super(StegaStampEncoder, self).__init__()
        # Message FC: maps num_bits → 7500 values
        # 7500 = 50 × 50 × 3 (will be reshaped to small image)
        self.secret_dense = Dense(7500, activation='relu', kernel_initializer='he_normal')
        # ... convolution layers defined here ...

    def call(self, inputs):
        secret, image = inputs
        # Center inputs around 0 for better training stability
        # Original range [0, 1] → [-0.5, 0.5]
        secret = secret - .5
        image = image - .5

        # Transform message bits to spatial features:
        # 1. Dense layer: (B, num_bits) → (B, 7500)
        secret = self.secret_dense(secret)
        # 2. Reshape to small "image": (B, 7500) → (B, 50, 50, 3)
        secret = Reshape((50, 50, 3))(secret)
        # 3. Upsample to match image size: (B, 50, 50, 3) → (B, 400, 400, 3)
        secret_enlarged = UpSampling2D(size=(8,8))(secret)

        # Concatenate along channel axis: 3 (image) + 3 (message) = 6 channels
        inputs = concatenate([secret_enlarged, image], axis=-1)
        # ... encoder-decoder processing ...
```

#### This Project's Implementation

Located in `picode/models/stegastamp/encoder.py`:

```python
def prepare_message(self, message: Tensor) -> Tensor:
    """Expand message bits to spatial feature map.

    Args:
        message: (B, num_bits) binary tensor

    Returns:
        (B, 3, 400, 400) spatial tensor
    """
    # Dense layer with ReLU: (B, num_bits) → (B, 7500)
    x = F.relu(self.msg_fc(message))
    # Reshape to 3-channel "image": (B, 7500) → (B, 3, 50, 50)
    # Note: PyTorch uses NCHW format (channels first)
    x = x.view(-1, 3, 50, 50)
    # Upsample to full resolution: (B, 3, 50, 50) → (B, 3, 400, 400)
    x = F.interpolate(x, size=(400, 400), mode="nearest")
    return x
```

### Key Insight: Why 50×50×3?

The magic number 7500 = 50 × 50 × 3 is chosen so that:

1. **50×50 is divisible by image size**: 400 / 50 = 8, so we can upsample by exactly 8×
2. **3 channels match image**: The message features have the same channel count as the image, creating a natural 6-channel input (3 image + 3 message)
3. **Reasonable compression**: Even 100 bits → 7500 values gives the network room to learn useful spatial representations

### Encoder Path (Downsampling)

Both implementations use 5 convolutional layers that progressively downsample the image:

| Level | Resolution | Channels | Stride |
|-------|------------|----------|--------|
| Input | 400×400 | 6 (image + message) | - |
| Conv1 | 400×400 | 32 | 1 |
| Conv2 | 200×200 | 32 | 2 |
| Conv3 | 100×100 | 64 | 2 |
| Conv4 | 50×50 | 128 | 2 |
| Conv5 | 25×25 | 256 | 2 |

**Stride of 2** means the convolution moves 2 pixels at a time, halving the spatial dimensions. This is equivalent to applying a convolution then max pooling, but more efficient.

#### StegaStamp Implementation

```python
# Convolution layers with stride 2 for downsampling
self.conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
self.conv2 = Conv2D(32, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
self.conv3 = Conv2D(64, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
self.conv4 = Conv2D(128, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
self.conv5 = Conv2D(256, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')

# In call():
conv1 = self.conv1(inputs)  # 400×400, 32 channels
conv2 = self.conv2(conv1)   # 200×200, 32 channels
conv3 = self.conv3(conv2)   # 100×100, 64 channels
conv4 = self.conv4(conv3)   # 50×50, 128 channels
conv5 = self.conv5(conv4)   # 25×25, 256 channels (bottleneck)
```

#### This Project's Implementation

```python
class ConvBlock(nn.Module):
    """Conv -> ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        # 3×3 convolution with padding=1 maintains spatial size when stride=1
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.conv(x))

# In Encoder.__init__():
self.conv1 = ConvBlock(6, 32)           # 6 input channels (image + message)
self.conv2 = ConvBlock(32, 32, stride=2)  # 200×200
self.conv3 = ConvBlock(32, 64, stride=2)  # 100×100
self.conv4 = ConvBlock(64, 128, stride=2) # 50×50
self.conv5 = ConvBlock(128, 256, stride=2) # 25×25
```

### Decoder Path (Upsampling with Skip Connections)

The decoder progressively upsamples back to the original resolution, concatenating skip connections at each level:

| Level | Input Resolution | Skip From | Output Resolution | Output Channels |
|-------|-----------------|-----------|-------------------|-----------------|
| Up6 | 25×25 | Conv4 (50×50) | 50×50 | 128 |
| Up7 | 50×50 | Conv3 (100×100) | 100×100 | 64 |
| Up8 | 100×100 | Conv2 (200×200) | 200×200 | 32 |
| Up9 | 200×200 | Conv1 (400×400) | 400×400 | 32 |

#### StegaStamp Implementation

```python
# Upsampling layers
self.up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')
self.conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
# ... similar for up7, up8, up9 ...

# In call():
# Upsample 2× then apply conv
up6 = self.up6(UpSampling2D(size=(2,2))(conv5))  # 25×25 → 50×50
# Concatenate with corresponding encoder level
merge6 = concatenate([conv4, up6], axis=3)  # 128 + 128 = 256 channels
conv6 = self.conv6(merge6)  # 256 → 128 channels

up7 = self.up7(UpSampling2D(size=(2,2))(conv6))
merge7 = concatenate([conv3, up7], axis=3)
conv7 = self.conv7(merge7)
# ... and so on ...

# Final merge includes original input for extra detail
merge9 = concatenate([conv1, up9, inputs], axis=3)  # 32 + 32 + 6 = 70 channels
```

#### This Project's Implementation

```python
class UpBlock(nn.Module):
    """Upsample -> Conv -> ReLU with skip connection."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        # 2×2 conv after upsampling (matches StegaStamp's pattern)
        self.up_conv = nn.Conv2d(in_ch, out_ch, 2, padding=0)
        # 3×3 conv after concatenation
        self.conv = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        # Upsample by 2× using nearest-neighbor interpolation
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Apply 2×2 conv (need to pad for valid output size)
        x = F.relu(self.up_conv(F.pad(x, (0, 1, 0, 1))))
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        # Final 3×3 conv
        return F.relu(self.conv(x))

# In Encoder.__init__():
self.up6 = UpBlock(256, 128, 128)  # Skip from conv4
self.up7 = UpBlock(128, 64, 64)    # Skip from conv3
self.up8 = UpBlock(64, 32, 32)     # Skip from conv2
self.up9 = UpBlock(32, 32, 32)     # Skip from conv1
```

### Residual Output

Both implementations output a **residual** (small modification) that's added to the original image, rather than generating the entire image from scratch. This has several benefits:

1. **Identity initialization**: A zero residual means no change to the image
2. **Easier optimization**: The network only needs to learn small adjustments
3. **Preserved image structure**: Original pixel values are mostly retained

#### StegaStamp Implementation

```python
# Final convolutions
self.conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
self.conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
# 1×1 conv to produce 3-channel residual (no activation!)
self.residual = Conv2D(3, 1, activation=None, padding='same', kernel_initializer='he_normal')

# In call():
conv9 = self.conv9(merge9)
conv10 = self.conv10(conv9)
residual = self.residual(conv9)  # Note: uses conv9, not conv10
return residual  # Caller adds this to original image
```

**Important note**: The StegaStamp encoder returns only the residual. The addition to the original image happens in `build_model`:

```python
# In models.py, build_model function:
residual_warped = encoder((secret_input, input_warped))
encoded_warped = residual_warped + input_warped  # Addition happens here
```

#### This Project's Implementation

```python
# In Encoder:
self.conv_out1 = ConvBlock(32, 32)
self.conv_out2 = nn.Conv2d(32, 3, 1)  # 1×1 conv, no activation

def forward(self, image: Tensor, message: Tensor) -> Tensor:
    # ... encoder-decoder processing ...

    # Output layers
    x = self.conv_out1(x)
    residual = self.conv_out2(x)

    # Add residual and clamp to valid range
    encoded = torch.clamp(image + residual, 0.0, 1.0)
    return encoded  # Returns the full encoded image
```

**Difference**: This project returns the final encoded image directly, clamping to [0, 1]. StegaStamp returns the residual, and the caller handles the addition.

### Key Differences: Encoder

| Aspect | StegaStamp | This Project |
|--------|------------|--------------|
| Input format | NHWC (channels last) | NCHW (channels first) |
| Input normalization | Subtracts 0.5 | No normalization |
| Return value | Residual only | Encoded image (clamped) |
| Final merge | Includes original input | Only skip from conv1 |
| Convolution blocks | Inline definitions | Modular ConvBlock/UpBlock |

---

## Decoder Architecture

The decoder is a simpler **CNN classifier** that extracts the hidden message from an (possibly distorted) encoded image. Unlike the encoder's U-Net structure, it's a straightforward feedforward network.

### Concept: Spatial Transformer Network (STN)

A critical difference between implementations is the **Spatial Transformer Network (STN)** in StegaStamp's decoder.

#### What is an STN?

An STN is a differentiable module that can:
1. **Learn** geometric transformations (rotation, scale, shear, perspective)
2. **Apply** them to an input image
3. **Backpropagate** through the transformation

This is crucial for real-world robustness: if you print a watermarked image and photograph it at an angle, the decoder needs to "straighten" the image before extracting the message.

```
Distorted Input
      │
      ▼
┌─────────────────┐
│ Localization    │  ← Small CNN that predicts transformation parameters
│ Network         │    Output: 6 parameters for 2D affine transform
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Grid Generator  │  ← Creates sampling grid based on parameters
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Sampler         │  ← Applies transformation via differentiable sampling
└────────┬────────┘
         │
         ▼
    Rectified Image → Regular CNN Decoder
```

#### The Affine Transform

The STN predicts a 2×3 affine transformation matrix:

```
[θ₁₁  θ₁₂  θ₁₃]   [x]   [x']
[θ₂₁  θ₂₂  θ₂₃] × [y] = [y']
                   [1]
```

Where:
- θ₁₁, θ₂₂ control **scale**
- θ₁₂, θ₂₁ control **rotation/shear**
- θ₁₃, θ₂₃ control **translation**

The network is initialized to the **identity transform**:
```
[1  0  0]
[0  1  0]
```

This means at the start of training, the STN does nothing, allowing the main decoder to train normally.

### StegaStamp Decoder with STN

Located in `models.py:64-100`:

```python
class StegaStampDecoder(Layer):
    def __init__(self, secret_size, height, width):
        super(StegaStampDecoder, self).__init__()
        self.height = height
        self.width = width

        # LOCALIZATION NETWORK: Predicts affine transform parameters
        self.stn_params = Sequential([
            # Downsampling CNN to extract geometric features
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),
            Flatten(),
            Dense(128, activation='relu')  # 128-dim feature vector
        ])

        # Initialize to identity transform
        initial = np.array([[1., 0, 0], [0, 1., 0]])  # 2×3 matrix
        initial = initial.astype('float32').flatten()  # 6 values

        # Linear layer to predict 6 affine parameters
        self.W_fc1 = tf.Variable(tf.zeros([128, 6]), name='W_fc1')
        self.b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')

        # MAIN DECODER: Extracts message from rectified image
        self.decoder = Sequential([
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(secret_size)  # No activation: logits for BCE loss
        ])

    def call(self, image):
        image = image - .5  # Center around 0

        # 1. Predict transformation parameters
        stn_params = self.stn_params(image)  # (B, 128)
        x = tf.matmul(stn_params, self.W_fc1) + self.b_fc1  # (B, 6)

        # 2. Apply spatial transformation
        # stn_transformer is a TensorFlow spatial transformer
        transformed_image = stn_transformer(image, x, [self.height, self.width, 3])

        # 3. Decode message from rectified image
        return self.decoder(transformed_image)
```

### This Project's Decoder (Without STN)

Located in `picode/models/stegastamp/decoder.py`:

```python
class Decoder(nn.Module):
    """CNN decoder that extracts message bits from an encoded image.

    Args:
        num_bits: Number of bits in the message (default: 100).
    """

    def __init__(self, num_bits: int = 100) -> None:
        super().__init__()
        self.num_bits = num_bits

        # Convolutional backbone - same structure as StegaStamp's decoder
        # Input: 3 channels, 400×400
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)   # → 200×200
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)            # → 200×200
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # → 100×100
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)            # → 100×100
        self.conv5 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # → 50×50
        self.conv6 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # → 25×25
        self.conv7 = nn.Conv2d(128, 128, 3, stride=2, padding=1) # → 13×13

        # FC head
        # After conv7 at 400×400 input: 128 × 13 × 13 = 21632
        self.fc1 = nn.Linear(128 * 13 * 13, 512)
        self.fc2 = nn.Linear(512, num_bits)

    def forward(self, image: Tensor) -> Tensor:
        """Extract message probabilities from image."""
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        x = x.flatten(start_dim=1)  # (B, 21632)
        x = F.relu(self.fc1(x))     # (B, 512)
        x = torch.sigmoid(self.fc2(x))  # (B, num_bits) in [0, 1]
        return x
```

### Why This Project Omits STN

The STN is primarily useful for:
1. **Physical robustness**: Handling photographed/scanned images
2. **Perspective correction**: Undoing camera angle distortions

This project's decoder focuses on the core steganography task without physical-world robustness. If you need perspective correction, you could:
1. Add an STN using `torchvision.transforms.functional.affine_grid` and `F.grid_sample`
2. Use Kornia's differentiable perspective transforms
3. Pre-process images with traditional computer vision before decoding

### Key Differences: Decoder

| Aspect | StegaStamp | This Project |
|--------|------------|--------------|
| Spatial Transformer | Yes (6-param affine) | No |
| Input normalization | Subtracts 0.5 | No normalization |
| Output activation | None (raw logits) | Sigmoid (probabilities) |
| Loss function used | Sigmoid cross-entropy | Binary cross-entropy |
| Feature dimensions | 400→200→100→50→25→13 | Same downsampling pattern |
| Final FC | 512 → secret_size | 512 → num_bits |

### Architectural Diagram

```
StegaStamp Decoder:

Input Image (400×400×3)
         │
         ├─────────────────────┐
         │                     │
         ▼                     ▼
┌─────────────────┐    ┌─────────────────┐
│ Localization    │    │ Main Decoder    │
│ Network         │    │ (after STN)     │
│                 │    │                 │
│ Conv 32, s=2    │    │ Conv 32, s=2    │
│ Conv 64, s=2    │    │ Conv 32         │
│ Conv 128, s=2   │    │ Conv 64, s=2    │
│ Flatten         │    │ Conv 64         │
│ Dense 128       │    │ Conv 64, s=2    │
│ Dense 6 (θ)     │    │ Conv 128, s=2   │
└────────┬────────┘    │ Conv 128, s=2   │
         │             │ Flatten         │
         ▼             │ Dense 512       │
    ┌──────────┐       │ Dense num_bits  │
    │   STN    │       └────────┬────────┘
    │ Transform│                │
    └────┬─────┘                │
         │                      │
         └──────────────────────┘
                  │
                  ▼
         Message Logits (num_bits)


This Project's Decoder:

Input Image (400×400×3)
         │
         ▼
    Conv 32, s=2 (200×200)
         │
         ▼
    Conv 32 (200×200)
         │
         ▼
    Conv 64, s=2 (100×100)
         │
         ▼
    Conv 64 (100×100)
         │
         ▼
    Conv 64, s=2 (50×50)
         │
         ▼
    Conv 128, s=2 (25×25)
         │
         ▼
    Conv 128, s=2 (13×13)
         │
         ▼
    Flatten (21632)
         │
         ▼
    Dense 512
         │
         ▼
    Dense num_bits + Sigmoid
         │
         ▼
    Message Probabilities
```

---

## Loss Functions

Both implementations use similar loss functions to train the encoder-decoder pair, but with different organization and some variations.

### The Three Core Losses

| Loss | Purpose | Effect on Training |
|------|---------|-------------------|
| **Message Loss** | Decoder should recover message | Pushes for high bit accuracy |
| **Image Loss (L2/MSE)** | Encoded image ≈ original | Pushes for imperceptibility |
| **Perceptual Loss (LPIPS)** | Perceptually similar | Preserves texture/structure |

### Message Loss: Binary Cross-Entropy

For each bit, we compute the cross-entropy between the true value and the predicted probability:

```
BCE(y, ŷ) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

Where:
- `y` is the true bit (0 or 1)
- `ŷ` is the predicted probability

#### StegaStamp Implementation

```python
# Uses logits (pre-sigmoid) for numerical stability
secret_loss_op = tf.losses.sigmoid_cross_entropy(secret_input, decoded_secret)
```

This applies sigmoid internally, which is numerically more stable than applying sigmoid first then using log.

#### This Project's Implementation

```python
def message_loss(decoded: Tensor, message: Tensor) -> Tensor:
    """Binary cross-entropy loss for message recovery."""
    # 'decoded' already has sigmoid applied in decoder.forward()
    return F.binary_cross_entropy(decoded, message)
```

### Image Loss: L2/MSE with YUV Weighting

Simple pixel-wise difference:

```
MSE = (1/N) × Σ(original - encoded)²
```

#### StegaStamp Implementation

StegaStamp uses **YUV color space** with separate weights:

```python
# Convert to YUV (separates brightness from color)
encoded_image_yuv = tf.image.rgb_to_yuv(encoded_image)
image_input_yuv = tf.image.rgb_to_yuv(image_input)

# Compute difference in YUV space
im_diff = encoded_image_yuv - image_input_yuv

# Add edge gain (penalize changes near edges more)
im_diff += im_diff * tf.expand_dims(falloff_im, axis=[-1])

# Weighted sum of Y, U, V losses
yuv_loss_op = tf.reduce_mean(tf.square(im_diff), axis=[0,1,2])
image_loss_op = tf.tensordot(yuv_loss_op, yuv_scales, axes=1)
```

**Why YUV?**
- **Y (luminance)**: Human eyes are most sensitive to brightness changes
- **U, V (chrominance)**: Less sensitive to color changes
- Allows different weights per channel

**Edge falloff**: The `falloff_im` mask penalizes changes near image borders more heavily, since watermark artifacts are most visible at edges.

#### This Project's Implementation

```python
def image_loss(encoded: Tensor, original: Tensor) -> Tensor:
    """L2 (MSE) loss between encoded and original image."""
    return F.mse_loss(encoded, original)
```

Simpler RGB-space MSE without edge weighting or YUV conversion.

### Perceptual Loss: LPIPS

LPIPS (Learned Perceptual Image Patch Similarity) measures how different two images look to a human, rather than pixel-wise difference.

It works by:
1. Passing both images through a pretrained CNN (VGG, AlexNet)
2. Extracting feature maps at multiple layers
3. Computing weighted distance between features

```
                    Original Image         Encoded Image
                           │                      │
                           ▼                      ▼
                    ┌─────────────────────────────────┐
                    │        Pretrained CNN           │
                    │    (VGG-16 or similar)          │
                    └─────────────────────────────────┘
                           │                      │
            ┌──────────────┼──────────────────────┼──────────────┐
            │              │                      │              │
            ▼              ▼                      ▼              ▼
        Layer 1        Layer 2                Layer 1        Layer 2
        Features       Features               Features       Features
            │              │                      │              │
            └──────────────┼──────────────────────┼──────────────┘
                           │                      │
                           ▼                      ▼
                    ┌─────────────────────────────────┐
                    │   Weighted Feature Distances    │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                             LPIPS Score
```

#### StegaStamp Implementation

```python
lpips_loss_op = tf.reduce_mean(lpips_tf.lpips(image_input, encoded_image))
```

Uses a custom TensorFlow LPIPS implementation.

#### This Project's Implementation

```python
if use_lpips and lpips_fn is not None:
    # LPIPS expects input in [-1, 1] range
    orig_scaled = original * 2 - 1
    enc_scaled = encoded * 2 - 1
    loss_lpips = lpips_fn(enc_scaled, orig_scaled).mean()
    total = total + weight_lpips * loss_lpips
```

Uses an external LPIPS function (e.g., from the `lpips` package).

### Total Loss

Both implementations combine losses with learned/fixed weights:

#### StegaStamp

```python
loss_op = (loss_scales[0] * image_loss_op +     # L2 loss
           loss_scales[1] * lpips_loss_op +      # Perceptual loss
           loss_scales[2] * secret_loss_op)      # Message loss

if not args.no_gan:
    loss_op += loss_scales[3] * G_loss  # Adversarial loss (optional)
```

Weights are ramped up during training to gradually increase difficulty.

#### This Project

```python
total = weight_msg * loss_msg + weight_l2 * loss_l2
if use_lpips:
    total = total + weight_lpips * loss_lpips
```

Fixed weights without ramping.

### Discriminator Loss (GAN)

StegaStamp optionally includes a **discriminator** network that tries to distinguish encoded images from originals:

```python
class Discriminator(Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = Sequential([
            Conv2D(8, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(16, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(1, (3, 3), activation=None, padding='same')  # Per-pixel discrimination
        ])

    def call(self, image):
        x = image - .5
        x = self.model(x)
        output = tf.reduce_mean(x)  # Average prediction
        return output, x  # Also return heatmap for visualization

# Wasserstein GAN loss (no sigmoid/log)
D_loss = D_output_real - D_output_fake  # Discriminator tries to maximize
G_loss = D_output_fake                   # Generator tries to maximize
```

This project omits the GAN component, relying on L2 + LPIPS for image quality.

---

## Training Pipeline

### Loss Ramping in StegaStamp

StegaStamp uses **curriculum learning** - gradually increasing difficulty during training:

```python
# Ramp function: linearly increases from 0 to 1 over 'ramp' steps
ramp_fn = lambda ramp: tf.minimum(tf.to_float(global_step) / ramp, 1.)

# Example: L2 loss scale ramps up over 20000 steps
l2_loss_scale = min(args.l2_loss_scale * global_step / args.l2_loss_ramp,
                    args.l2_loss_scale)
```

This helps because:
1. **Early training**: Focus on message recovery (high secret_loss weight)
2. **Later training**: Add image quality constraints (higher l2/lpips weights)

### Distortions During Training

StegaStamp applies distortions **during training** through `transform_net`:

```python
def transform_net(encoded_image, args, global_step):
    # Each distortion is ramped up during training
    ramp_fn = lambda ramp: tf.minimum(tf.to_float(global_step) / ramp, 1.)

    # Apply distortions in sequence:
    # 1. Blur (25% gaussian, 25% motion, 50% identity)
    encoded_image = tf.nn.conv2d(encoded_image, blur_kernel, ...)

    # 2. Gaussian noise
    noise = tf.random_normal(shape=tf.shape(encoded_image), stddev=rnd_noise)
    encoded_image = encoded_image + noise

    # 3. Contrast scaling
    encoded_image = encoded_image * contrast_scale

    # 4. Brightness/hue shift
    encoded_image = encoded_image + rnd_brightness

    # 5. Saturation adjustment
    encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * luminance

    # 6. JPEG compression (differentiable)
    encoded_image = utils.jpeg_compress_decompress(encoded_image, ...)

    return encoded_image
```

### This Project's Training Step

```python
def train_step(
    encoder: Encoder,
    decoder: Decoder,
    images: Tensor,
    distortion: Callable[[Tensor], Tensor] | None = None,
    optimizer: Optimizer | None = None,
    num_bits: int = 100,
    use_lpips: bool = True,
    lpips_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> dict[str, float]:
    """Execute a single training step."""
    batch_size = images.shape[0]
    device = images.device

    # Generate random messages
    messages = torch.randint(0, 2, (batch_size, num_bits), device=device).float()

    # Encode messages into images
    encoded = encoder(images, messages)

    # Apply distortions (if provided)
    if distortion is not None:
        distorted = distortion(encoded)
    else:
        distorted = encoded

    # Decode messages from (distorted) images
    decoded = decoder(distorted)

    # Compute losses
    losses = compute_loss(
        original=images,
        encoded=encoded,
        message=messages,
        decoded=decoded,
        lpips_fn=lpips_fn,
        use_lpips=use_lpips,
    )

    # Backpropagate and update
    if optimizer is not None:
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

    return {k: v.detach().item() for k, v in losses.items()}
```

**Key difference**: Distortion is passed as a callable, making it easy to use the separate distortions library.

### Perspective Warping in StegaStamp

StegaStamp applies perspective transforms during training to improve robustness:

```python
# In build_model():
# 1. Warp input image with random perspective
input_warped = tf.contrib.image.transform(image_input, M[:,1,:], ...)
mask_warped = tf.contrib.image.transform(tf.ones_like(input_warped), M[:,1,:], ...)
input_warped += (1-mask_warped) * image_input  # Fill gaps with original

# 2. Encode warped image
residual_warped = encoder((secret_input, input_warped))
encoded_warped = residual_warped + input_warped

# 3. Unwarp back to original perspective
residual = tf.contrib.image.transform(residual_warped, M[:,0,:], ...)  # Inverse transform
encoded_image = image_input + residual  # Final encoded image in original perspective
```

This trains the encoder to handle perspective variations by:
1. Encoding in a distorted perspective
2. Transforming back to original view

The decoder's STN learns to undo these transforms during inference.

---

## Summary

### Implementation Comparison Table

| Component | StegaStamp | This Project |
|-----------|------------|--------------|
| **Framework** | TensorFlow 1.x + Keras | PyTorch |
| **Tensor format** | NHWC | NCHW |
| **Encoder architecture** | U-Net with skip connections | Same structure |
| **Encoder output** | Residual only | Full encoded image |
| **Decoder architecture** | STN + CNN | CNN only |
| **Spatial transformer** | Yes (affine) | No |
| **Message output** | Logits | Sigmoid probabilities |
| **Image loss** | YUV-weighted L2 + edge falloff | Simple MSE |
| **Perceptual loss** | LPIPS | LPIPS (optional) |
| **GAN loss** | Optional discriminator | Not included |
| **Loss ramping** | Yes (curriculum learning) | No |
| **Default message size** | 20 bits | 100 bits |

### When to Use Which

**Use the original StegaStamp if you need:**
- Physical robustness (printed/photographed images)
- Perspective correction (camera at angles)
- GAN-based image quality
- Curriculum learning for complex training

**Use this project's implementation if you need:**
- Simpler PyTorch integration
- Modular distortions library
- Cleaner code structure
- Easy experimentation without STN complexity

### Extending This Project

To add STN support:

```python
import torch.nn.functional as F

class DecoderWithSTN(nn.Module):
    def __init__(self, num_bits: int = 100):
        super().__init__()
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 50 * 50, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6 affine parameters
        )

        # Initialize to identity
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

        # Main decoder (same as before)
        self.decoder = Decoder(num_bits)

    def forward(self, x: Tensor) -> Tensor:
        # Predict affine transform
        theta = self.localization(x).view(-1, 2, 3)

        # Create sampling grid
        grid = F.affine_grid(theta, x.size(), align_corners=False)

        # Apply transform
        x_transformed = F.grid_sample(x, grid, align_corners=False)

        # Decode from rectified image
        return self.decoder(x_transformed)
```

### References

- [StegaStamp Paper (CVPR 2020)](https://arxiv.org/abs/1904.05343)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Spatial Transformer Networks Paper](https://arxiv.org/abs/1506.02025)
- [LPIPS Paper](https://arxiv.org/abs/1801.03924)
