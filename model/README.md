# stegastamp

StegaStamp encoder/decoder for picode. Based on [StegaStamp](https://github.com/tancik/StegaStamp) (Tancik et al., CVPR 2020).

Part of the [picode](../README.md) project.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### Encoding and Decoding

```python
from stegastamp import Encoder, Decoder
import torch

# Initialize models
encoder = Encoder(num_bits=100)
decoder = Decoder(num_bits=100)

# Encode a message into an image
image = torch.rand(1, 3, 400, 400)  # [0, 1] range
message = torch.randint(0, 2, (1, 100)).float()  # Binary message

encoded_image = encoder(image, message)
recovered_message = decoder(encoded_image)
binary_message = (recovered_message > 0.5).float()
```

### Training

```python
from stegastamp import StegaStampTrainer, train_step
from distortions import Compose, GaussianNoise, JPEGCompression

# Create trainer
trainer = StegaStampTrainer(num_bits=100)

# Define distortion pipeline
distortion = Compose([
    GaussianNoise(intensity=0.2),
    JPEGCompression(intensity=0.5),
])

# Training step
losses = train_step(
    encoder=trainer.encoder,
    decoder=trainer.decoder,
    images=batch_images,
    distortion=distortion,
    optimizer=optimizer,
)

print(f"Loss: {losses['loss']:.4f}")
```

## Architecture

- **Encoder**: U-Net that takes (image, message) and outputs encoded image
- **Decoder**: CNN that extracts message probabilities from (distorted) image
- **Loss**: BCE (message) + L2 (image) + optional LPIPS (perceptual)

## API Reference

### Encoder

```python
Encoder(num_bits: int = 100)
```

- `forward(image, message) -> encoded_image`
- `prepare_message(message) -> spatial_tensor`

### Decoder

```python
Decoder(num_bits: int = 100)
```

- `forward(image) -> probabilities`
- `decode(image) -> binary_message`

### Loss Functions

```python
compute_loss(original, encoded, message, decoded, ...) -> dict
message_loss(decoded, message) -> Tensor
image_loss(encoded, original) -> Tensor
```
