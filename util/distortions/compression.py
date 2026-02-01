"""Compression distortions.

Implements differentiable JPEG compression matching StegaStamp's
utils.py jpeg_compress_decompress function.

The key innovation is using a straight-through estimator to allow
gradients to flow through the quantization step.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from distortions.base import Distortion


def _create_dct_matrix(n: int = 8) -> Tensor:
    """Create DCT transformation matrix.

    Args:
        n: Size of the DCT matrix (default 8 for JPEG).

    Returns:
        DCT matrix of shape (n, n).
    """
    dct = torch.zeros(n, n)
    for k in range(n):
        for i in range(n):
            if k == 0:
                dct[k, i] = 1.0 / math.sqrt(n)
            else:
                dct[k, i] = math.sqrt(2.0 / n) * math.cos(math.pi * k * (2 * i + 1) / (2 * n))
    return dct


def _get_jpeg_quantization_matrix(quality: int) -> Tensor:
    """Get standard JPEG luminance quantization matrix scaled by quality.

    Args:
        quality: JPEG quality (1-100). Lower means more compression.

    Returns:
        Quantization matrix of shape (8, 8).
    """
    # Standard JPEG luminance quantization matrix
    base_matrix = torch.tensor(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ],
        dtype=torch.float32,
    )

    # Scale quality factor (JPEG standard scaling)
    if quality < 50:
        scale = 5000.0 / quality
    else:
        scale = 200.0 - 2.0 * quality

    # Apply scaling and clamp to valid range
    q_matrix = torch.floor((base_matrix * scale + 50.0) / 100.0)
    q_matrix = torch.clamp(q_matrix, 1.0, 255.0)

    return q_matrix


class JPEGCompression(Distortion):
    """Differentiable JPEG compression distortion.

    Implements DCT-based JPEG compression with straight-through estimator
    for gradient flow through quantization. Matches StegaStamp's
    jpeg_compress_decompress implementation.

    Pipeline:
        RGB -> YCbCr -> DCT -> Quantize -> IDCT -> RGB

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        quality: JPEG quality factor (1-100, default 50).
            Lower values produce more compression artifacts.
    """

    name = "jpeg_compression"

    def __init__(self, intensity: float = 0.5, quality: int = 50):
        super().__init__(intensity)
        self.quality = max(1, min(100, quality))

        # Create and register DCT matrix as buffer (not parameter)
        dct_matrix = _create_dct_matrix(8)
        self.register_buffer("dct_matrix", dct_matrix)

        # Create and register quantization matrix as buffer
        q_matrix = _get_jpeg_quantization_matrix(self.quality)
        self.register_buffer("q_matrix", q_matrix)

    def _rgb_to_ycbcr(self, x: Tensor) -> Tensor:
        """Convert RGB to YCbCr color space.

        Args:
            x: RGB tensor of shape (B, 3, H, W) in [0, 1].

        Returns:
            YCbCr tensor of shape (B, 3, H, W).
        """
        # Conversion matrix from RGB to YCbCr
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5

        return torch.cat([y, cb, cr], dim=1)

    def _ycbcr_to_rgb(self, x: Tensor) -> Tensor:
        """Convert YCbCr to RGB color space.

        Args:
            x: YCbCr tensor of shape (B, 3, H, W).

        Returns:
            RGB tensor of shape (B, 3, H, W) in [0, 1].
        """
        y, cb, cr = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        cb = cb - 0.5
        cr = cr - 0.5

        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb

        return torch.cat([r, g, b], dim=1)

    def _blockify(self, x: Tensor, block_size: int = 8) -> Tensor:
        """Reshape image into 8x8 blocks for DCT processing.

        Args:
            x: Tensor of shape (B, C, H, W).
            block_size: Size of blocks (default 8).

        Returns:
            Tensor of shape (B, C, H//8, W//8, 8, 8).
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H // block_size, block_size, W // block_size, block_size)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (B, C, H//8, W//8, 8, 8)
        return x

    def _deblockify(self, x: Tensor, H: int, W: int, block_size: int = 8) -> Tensor:
        """Reshape blocks back to image.

        Args:
            x: Tensor of shape (B, C, H//8, W//8, 8, 8).
            H: Original image height.
            W: Original image width.
            block_size: Size of blocks (default 8).

        Returns:
            Tensor of shape (B, C, H, W).
        """
        B, C = x.shape[:2]
        x = x.permute(0, 1, 2, 4, 3, 5)  # (B, C, H//8, 8, W//8, 8)
        x = x.contiguous().view(B, C, H, W)
        return x

    def _dct_2d(self, blocks: Tensor) -> Tensor:
        """Apply 2D DCT to 8x8 blocks.

        Uses matrix multiplication: DCT * block * DCT^T

        Args:
            blocks: Tensor of shape (B, C, H//8, W//8, 8, 8).

        Returns:
            DCT coefficients of same shape.
        """
        # DCT: D @ x @ D^T
        # Using einsum for efficient batched matrix multiplication
        dct = self.dct_matrix
        return torch.einsum("ij,bcnmjk,lk->bcnmil", dct, blocks, dct)

    def _idct_2d(self, blocks: Tensor) -> Tensor:
        """Apply 2D inverse DCT to 8x8 blocks.

        Uses matrix multiplication: DCT^T * block * DCT

        Args:
            blocks: DCT coefficients of shape (B, C, H//8, W//8, 8, 8).

        Returns:
            Spatial domain blocks of same shape.
        """
        # IDCT: D^T @ x @ D
        dct = self.dct_matrix
        return torch.einsum("ji,bcnmjk,kl->bcnmil", dct, blocks, dct)

    def _quantize(self, dct_blocks: Tensor) -> Tensor:
        """Quantize DCT coefficients with straight-through estimator.

        Uses the straight-through estimator trick:
            quantized + (rounded - quantized).detach()

        This allows gradients to flow through the quantization step
        during backpropagation.

        Args:
            dct_blocks: DCT coefficients of shape (B, C, H//8, W//8, 8, 8).

        Returns:
            Quantized coefficients of same shape.
        """
        # Divide by quantization matrix
        q_matrix = self.q_matrix.view(1, 1, 1, 1, 8, 8)
        divided = dct_blocks / q_matrix

        # Round with straight-through estimator
        rounded = torch.round(divided)
        # Straight-through: gradient flows through as if no rounding
        quantized = divided + (rounded - divided).detach()

        # Multiply back by quantization matrix
        return quantized * q_matrix

    def forward(self, x: Tensor) -> Tensor:
        """Apply differentiable JPEG compression.

        Args:
            x: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Compressed tensor of same shape, clamped to [0, 1].
        """
        if self.intensity == 0.0:
            return x

        B, C, H, W = x.shape

        # Pad to multiple of 8
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_padded = x

        _, _, H_pad, W_pad = x_padded.shape

        # Convert to YCbCr
        ycbcr = self._rgb_to_ycbcr(x_padded)

        # JPEG works with values in [0, 255] then shifts by -128
        # Scale to [0, 255] and shift to [-128, 127]
        ycbcr_scaled = ycbcr * 255.0 - 128.0

        # Reshape into 8x8 blocks
        blocks = self._blockify(ycbcr_scaled, 8)

        # Apply 2D DCT
        dct_blocks = self._dct_2d(blocks)

        # Quantize with straight-through estimator
        quantized = self._quantize(dct_blocks)

        # Apply 2D inverse DCT
        reconstructed = self._idct_2d(quantized)

        # Reshape back to image
        ycbcr_reconstructed = self._deblockify(reconstructed, H_pad, W_pad, 8)

        # Shift back to [0, 255] and scale to [0, 1]
        ycbcr_reconstructed = (ycbcr_reconstructed + 128.0) / 255.0

        # Convert back to RGB
        rgb_reconstructed = self._ycbcr_to_rgb(ycbcr_reconstructed)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            rgb_reconstructed = rgb_reconstructed[:, :, :H, :W]

        # Blend with original based on intensity
        output = x + self.intensity * (rgb_reconstructed - x)

        # Clamp to valid range
        return torch.clamp(output, 0.0, 1.0)

    def sample_parameters(self) -> dict:
        """Sample random compression parameters.

        Samples quality in a range that produces visible artifacts.
        """
        # Sample quality: lower = more artifacts
        quality = int(10 + torch.rand(1).item() * 80)  # 10-90
        return {"quality": quality}
