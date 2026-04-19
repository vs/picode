"""Compression distortions.

Implements differentiable JPEG compression matching StegaStamp's
utils.py jpeg_compress_decompress function.

The key innovation is using a straight-through estimator to allow
gradients to flow through the quantization step.

StegaStamp-specific features:
- Separate Y (luminance) and C (chroma) quantization tables
- 4:2:0 chroma subsampling (Cb/Cr downsampled by 2x)
- round_only_at_0 differentiable rounding function
"""

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from picode.distortions.base import Distortion

# Y (luminance) quantization table - JPEG standard (transposed for column-major access)
Y_TABLE = torch.tensor(
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
).T

# C (chroma) quantization table - much coarser than Y (transposed for column-major access)
C_TABLE = torch.full((8, 8), 99, dtype=torch.float32)
C_TABLE[:4, :4] = torch.tensor(
    [
        [17, 18, 24, 47],
        [18, 21, 26, 66],
        [24, 26, 56, 99],
        [47, 66, 99, 99],
    ],
    dtype=torch.float32,
).T


def round_only_at_0(x: Tensor) -> Tensor:
    """Differentiable rounding that only affects values near 0.

    StegaStamp's default rounding function. For values with |x| < 0.5,
    uses x^3 (which approaches 0). For values with |x| >= 0.5, leaves
    unchanged. This provides differentiable behavior while still
    quantizing small values to zero.

    Args:
        x: Input tensor (typically DCT coefficients divided by quantization matrix).

    Returns:
        Rounded tensor with same shape.
    """
    cond = (x.abs() < 0.5).float()
    return cond * (x**3) + (1 - cond) * x


def downsample_420(
    y: Tensor, cb: Tensor, cr: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """4:2:0 chroma subsampling - downsample Cb/Cr by 2x.

    Real JPEG uses 4:2:0 subsampling where chroma channels are at half
    resolution. This simulates the chroma information loss that occurs
    in standard JPEG compression.

    Args:
        y: Luminance tensor of shape (B, H, W).
        cb: Cb chroma tensor of shape (B, H, W).
        cr: Cr chroma tensor of shape (B, H, W).

    Returns:
        Tuple of (y, cb_down, cr_down) where cb_down and cr_down have
        shape (B, H//2, W//2).
    """
    cb_down = F.avg_pool2d(cb.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
    cr_down = F.avg_pool2d(cr.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
    return y, cb_down, cr_down


def upsample_420(
    y: Tensor, cb: Tensor, cr: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Upsample Cb/Cr back to full resolution (nearest neighbor).

    Args:
        y: Luminance tensor of shape (B, H, W).
        cb: Cb chroma tensor of shape (B, H//2, W//2).
        cr: Cr chroma tensor of shape (B, H//2, W//2).

    Returns:
        Tuple of (y, cb_up, cr_up) where cb_up and cr_up have
        shape (B, H, W) matching y.
    """
    # Use repeat_interleave for nearest neighbor upsampling (2x in each dimension)
    cb_up = cb.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    cr_up = cr.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    return y, cb_up, cr_up


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


def _scale_quantization_table(base_matrix: Tensor, quality: int) -> Tensor:
    """Scale a quantization matrix by JPEG quality factor.

    Args:
        base_matrix: Base quantization matrix of shape (8, 8).
        quality: JPEG quality (1-100). Lower means more compression.

    Returns:
        Scaled quantization matrix of shape (8, 8).
    """
    # Scale quality factor (JPEG standard scaling)
    if quality < 50:
        scale = 5000.0 / quality
    else:
        scale = 200.0 - 2.0 * quality

    # Apply scaling and clamp to valid range
    q_matrix = torch.floor((base_matrix * scale + 50.0) / 100.0)
    q_matrix = torch.clamp(q_matrix, 1.0, 255.0)

    return q_matrix


def _get_jpeg_quantization_matrix(quality: int) -> Tensor:
    """Get standard JPEG luminance quantization matrix scaled by quality.

    Args:
        quality: JPEG quality (1-100). Lower means more compression.

    Returns:
        Quantization matrix of shape (8, 8).
    """
    return _scale_quantization_table(Y_TABLE, quality)


class JPEGCompression(Distortion):
    """Differentiable JPEG compression distortion.

    Implements DCT-based JPEG compression with straight-through estimator
    for gradient flow through quantization. Matches StegaStamp's
    jpeg_compress_decompress implementation.

    Pipeline:
        RGB -> YCbCr -> [4:2:0 downsample] -> DCT -> Quantize -> IDCT
        -> [4:2:0 upsample] -> RGB

    StegaStamp-specific features:
        - Separate Y and C quantization tables (chroma is more heavily quantized)
        - 4:2:0 chroma subsampling when downsample_c=True
        - round_only_at_0 differentiable rounding by default

    Args:
        intensity: Strength of effect (0.0 to 1.0).
        quality: JPEG quality factor (1-100, default 50).
            Lower values produce more compression artifacts.
        downsample_c: Whether to apply 4:2:0 chroma subsampling (default True).
            This matches real JPEG behavior where chroma is at half resolution.
        rounding: Rounding function to use. Options:
            - "round_only_at_0" (default): StegaStamp's differentiable rounding
            - "straight_through": Standard straight-through estimator (torch.round)
    """

    name = "jpeg_compression"
    dct_matrix: Tensor
    y_q_matrix: Tensor
    c_q_matrix: Tensor

    def __init__(
        self,
        intensity: float = 0.5,
        quality: int = 50,
        downsample_c: bool = True,
        rounding: str = "round_only_at_0",
    ):
        super().__init__(intensity)
        self.quality = max(1, min(100, quality))
        self.downsample_c = downsample_c
        self.rounding = rounding

        # Create and register DCT matrix as buffer (not parameter)
        dct_matrix = _create_dct_matrix(8)
        self.register_buffer("dct_matrix", dct_matrix)

        # Create and register Y (luminance) quantization matrix
        y_q_matrix = _scale_quantization_table(Y_TABLE, self.quality)
        self.register_buffer("y_q_matrix", y_q_matrix)

        # Create and register C (chroma) quantization matrix
        c_q_matrix = _scale_quantization_table(C_TABLE, self.quality)
        self.register_buffer("c_q_matrix", c_q_matrix)

        # For backward compatibility, keep q_matrix as alias to y_q_matrix
        self.register_buffer("q_matrix", y_q_matrix.clone())

        # Select rounding function
        self._round_fn: Callable[[Tensor], Tensor]
        if rounding == "round_only_at_0":
            self._round_fn = round_only_at_0
        elif rounding == "straight_through":
            self._round_fn = self._straight_through_round
        else:
            raise ValueError(f"Unknown rounding mode: {rounding}")

    @staticmethod
    def _straight_through_round(x: Tensor) -> Tensor:
        """Standard straight-through estimator rounding."""
        rounded = torch.round(x)
        return x + (rounded - x).detach()

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
        dct = self.dct_matrix.to(blocks.device)
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
        dct = self.dct_matrix.to(blocks.device)
        return torch.einsum("ji,bcnmjk,kl->bcnmil", dct, blocks, dct)

    def _quantize_channel(self, dct_blocks: Tensor, q_matrix: Tensor) -> Tensor:
        """Quantize DCT coefficients for a single channel.

        Args:
            dct_blocks: DCT coefficients of shape (B, H//8, W//8, 8, 8).
            q_matrix: Quantization matrix of shape (8, 8).

        Returns:
            Quantized coefficients of same shape.
        """
        q = q_matrix.to(dct_blocks.device).view(1, 1, 1, 8, 8)
        divided = dct_blocks / q

        # Apply configurable rounding function
        rounded = self._round_fn(divided)

        # Multiply back by quantization matrix
        return rounded * q

    def _quantize(self, dct_blocks: Tensor) -> Tensor:
        """Quantize DCT coefficients with separate Y/C tables.

        Uses the configurable rounding function (round_only_at_0 by default)
        and applies different quantization tables for luminance and chroma.

        Args:
            dct_blocks: DCT coefficients of shape (B, C, H//8, W//8, 8, 8).

        Returns:
            Quantized coefficients of same shape.
        """
        # Split into Y and C channels
        y_blocks = dct_blocks[:, 0]  # (B, H//8, W//8, 8, 8)
        cb_blocks = dct_blocks[:, 1]
        cr_blocks = dct_blocks[:, 2]

        # Quantize Y with luminance table
        y_quantized = self._quantize_channel(y_blocks, self.y_q_matrix)

        # Quantize Cb/Cr with chroma table
        cb_quantized = self._quantize_channel(cb_blocks, self.c_q_matrix)
        cr_quantized = self._quantize_channel(cr_blocks, self.c_q_matrix)

        # Recombine
        return torch.stack([y_quantized, cb_quantized, cr_quantized], dim=1)

    def _process_channel(
        self, channel: Tensor, q_matrix: Tensor, H: int, W: int
    ) -> Tensor:
        """Process a single channel through DCT/quantize/IDCT pipeline.

        Args:
            channel: Input tensor of shape (B, H, W) in [-128, 127].
            q_matrix: Quantization matrix of shape (8, 8).
            H: Padded height (multiple of 8).
            W: Padded width (multiple of 8).

        Returns:
            Reconstructed channel of shape (B, H, W).
        """
        # Add channel dimension for blockify: (B, 1, H, W)
        channel_4d = channel.unsqueeze(1)

        # Reshape into 8x8 blocks: (B, 1, H//8, W//8, 8, 8)
        blocks = self._blockify(channel_4d, 8)

        # Apply 2D DCT
        dct_blocks = self._dct_2d(blocks)

        # Quantize: squeeze out channel dim for _quantize_channel
        dct_squeezed = dct_blocks.squeeze(1)  # (B, H//8, W//8, 8, 8)
        quantized = self._quantize_channel(dct_squeezed, q_matrix)
        quantized = quantized.unsqueeze(1)  # (B, 1, H//8, W//8, 8, 8)

        # Apply 2D inverse DCT
        reconstructed = self._idct_2d(quantized)

        # Reshape back to image: (B, 1, H, W) -> (B, H, W)
        return self._deblockify(reconstructed, H, W, 8).squeeze(1)

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

        # Pad to multiple of 16 when using 4:2:0 subsampling (chroma is half res)
        # Otherwise pad to multiple of 8
        pad_multiple = 16 if self.downsample_c else 8
        pad_h = (pad_multiple - H % pad_multiple) % pad_multiple
        pad_w = (pad_multiple - W % pad_multiple) % pad_multiple
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

        # Extract Y, Cb, Cr channels: (B, H, W) each
        y_channel = ycbcr_scaled[:, 0]
        cb_channel = ycbcr_scaled[:, 1]
        cr_channel = ycbcr_scaled[:, 2]

        if self.downsample_c:
            # 4:2:0 chroma subsampling
            y_channel, cb_down, cr_down = downsample_420(y_channel, cb_channel, cr_channel)

            # Process Y at full resolution
            y_reconstructed = self._process_channel(
                y_channel, self.y_q_matrix, H_pad, W_pad
            )

            # Process Cb/Cr at half resolution
            cb_reconstructed = self._process_channel(
                cb_down, self.c_q_matrix, H_pad // 2, W_pad // 2
            )
            cr_reconstructed = self._process_channel(
                cr_down, self.c_q_matrix, H_pad // 2, W_pad // 2
            )

            # Upsample chroma back to full resolution
            _, cb_reconstructed, cr_reconstructed = upsample_420(
                y_reconstructed, cb_reconstructed, cr_reconstructed
            )
        else:
            # Process all channels at full resolution (no subsampling)
            y_reconstructed = self._process_channel(
                y_channel, self.y_q_matrix, H_pad, W_pad
            )
            cb_reconstructed = self._process_channel(
                cb_channel, self.c_q_matrix, H_pad, W_pad
            )
            cr_reconstructed = self._process_channel(
                cr_channel, self.c_q_matrix, H_pad, W_pad
            )

        # Recombine channels: (B, 3, H, W)
        ycbcr_reconstructed = torch.stack(
            [y_reconstructed, cb_reconstructed, cr_reconstructed], dim=1
        )

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

    def sample_parameters(self) -> dict[str, Any]:
        """Sample random compression parameters.

        Samples quality in a range that produces visible artifacts.
        """
        # Sample quality: lower = more artifacts
        quality = int(10 + torch.rand(1).item() * 80)  # 10-90
        return {"quality": quality}
