"""StegaStamp encoder network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBlock(nn.Module):
    """Conv -> ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.conv(x))


class UpBlock(nn.Module):
    """Upsample -> Conv -> ReLU with skip connection."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up_conv = nn.Conv2d(in_ch, out_ch, 2, padding=0)
        self.conv = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up_conv(F.pad(x, (0, 1, 0, 1))))  # Pad to match 2x2 conv
        x = torch.cat([x, skip], dim=1)
        return F.relu(self.conv(x))


class Encoder(nn.Module):
    """U-Net encoder that embeds a bit message into an image.

    Args:
        num_bits: Number of bits in the message (default: 100).
    """

    def __init__(self, num_bits: int = 100) -> None:
        super().__init__()
        self.num_bits = num_bits

        # Message preparation: num_bits -> 7500 -> (50, 50, 3) -> upsample to (400, 400, 3)
        self.msg_fc = nn.Linear(num_bits, 50 * 50 * 3)

        # Encoder (downsampling path)
        # Input: 6 channels (image + message)
        self.conv1 = ConvBlock(6, 32)
        self.conv2 = ConvBlock(32, 32, stride=2)  # 200x200
        self.conv3 = ConvBlock(32, 64, stride=2)  # 100x100
        self.conv4 = ConvBlock(64, 128, stride=2)  # 50x50
        self.conv5 = ConvBlock(128, 256, stride=2)  # 25x25

        # Decoder (upsampling path with skip connections)
        self.up6 = UpBlock(256, 128, 128)  # 50x50
        self.up7 = UpBlock(128, 64, 64)  # 100x100
        self.up8 = UpBlock(64, 32, 32)  # 200x200
        self.up9 = UpBlock(32, 32, 32)  # 400x400

        self.conv_out1 = ConvBlock(32, 32)
        self.conv_out2 = nn.Conv2d(32, 3, 1)  # 1x1 conv to 3 channels

    def prepare_message(self, message: Tensor) -> Tensor:
        """Expand message bits to spatial feature map.

        Args:
            message: (B, num_bits) binary tensor

        Returns:
            (B, 3, 400, 400) spatial tensor
        """
        x = F.relu(self.msg_fc(message))  # (B, 7500)
        x = x.view(-1, 3, 50, 50)  # (B, 3, 50, 50)
        x = F.interpolate(x, size=(400, 400), mode="nearest")  # (B, 3, 400, 400)
        return x

    def forward(self, image: Tensor, message: Tensor) -> Tensor:
        """Encode message into image.

        Args:
            image: (B, 3, 400, 400) in [0, 1]
            message: (B, num_bits) binary tensor

        Returns:
            Encoded image (B, 3, 400, 400) in [0, 1]
        """
        # Prepare message and concatenate with image
        msg_spatial = self.prepare_message(message)
        x = torch.cat([image, msg_spatial], dim=1)  # (B, 6, 400, 400)

        # Encoder path (save activations for skip connections)
        c1 = self.conv1(x)  # (B, 32, 400, 400)
        c2 = self.conv2(c1)  # (B, 32, 200, 200)
        c3 = self.conv3(c2)  # (B, 64, 100, 100)
        c4 = self.conv4(c3)  # (B, 128, 50, 50)
        c5 = self.conv5(c4)  # (B, 256, 25, 25)

        # Decoder path with skip connections
        x = self.up6(c5, c4)  # (B, 128, 50, 50)
        x = self.up7(x, c3)  # (B, 64, 100, 100)
        x = self.up8(x, c2)  # (B, 32, 200, 200)
        x = self.up9(x, c1)  # (B, 32, 400, 400)

        # Output layers
        x = self.conv_out1(x)
        residual = self.conv_out2(x)  # (B, 3, 400, 400)

        # Add residual to original image and clamp
        encoded = torch.clamp(image + residual, 0.0, 1.0)
        return encoded
