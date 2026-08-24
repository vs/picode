"""Render the README hero figure: original | encoded (with metrics) | residual.

Uses only public-domain / CC0 sample photos from scikit-image's data directory,
downloaded at run time, so no private images ever end up in the repository.
Encoding follows the production Sobel adaptive strategy (docs/sobel_adaptive.md):
raw residual at s=1.0, scaled by a Sobel texture mask, with batch ID selection
(the most confidently decoded of N random payloads) at the lowest strength on
the ladder where some payload decodes exactly.

Usage:
    python scripts/readme_figures.py CHECKPOINT [--output ../docs/assets/hero.jpg]
"""

import argparse
import io
import math
import urllib.request

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torchvision.transforms.functional import center_crop, resize, to_pil_image, to_tensor

from picode.distortions.native.blur import GaussianBlur
from picode.distortions.native.compression import JPEGCompression
from picode.distortions.native.noise import GaussianNoise
from picode.models.factory import create_decoder, create_encoder
from picode.training.config import ModelConfig

SKIMAGE_DATA = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data"
# All three are public domain or CC0 (see skimage/data/README.txt).
SAMPLES = ["rocket.jpg", "coffee.png", "chelsea.png"]
PANEL = 448
GUTTER = 12
HEADER = 28


def sobel_texture_mask(image: Tensor, sigma: float = 5.0, floor: float = 0.85) -> Tensor:
    """Sobel gradient texture mask in [floor, 1], shape (1, 1, H, W)."""
    gray = image.mean(dim=1, keepdim=True)
    sx = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    grad = (F.conv2d(gray, sx, padding=1) ** 2 + F.conv2d(gray, sx.mT, padding=1) ** 2).sqrt()
    k = 2 * math.ceil(3 * sigma) + 1
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    gk = (g[:, None] * g[None, :] / g.sum() ** 2).view(1, 1, k, k)
    smooth = F.conv2d(grad, gk, padding=k // 2)
    return floor + (1.0 - floor) * smooth / (smooth.amax() + 1e-8)


def texture_residual_correlation(image: Tensor, residual: Tensor) -> float:
    """Share of residual energy in above-median local-variance (textured) pixels."""
    gray = image.mean(dim=1, keepdim=True)
    kb = torch.ones(1, 1, 7, 7) / 49.0
    mean = F.conv2d(gray, kb, padding=3)
    var = (F.conv2d(gray**2, kb, padding=3) - mean**2).clamp(min=0)
    textured = (var > var.median()).float().expand_as(residual)
    energy = residual.pow(2)
    return (energy * textured).sum().item() / (energy.sum().item() + 1e-8)


def load_sample(name: str, size: int) -> Tensor:
    with urllib.request.urlopen(f"{SKIMAGE_DATA}/{name}", timeout=30) as resp:
        img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    img = center_crop(img, [min(img.size)] * 2)
    return to_tensor(resize(img, [size, size], antialias=True)).unsqueeze(0)


def draw_label(img: Image.Image, lines: list[str], font: ImageFont.ImageFont) -> None:
    draw = ImageDraw.Draw(img)
    y = 6
    for line in lines:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((8 + dx, y + dy), line, fill="black", font=font)
        draw.text((8, y), line, fill="white", font=font)
        y += 16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="../docs/assets/hero.jpg")
    parser.add_argument("--strengths", default="0.010,0.012,0.015,0.020")
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    ladder = [float(v) for v in args.strengths.split(",")]
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ModelConfig(**ckpt["config"].get("model", {}))
    tc = ckpt["config"].get("training", {})
    num_bits = tc.get("num_bits", 72)
    size = mc.encoder_size

    encoder = create_encoder(mc, num_bits=num_bits, strength=1.0)
    encoder.load_state_dict(ckpt["encoder_state"], strict=False)
    decoder = create_decoder(mc, num_bits=num_bits)
    decoder.load_state_dict(ckpt["decoder_state"])
    encoder.eval()
    decoder.eval()
    blur_sigma = float(tc.get("decoder_blur_sigma", 0.0))
    dec_blur = GaussianBlur(sigma=blur_sigma) if blur_sigma > 0 else None

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
        title_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16)
    except OSError:
        font = title_font = ImageFont.load_default()

    width = 3 * PANEL + 4 * GUTTER
    height = HEADER + len(SAMPLES) * (PANEL + GUTTER) + GUTTER
    canvas = Image.new("RGB", (width, height), "white")
    header = ImageDraw.Draw(canvas)
    for col, title in enumerate(["Original", f"Encoded ({num_bits} bits)", "Residual (x10)"]):
        header.text((GUTTER + col * (PANEL + GUTTER), 6), title, fill="black", font=title_font)

    for row, name in enumerate(SAMPLES):
        image = load_sample(name, size)
        mask = sobel_texture_mask(image)

        # Batch ID selection: encode N random payloads once at s=1.0, then walk the
        # strength ladder and keep the most confident payload that decodes exactly.
        with torch.no_grad():
            candidates = torch.randint(0, 2, (args.candidates, num_bits)).float()
            batch = image.expand(args.candidates, -1, -1, -1)
            out = encoder(batch, candidates)
            raw = (out["encoded"] if isinstance(out, dict) else out) - batch
            for strength in ladder:
                trial = (batch + strength * raw * mask).clamp(0, 1)
                probs = torch.sigmoid(decoder(dec_blur(trial) if dec_blur else trial))
                correct = ((probs > 0.5).float() == candidates).float()
                margin = ((probs - 0.5).abs() * correct).min(1).values
                best = int(margin.argmax())
                if margin[best] > 0:
                    break
        message = candidates[best : best + 1]
        residual = strength * raw[best : best + 1] * mask
        encoded = trial[best : best + 1]

        def bit_accuracy(img: Tensor, msg: Tensor = message) -> float:
            if dec_blur is not None:
                img = dec_blur(img)
            with torch.no_grad():
                bits = (torch.sigmoid(decoder(img)) > 0.5).float()
            return (bits == msg).float().mean().item()

        psnr = 10 * math.log10(1.0 / F.mse_loss(encoded, image).item())
        lines = [
            f"s={strength:.3f}  PSNR {psnr:.1f} dB  "
            f"TRC {texture_residual_correlation(image, residual):.3f}",
            f"clean {bit_accuracy(encoded):.0%}  "
            f"JPEG50 {bit_accuracy(JPEGCompression(quality=50)(encoded)):.0%}  "
            f"JPEG10 {bit_accuracy(JPEGCompression(quality=10)(encoded)):.0%}",
            f"blur3 {bit_accuracy(GaussianBlur(sigma=3.0)(encoded)):.0%}  "
            f"noise0.1 {bit_accuracy(GaussianNoise(std=0.1)(encoded)):.0%}  (raw bits, no ECC)",
        ]
        print(f"{name}: " + " | ".join(lines))

        panels = [
            to_pil_image(image[0]),
            to_pil_image(encoded[0]),
            to_pil_image((0.5 + 10 * (encoded - image)[0]).clamp(0, 1)),
        ]
        draw_label(panels[1], lines, font)
        y = HEADER + row * (PANEL + GUTTER)
        for col, panel in enumerate(panels):
            panel = panel.resize((PANEL, PANEL), Image.Resampling.LANCZOS)
            canvas.paste(panel, (GUTTER + col * (PANEL + GUTTER), y))

    if args.output.lower().endswith((".jpg", ".jpeg")):
        canvas.save(args.output, quality=90, optimize=True, progressive=True)
    else:
        canvas.save(args.output, optimize=True)
    print(f"Saved {args.output} ({width}x{height})")


if __name__ == "__main__":
    main()
