"""Generate annotated sample images from a checkpoint at multiple strengths."""

import torch
import torch.nn.functional as F
import numpy as np
import math
import glob
import os
import argparse
from picode.training.config import ModelConfig
from picode.models.factory import create_encoder, create_decoder
from picode.distortions.native.compression import JPEGCompression
from picode.distortions.native.blur import GaussianBlur
from picode.distortions.native.noise import GaussianNoise
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--dir", default="data/train")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strengths", default="0.010,0.015,0.020,0.025")
    args = parser.parse_args()

    data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ModelConfig(**data["config"].get("model", {}))
    tc = data["config"].get("training", {})
    num_bits = tc.get("num_bits", 72)
    dec_blur = tc.get("decoder_blur_sigma", 0.5)
    is_film = tc.get("strength_conditioned", False)

    dec = create_decoder(mc, num_bits=num_bits)
    dec.load_state_dict(data["decoder_state"])
    dec.eval()

    # Decoder blur kernel (skip if sigma=0)
    sigma = dec_blur
    if sigma > 0:
        k = 2 * math.ceil(3 * sigma) + 1
        ax = torch.arange(k, dtype=torch.float32) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kern = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kern = (kern / kern.sum()).view(1, 1, k, k).expand(3, -1, -1, -1).contiguous()
        bp = k // 2
    else:
        kern = None
        bp = 0

    tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    msg = torch.zeros(1, num_bits)
    for i, c in enumerate("Test message 64bit".encode("utf-8")):
        for b in range(8):
            idx = i * 8 + b
            if idx < num_bits:
                msg[0, idx] = float((c >> (7 - b)) & 1)

    def decode_acc(img):
        if kern is not None:
            img = F.conv2d(img, kern, padding=bp, groups=3)
        return (
            (torch.sigmoid(dec(img)) > 0.5).float() == msg
        ).float().mean().item()

    def compute_trc(img, res):
        gray = img.mean(dim=1, keepdim=True)
        kb = torch.ones(1, 1, 7, 7) / 49.0
        lm = F.conv2d(gray, kb, padding=3)
        lv = (F.conv2d(gray ** 2, kb, padding=3) - lm ** 2).clamp(min=0)
        t = (lv > lv.median()).float().expand_as(res)
        re = res.pow(2)
        return ((re * t).sum() / (re.sum() + 1e-8)).item()

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11)
    except Exception:
        font = ImageFont.load_default()

    strengths = [float(s) for s in args.strengths.split(",")]
    outdir = args.output or f"data/{os.path.splitext(os.path.basename(args.checkpoint))[0]}_samples"
    os.makedirs(outdir, exist_ok=True)

    imgs = sorted(glob.glob(f"{args.dir}/sample*.jpg"))[:args.max_images]
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {num_bits} bits, dec_blur={dec_blur}, film={is_film}")
    print(f"Strengths: {strengths}")
    print(f"Images: {len(imgs)}, Output: {outdir}")

    for imgpath in imgs:
        name = os.path.splitext(os.path.basename(imgpath))[0]
        img = tf(Image.open(imgpath).convert("RGB")).unsqueeze(0)
        Image.fromarray(
            (img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        ).save(f"{outdir}/{name}_0_original.png")

        for si, sv in enumerate(strengths):
            enc = create_encoder(
                mc, num_bits=num_bits, strength=sv,
                strength_conditioned=is_film,
            )
            enc.load_state_dict(data["encoder_state"], strict=False)
            enc.eval()
            with torch.no_grad():
                e = enc(img, msg)["encoded"].clamp(0, 1)
                r = e - img
                psnr = 10 * torch.log10(1.0 / F.mse_loss(e, img)).item()
                tc_val = compute_trc(img, r)
                ac = decode_acc(e)
                a10 = decode_acc(JPEGCompression(quality=10)(e))
                a50 = decode_acc(JPEGCompression(quality=50)(e))
                ab = decode_acc(GaussianBlur(sigma=3.0)(e))
                an = decode_acc(GaussianNoise(std=0.1)(e))

            sl = f"s{int(sv * 1000):03d}"
            rv = (0.5 + r[0].permute(1, 2, 0).numpy() * 10).clip(0, 1)
            Image.fromarray((rv * 255).astype(np.uint8)).save(
                f"{outdir}/{name}_{si * 2 + 2}_{sl}_residual.png"
            )
            ep = Image.fromarray(
                (e[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            d = ImageDraw.Draw(ep)
            lines = [
                f"PSNR: {psnr:.1f} dB  TRC: {tc_val:.3f}  s={sv}  blur={dec_blur}",
                f"Clean: {ac * 100:.0f}%  JPEG10: {a10 * 100:.0f}%  JPEG50: {a50 * 100:.0f}%",
                f"Blur3: {ab * 100:.0f}%  Noise0.1: {an * 100:.0f}%",
            ]
            y = 4
            for line in lines:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    d.text((6 + dx, y + dy), line, fill="black", font=font)
                d.text((6, y), line, fill="white", font=font)
                y += 15
            ep.save(f"{outdir}/{name}_{si * 2 + 1}_{sl}_encoded.png")

        print(f"{name}: done")
    print("All done")


if __name__ == "__main__":
    main()
