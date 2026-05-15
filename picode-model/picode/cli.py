"""CLI for encoding and decoding hidden messages in images."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.utils import save_image

from picode.models.factory import create_decoder, create_encoder
from picode.training.config import ModelConfig


def load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    """Load encoder and decoder from checkpoint.

    Returns:
        Tuple of (encoder, decoder, num_bits, image_size).
    """
    data = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = data.get("config", {})
    num_bits = config.get("training", {}).get("num_bits", 100)
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "stegastamp")
    image_size = model_cfg.get("encoder_size", 400)

    mc = ModelConfig(
        type=model_type,
        encoder_size=model_cfg.get("encoder_size", 400),
        decoder_size=model_cfg.get("decoder_size", 400),
    )
    encoder = create_encoder(mc, num_bits=num_bits).to(device)
    decoder = create_decoder(mc, num_bits=num_bits).to(device)

    encoder.load_state_dict(data["encoder_state"])
    decoder.load_state_dict(data["decoder_state"])

    encoder.eval()
    decoder.eval()

    return encoder, decoder, num_bits, image_size


def text_to_bits(text: str, num_bits: int) -> list[int]:
    """Convert text to binary representation."""
    text_bytes = text.encode("utf-8")
    bits = []
    for byte in text_bytes:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)

    if len(bits) < num_bits:
        bits.extend([0] * (num_bits - len(bits)))
    else:
        bits = bits[:num_bits]

    return bits


def bits_to_text(bits: list[int]) -> str:
    """Convert binary representation back to text."""
    text_bytes = []
    for i in range(0, len(bits), 8):
        if i + 8 > len(bits):
            break
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(bits[i + j])
        if byte == 0:
            break
        text_bytes.append(byte)

    try:
        return bytes(text_bytes).decode("utf-8")
    except UnicodeDecodeError:
        return bytes(text_bytes).decode("utf-8", errors="replace")


def get_device(device_str: str) -> torch.device:
    """Get torch device from string."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def encode_command(args: argparse.Namespace) -> None:
    """Handle encode subcommand."""
    device = get_device(args.device)
    encoder, decoder, num_bits, image_size = load_model(args.checkpoint, device)
    size = args.size or image_size

    # Check model type from checkpoint
    data = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model_type = data.get("config", {}).get("model", {}).get("type", "stegastamp")

    # Convert message to bits
    message = args.message
    if message.startswith("0b") or all(c in "01" for c in message):
        bits = [int(c) for c in message.replace("0b", "")]
        if len(bits) < num_bits:
            bits.extend([0] * (num_bits - len(bits)))
        else:
            bits = bits[:num_bits]
    else:
        bits = text_to_bits(message, num_bits)

    message_tensor = torch.tensor(bits, dtype=torch.float32, device=device).unsqueeze(0)

    if model_type == "picodeframe":
        # PicodeFrame: resize to inner size, reflection-pad, encode with frame_width
        frame_pct = args.frame_pct or 0.04
        frame_width = int(size * frame_pct)
        inner_size = size - 2 * frame_width

        # Load and resize image to inner size
        image = Image.open(args.input).convert("RGB")
        image_cropped = ImageOps.fit(image, (inner_size, inner_size), method=Image.LANCZOS)

        to_tensor = transforms.ToTensor()
        inner_tensor = to_tensor(image_cropped).unsqueeze(0).to(device)

        # Reflection-pad to full size
        import torch.nn.functional as F
        padded_tensor = F.pad(inner_tensor, (frame_width,) * 4, mode="reflect")

        with torch.no_grad():
            encoded = encoder(padded_tensor, message_tensor, frame_width=frame_width)

        # Save encoded image
        save_image(encoded, args.output)

        # Save original if requested (the inner image at full output size)
        if args.save_original:
            save_image(padded_tensor, args.save_original)
            print(f"Saved original to {args.save_original}")

        # Save residual if requested
        if args.save_residual:
            residual = encoded - padded_tensor
            residual_vis = (residual * 10 + 0.5).clamp(0, 1)
            save_image(residual_vis, args.save_residual)
            print(f"Saved residual to {args.save_residual}")

        print(f"Encoded message into {args.output} ({size}x{size})")
        print(f"Frame width: {frame_width}px ({frame_pct*100:.0f}%)")
        print(f"Inner image: {inner_size}x{inner_size}")
    else:
        # StegaStamp/PicodeLite: standard full-image encoding
        image = Image.open(args.input).convert("RGB")
        image_cropped = ImageOps.fit(image, (size, size), method=Image.LANCZOS)

        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(image_cropped).unsqueeze(0).to(device)

        with torch.no_grad():
            encoded = encoder(image_tensor, message_tensor)

        save_image(encoded, args.output)

        if args.save_original:
            save_image(image_tensor, args.save_original)
            print(f"Saved original to {args.save_original}")

        if args.save_residual:
            residual = encoded - image_tensor
            residual_vis = (residual * 10 + 0.5).clamp(0, 1)
            save_image(residual_vis, args.save_residual)
            print(f"Saved residual to {args.save_residual}")

        print(f"Encoded message into {args.output} ({size}x{size})")

    print(f"Message: {message}")
    print(f"Bits used: {num_bits}")


def decode_command(args: argparse.Namespace) -> None:
    """Handle decode subcommand."""
    device = get_device(args.device)
    _, decoder, num_bits, image_size = load_model(args.checkpoint, device)
    size = args.size or image_size

    # Check model type from checkpoint
    data = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model_type = data.get("config", {}).get("model", {}).get("type", "stegastamp")

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    image = Image.open(args.input).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_type == "picodeframe":
            # PicodeFrame: create mask and pass to decoder for center-masking
            import torch.nn.functional as F
            frame_pct = args.frame_pct or 0.04
            frame_width = int(size * frame_pct)
            mask = torch.zeros(1, 1, size, size, device=device)
            mask[:, :, frame_width:size - frame_width, frame_width:size - frame_width] = 1.0
            decoded_logits = decoder(image_tensor, mask=mask)
        else:
            decoded_logits = decoder(image_tensor)
        decoded_bits = (decoded_logits > 0).float().squeeze(0).cpu().numpy().tolist()

    decoded_bits = [int(b) for b in decoded_bits]
    bits_str = "".join(str(b) for b in decoded_bits)
    decoded_text = bits_to_text(decoded_bits)

    if args.raw:
        print(bits_str)
    else:
        print(f"Decoded from: {args.input}")
        print(f"Text: {decoded_text}")
        print(f"Bits: {bits_str[:50]}..." if len(bits_str) > 50 else f"Bits: {bits_str}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="picode",
        description="Encode and decode hidden messages in images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encode a text message
  picode encode input.jpg output.png -m "Hello World"

  # Encode a binary message
  picode encode input.jpg output.png -m "1010101010"

  # Decode a message
  picode decode encoded.png

  # Use custom checkpoint
  picode encode input.jpg output.png -m "Secret" -c path/to/checkpoint.pt
""",
    )

    parser.add_argument(
        "-c", "--checkpoint",
        type=Path,
        default=Path("checkpoints/modal_training/picode_coco/best.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device (auto, cpu, cuda, mps)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Output image size; input is center-cropped to square (default: from checkpoint)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Encode subcommand
    encode_parser = subparsers.add_parser("encode", help="Encode a message into an image")
    encode_parser.add_argument("input", type=Path, help="Input image path")
    encode_parser.add_argument("output", type=Path, help="Output image path")
    encode_parser.add_argument("-m", "--message", required=True, help="Message to encode")
    encode_parser.add_argument("--save-original", type=Path, help="Save original image to path")
    encode_parser.add_argument(
        "--save-residual", type=Path, help="Save residual (amplified) to path"
    )
    encode_parser.add_argument(
        "--frame-pct", type=float, default=None, dest="frame_pct",
        help="Frame width as fraction of image (PicodeFrame only, default: 0.04)"
    )
    encode_parser.set_defaults(func=encode_command)

    # Decode subcommand
    decode_parser = subparsers.add_parser("decode", help="Decode a message from an image")
    decode_parser.add_argument("input", type=Path, help="Input image path")
    decode_parser.add_argument("--raw", action="store_true", help="Show raw bits only")
    decode_parser.add_argument(
        "--frame-pct", type=float, default=None, dest="frame_pct",
        help="Frame width as fraction of image (PicodeFrame only, default: 0.04)"
    )
    decode_parser.set_defaults(func=decode_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
