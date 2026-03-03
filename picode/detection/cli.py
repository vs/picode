"""CLI for blind steganographic image detection."""

import json
from pathlib import Path

import click
import torch

from picode.detection import Detector
from picode.models.stegastamp import Decoder


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "--checkpoint",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to decoder checkpoint file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output JSON file for results.",
)
@click.option(
    "--save-crop",
    type=click.Path(),
    help="Save detected region to image file.",
)
@click.option(
    "--threshold",
    default=0.15,
    type=float,
    help="Confidence threshold (0-0.5).",
)
@click.option(
    "--scales",
    default="0.25,0.35,0.5,0.65,0.75",
    help="Comma-separated detection scales.",
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device for inference (cuda/cpu).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed progress.",
)
def main(
    input: str,
    checkpoint: str,
    output: str | None,
    save_crop: str | None,
    threshold: float,
    scales: str,
    device: str,
    verbose: bool,
) -> None:
    """Detect steganographic images in photos.

    INPUT is the path to an image file.
    """
    input_path = Path(input)
    checkpoint_path = Path(checkpoint)

    # Parse scales
    scale_list = [float(s.strip()) for s in scales.split(",")]

    if verbose:
        click.echo(f"Loading decoder from {checkpoint_path}")

    # Load decoder
    decoder = Decoder(num_bits=100)
    decoder.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    decoder.to(device)
    decoder.eval()

    # Create detector
    detector = Detector(
        decoder=decoder,
        scales=scale_list,
        confidence_threshold=threshold,
        device=device,
    )

    if verbose:
        click.echo(f"Detecting in {input_path}")
        click.echo(f"Scales: {scale_list}, Threshold: {threshold}")

    # Run detection
    result = detector.detect(input_path)

    # Format output
    output_data: dict[str, object] = {
        "input": str(input_path),
        "detections": [],
    }

    if result:
        detection_dict = {
            "bbox": list(result.bbox),
            "confidence": result.confidence,
            "message_bits": result.message_bits.tolist(),
        }
        detections_list: list[dict[str, object]] = []
        detections_list.append(detection_dict)
        output_data["detections"] = detections_list

        click.echo("Detection found!")
        click.echo(f"  Bounding box: {result.bbox}")
        click.echo(f"  Confidence: {result.confidence:.4f}")

        # Save crop if requested
        if save_crop:
            from PIL import Image

            img = Image.open(input_path).convert("RGB")
            x, y, w, h = result.bbox
            crop = img.crop((x, y, x + w, y + h))
            crop.save(save_crop)
            click.echo(f"  Saved crop to: {save_crop}")
    else:
        click.echo("No detection found above threshold.")

    # Write JSON output if requested
    if output:
        output_path = Path(output)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        click.echo(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
