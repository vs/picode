"""CLI for blind steganographic image detection."""

import json
from pathlib import Path

import click
import torch

from picode.detection import Detector
from picode.detection.fast_detector import FastDetector, FastDetectorModel
from picode.detection.pipeline import DetectionPipeline
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
    "--detector",
    "-d",
    type=click.Choice(["slow", "fast"]),
    default="slow",
    help="Detector type: 'slow' (exhaustive) or 'fast' (single-pass).",
)
@click.option(
    "--detector-checkpoint",
    type=click.Path(exists=True),
    help="Path to FastDetector checkpoint (required if --detector=fast).",
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
    help="Confidence threshold (0-0.5 for slow, 0-1 for fast).",
)
@click.option(
    "--scales",
    default="0.25,0.35,0.5,0.65,0.75",
    help="Comma-separated detection scales (slow detector only).",
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
@click.option(
    "--sample-rate",
    default=1,
    type=int,
    help="For video: process every Nth frame.",
)
@click.option(
    "--decode/--no-decode",
    default=True,
    help="Decode message from detected region (fast detector).",
)
def main(
    input: str,
    checkpoint: str,
    detector: str,
    detector_checkpoint: str | None,
    output: str | None,
    save_crop: str | None,
    threshold: float,
    scales: str,
    device: str,
    verbose: bool,
    sample_rate: int,
    decode: bool,
) -> None:
    """Detect steganographic images in photos or videos.

    INPUT is the path to an image or video file.

    Examples:

        # Slow detector (exhaustive, uses decoder)
        detect image.jpg -c decoder.pt

        # Fast detector (single-pass, mobile-optimized)
        detect image.jpg -c decoder.pt -d fast --detector-checkpoint detector.pt

        # Fast detector without decoding (detection only)
        detect image.jpg -d fast --detector-checkpoint detector.pt --no-decode
    """
    input_path = Path(input)
    checkpoint_path = Path(checkpoint)

    # Validate fast detector checkpoint
    if detector == "fast" and detector_checkpoint is None:
        raise click.UsageError("--detector-checkpoint required when using --detector=fast")

    if verbose:
        click.echo(f"Using {detector} detector")
        click.echo(f"Loading decoder from {checkpoint_path}")

    # Load decoder
    decoder_model = Decoder(num_bits=100)
    decoder_model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    decoder_model.to(device)
    decoder_model.eval()

    # Create detector based on type
    if detector == "slow":
        scale_list = [float(s.strip()) for s in scales.split(",")]
        det = Detector(
            decoder=decoder_model,
            scales=scale_list,
            confidence_threshold=threshold,
            device=device,
        )
        pipeline = None  # Slow detector handles decoding internally

        if verbose:
            click.echo(f"Scales: {scale_list}, Threshold: {threshold}")
    else:
        # Fast detector
        fast_model = FastDetectorModel(input_size=320, pretrained=False)
        ckpt = torch.load(detector_checkpoint, map_location=device)
        fast_model.load_state_dict(ckpt["model_state_dict"])

        fast_det = FastDetector(
            model=fast_model,
            threshold=threshold,
            device=device,
        )

        if decode:
            pipeline = DetectionPipeline(
                detector=fast_det,
                decoder=decoder_model,
                device=device,
            )
        else:
            pipeline = None
            det = fast_det  # Use fast detector directly

        if verbose:
            click.echo(f"Threshold: {threshold}")

    # Format output
    output_data: dict[str, object] = {
        "input": str(input_path),
        "detector": detector,
        "detections": [],
    }

    # Check if input is video
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    is_video = input_path.suffix.lower() in video_extensions

    if is_video:
        if detector == "fast":
            click.echo("Video processing not yet supported for fast detector.")
            return

        if verbose:
            click.echo("Processing video...")

        detections_list: list[dict[str, object]] = []
        for frame_num, result in det.detect_video(input_path, sample_rate=sample_rate):
            if result:
                detection_dict: dict[str, object] = {
                    "frame": frame_num,
                    "bbox": list(result.bbox),
                    "confidence": result.confidence,
                    "message_bits": result.message_bits.tolist(),
                }
                detections_list.append(detection_dict)

                if verbose:
                    click.echo(f"  Frame {frame_num}: confidence {result.confidence:.4f}")

        output_data["detections"] = detections_list
        click.echo(f"Processed video, found {len(detections_list)} detections.")
    else:
        # Image detection
        if detector == "slow":
            result = det.detect(input_path)

            if result:
                detection_dict = {
                    "bbox": list(result.bbox),
                    "confidence": result.confidence,
                    "message_bits": result.message_bits.tolist(),
                }
                output_data["detections"] = [detection_dict]

                click.echo("Detection found!")
                click.echo(f"  Bounding box: {result.bbox}")
                click.echo(f"  Confidence: {result.confidence:.4f}")

                if save_crop:
                    _save_crop(input_path, result.bbox, save_crop)
                    click.echo(f"  Saved crop to: {save_crop}")
            else:
                click.echo("No detection found above threshold.")
        else:
            # Fast detector
            if pipeline and decode:
                result = pipeline.process(input_path)
                if result:
                    detection_dict = result.to_dict()
                    output_data["detections"] = [detection_dict]

                    click.echo("Detection found!")
                    click.echo(f"  Bounding box: {result.detection.bbox}")
                    click.echo(f"  Detection confidence: {result.detection.confidence:.4f}")
                    click.echo(f"  Decode confidence: {result.decode_confidence:.4f}")

                    if save_crop:
                        _save_crop(input_path, result.detection.bbox, save_crop)
                        click.echo(f"  Saved crop to: {save_crop}")
                else:
                    click.echo("No detection found above threshold.")
            else:
                # Detection only, no decoding
                detection = fast_det.detect(input_path)
                if detection:
                    detection_dict = {
                        "bbox": list(detection.bbox),
                        "confidence": detection.confidence,
                        "corners": detection.corners.to_tensor().tolist(),
                    }
                    output_data["detections"] = [detection_dict]

                    click.echo("Detection found!")
                    click.echo(f"  Bounding box: {detection.bbox}")
                    click.echo(f"  Confidence: {detection.confidence:.4f}")

                    if save_crop:
                        _save_crop(input_path, detection.bbox, save_crop)
                        click.echo(f"  Saved crop to: {save_crop}")
                else:
                    click.echo("No detection found above threshold.")

    # Write JSON output if requested
    if output:
        output_path = Path(output)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        click.echo(f"Results saved to: {output_path}")


def _save_crop(input_path: Path, bbox: tuple[int, int, int, int], save_path: str) -> None:
    """Save cropped region to file."""
    from PIL import Image

    img = Image.open(input_path).convert("RGB")
    x, y, w, h = bbox
    crop = img.crop((x, y, x + w, y + h))
    crop.save(save_path)


if __name__ == "__main__":
    main()
