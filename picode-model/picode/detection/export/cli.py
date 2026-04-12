# picode/detection/export/cli.py
"""CLI for exporting FastDetector to mobile formats."""

from __future__ import annotations

from pathlib import Path

import click
import torch


@click.group()
def main() -> None:
    """Export FastDetector models to mobile formats."""
    pass


@main.command("coreml")
@click.argument("checkpoint", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option(
    "--input-size",
    default=320,
    type=int,
    help="Input image size (default: 320).",
)
@click.option(
    "--compute-units",
    default="ALL",
    type=click.Choice(["ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY"]),
    help="Core ML compute units (default: ALL).",
)
@click.option(
    "--min-ios",
    default="iOS15",
    type=click.Choice(["iOS13", "iOS14", "iOS15", "iOS16", "iOS17"]),
    help="Minimum iOS deployment target (default: iOS15).",
)
@click.option(
    "--fp16/--no-fp16",
    default=True,
    help="Convert weights to float16 (default: enabled).",
)
@click.option(
    "--int8/--no-int8",
    default=False,
    help="Quantize to int8 (default: disabled).",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Validate exported model (default: enabled).",
)
def export_coreml(
    checkpoint: str,
    output: str,
    input_size: int,
    compute_units: str,
    min_ios: str,
    fp16: bool,
    int8: bool,
    validate: bool,
) -> None:
    """Export FastDetector to Core ML format for iOS.

    CHECKPOINT is the path to the FastDetector checkpoint (.pt file).
    OUTPUT is the path for the exported .mlpackage file.

    Examples:

        # Basic export
        detect-export coreml detector.pt FastDetector.mlpackage

        # Export with int8 quantization
        detect-export coreml detector.pt FastDetector.mlpackage --int8

        # Export for older iOS versions
        detect-export coreml detector.pt FastDetector.mlpackage --min-ios iOS13
    """
    try:
        from picode.detection.export.coreml import (
            CoreMLExportConfig,
            convert_to_coreml,
            validate_coreml_model,
        )
    except ImportError as e:
        raise click.ClickException(
            "coremltools is required for Core ML export.\n"
            "Install it with: pip install coremltools"
        ) from e

    from picode.detection.fast_detector import FastDetectorModel

    click.echo(f"Loading checkpoint: {checkpoint}")

    # Load model
    model = FastDetectorModel(input_size=input_size, pretrained=False)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    # Configure export
    config = CoreMLExportConfig(
        input_size=input_size,
        compute_units=compute_units,
        minimum_deployment_target=min_ios,
        convert_to_fp16=fp16,
        quantize_to_int8=int8,
    )

    click.echo("Exporting to Core ML...")
    click.echo(f"  Input size: {input_size}x{input_size}")
    click.echo(f"  Compute units: {compute_units}")
    click.echo(f"  Min iOS: {min_ios}")
    click.echo(f"  FP16: {fp16}, INT8: {int8}")

    # Export
    output_path = convert_to_coreml(model, output, config)
    click.echo(f"Exported to: {output_path}")

    # Validate
    if validate:
        click.echo("Validating exported model...")
        try:
            results = validate_coreml_model(output_path, input_size=input_size)
            click.echo(f"  Valid: {results['valid']}")
            click.echo(f"  Inference time: {results['inference_time_ms']:.2f}ms")
            click.echo(f"  Output shapes: {results['output_shapes']}")
        except Exception as e:
            click.echo(f"  Validation failed: {e}", err=True)


@main.command("tflite")
@click.argument("checkpoint", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option(
    "--input-size",
    default=320,
    type=int,
    help="Input image size (default: 320).",
)
@click.option(
    "--fp16/--no-fp16",
    default=True,
    help="Quantize to float16 (default: enabled).",
)
@click.option(
    "--int8/--no-int8",
    default=False,
    help="Quantize to int8 (default: disabled).",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Validate exported model (default: enabled).",
)
def export_tflite(
    checkpoint: str,
    output: str,
    input_size: int,
    fp16: bool,
    int8: bool,
    validate: bool,
) -> None:
    """Export FastDetector to TFLite format for Android.

    CHECKPOINT is the path to the FastDetector checkpoint (.pt file).
    OUTPUT is the path for the exported .tflite file.

    Examples:

        # Basic export
        detect-export tflite detector.pt FastDetector.tflite

        # Export with int8 quantization
        detect-export tflite detector.pt FastDetector.tflite --int8

        # Export without validation
        detect-export tflite detector.pt FastDetector.tflite --no-validate
    """
    try:
        from picode.detection.export.tflite import (
            TFLiteExportConfig,
            convert_to_tflite,
            validate_tflite_model,
        )
    except ImportError as e:
        raise click.ClickException(
            "TFLite export requires additional dependencies.\n"
            "Install with: pip install ai-edge-torch\n"
            "Or: pip install onnx tf2onnx tensorflow"
        ) from e

    from picode.detection.fast_detector import FastDetectorModel

    click.echo(f"Loading checkpoint: {checkpoint}")

    # Load model
    model = FastDetectorModel(input_size=input_size, pretrained=False)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    # Configure export
    config = TFLiteExportConfig(
        input_size=input_size,
        quantize_to_fp16=fp16,
        quantize_to_int8=int8,
    )

    click.echo("Exporting to TFLite...")
    click.echo(f"  Input size: {input_size}x{input_size}")
    click.echo(f"  FP16: {fp16}, INT8: {int8}")

    # Export
    output_path = convert_to_tflite(model, output, config)
    click.echo(f"Exported to: {output_path}")

    # Get file size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    click.echo(f"  Model size: {size_mb:.2f} MB")

    # Validate
    if validate:
        click.echo("Validating exported model...")
        try:
            results = validate_tflite_model(output_path, input_size=input_size)
            click.echo(f"  Valid: {results['valid']}")
            click.echo(f"  Inference time: {results['inference_time_ms']:.2f}ms")
            click.echo(f"  Output shapes: {results['output_shapes']}")
        except Exception as e:
            click.echo(f"  Validation failed: {e}", err=True)


@main.command("info")
@click.argument("model_path", type=click.Path(exists=True))
def model_info(model_path: str) -> None:
    """Show information about an exported model.

    MODEL_PATH is the path to an exported model (.mlpackage or .tflite).

    Examples:

        detect-export info FastDetector.mlpackage
        detect-export info FastDetector.tflite
    """
    path = Path(model_path)

    if path.suffix == ".tflite":
        try:
            from picode.detection.export.tflite import get_tflite_model_info

            info = get_tflite_model_info(path)

            click.echo(f"TFLite Model: {info['model_path']}")
            click.echo(f"Size: {info['model_size_mb']:.2f} MB")
            click.echo("Inputs:")
            for inp in info["inputs"]:
                click.echo(f"  - {inp['name']}: {inp['shape']} ({inp['dtype']})")
            click.echo("Outputs:")
            for out in info["outputs"]:
                click.echo(f"  - {out['name']}: {out['shape']} ({out['dtype']})")
        except ImportError:
            click.echo("tensorflow is required to inspect TFLite models", err=True)

    elif path.suffix == ".mlpackage" or path.is_dir():
        try:
            import coremltools as ct

            model = ct.models.MLModel(str(path))
            spec = model.get_spec()

            click.echo(f"Core ML Model: {path}")
            click.echo(f"Author: {model.author}")
            click.echo(f"Description: {model.short_description}")
            click.echo(f"Version: {model.version}")
            click.echo("Inputs:")
            for inp in spec.description.input:
                click.echo(f"  - {inp.name}: {inp.shortDescription}")
            click.echo("Outputs:")
            for out in spec.description.output:
                click.echo(f"  - {out.name}: {out.shortDescription}")
        except ImportError:
            click.echo("coremltools is required to inspect Core ML models", err=True)
    else:
        click.echo(f"Unknown model format: {path.suffix}", err=True)


if __name__ == "__main__":
    main()
