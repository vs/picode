#!/usr/bin/env python
"""Export trained FastDetector to CoreML for iOS.

Usage:
    python scripts/export_detector.py checkpoints/detection/best.pt \\
        exports/FastDetector.mlpackage

    python scripts/export_detector.py checkpoints/detection/best.pt \\
        exports/FastDetector.mlpackage --fp16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from picode.detection.export.coreml import (
    CoreMLExportConfig,
    convert_to_coreml,
    validate_coreml_model,
)
from picode.detection.fast_detector import FastDetectorModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FastDetector to CoreML")
    parser.add_argument("checkpoint", type=str, help="Path to detector checkpoint")
    parser.add_argument("output", type=str, help="Output path for .mlpackage")
    parser.add_argument(
        "--fp16", action="store_true", help="Convert weights to FP16"
    )
    parser.add_argument(
        "--int8", action="store_true", help="Quantize to INT8 (aggressive)"
    )
    parser.add_argument(
        "--ios-version",
        type=str,
        default="iOS16",
        choices=["iOS15", "iOS16", "iOS17"],
        help="Minimum iOS deployment target",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Run validation after export"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get model config
    model_config = ckpt.get("model_config", {"input_size": 320})
    input_size = model_config.get("input_size", 320)

    # Create model
    model = FastDetectorModel(input_size=input_size, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Model loaded (input_size={input_size})")

    # Export config
    config = CoreMLExportConfig(
        input_size=input_size,
        minimum_deployment_target=args.ios_version,
        convert_to_fp16=args.fp16,
        quantize_to_int8=args.int8,
    )

    print(f"Exporting to: {output_path}")
    print(f"  iOS target: {args.ios_version}")
    print(f"  FP16: {args.fp16}")
    print(f"  INT8: {args.int8}")

    # Export
    result_path = convert_to_coreml(
        model=model,
        output_path=output_path,
        config=config,
    )

    print(f"\nExport complete: {result_path}")

    # Validate
    if args.validate:
        print("\nValidating exported model...")
        try:
            results = validate_coreml_model(result_path, input_size=input_size)
            print(f"  Inference time: {results['inference_time_ms']:.2f}ms")
            print(f"  Output keys: {results['output_keys']}")
            print(f"  Output shapes: {results['output_shapes']}")
        except Exception as e:
            print(f"  Validation failed: {e}")


if __name__ == "__main__":
    main()
