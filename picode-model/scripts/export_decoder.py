#!/usr/bin/env python
"""Export trained StegaStamp decoder to CoreML for iOS.

Usage:
    python scripts/export_decoder.py checkpoints/best.pt \\
        exports/StegaStampDecoder.mlpackage

    python scripts/export_decoder.py checkpoints/best.pt \\
        exports/StegaStampDecoder.mlpackage --fp16 --validate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from picode.models.stegastamp import Decoder


class _DecoderWrapper(nn.Module):
    """Wrapper to ensure consistent output for CoreML tracing."""

    def __init__(self, decoder: Decoder) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export StegaStamp decoder to CoreML")
    parser.add_argument("checkpoint", type=str, help="Path to training checkpoint")
    parser.add_argument("output", type=str, help="Output path for .mlpackage")
    parser.add_argument("--fp16", action="store_true", help="Convert weights to FP16")
    parser.add_argument(
        "--ios-version",
        type=str,
        default="iOS16",
        choices=["iOS15", "iOS16", "iOS17"],
        help="Minimum iOS deployment target",
    )
    parser.add_argument("--validate", action="store_true", help="Run validation after export")
    return parser.parse_args()


def main() -> None:
    try:
        import coremltools as ct
    except ImportError as e:
        raise ImportError("Install coremltools: pip install coremltools") from e

    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get config from checkpoint
    config = ckpt.get("config", {})
    num_bits = config.get("training", {}).get("num_bits", 100)
    input_size = config.get("training", {}).get("image_size", 400)

    # Create and load decoder
    decoder = Decoder(num_bits=num_bits, height=input_size, width=input_size)
    decoder.load_state_dict(ckpt["decoder_state"])
    decoder.eval()

    print(f"Decoder loaded (num_bits={num_bits}, input_size={input_size})")

    # Wrap for tracing
    wrapped = _DecoderWrapper(decoder)
    wrapped.eval()

    # Trace
    example_input = torch.rand(1, 3, input_size, input_size)
    traced = torch.jit.trace(wrapped, example_input)

    # iOS version mapping
    ios_targets = {
        "iOS15": ct.target.iOS15,
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
    }

    # Convert to CoreML
    print(f"Exporting to: {output_path}")
    print(f"  iOS target: {args.ios_version}")
    print(f"  FP16: {args.fp16}")

    # For mlprogram models, FP16 is set via compute_precision
    compute_precision = ct.precision.FLOAT16 if args.fp16 else ct.precision.FLOAT32

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, input_size, input_size),
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name="logits")],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ios_targets.get(args.ios_version, ct.target.iOS16),
        convert_to="mlprogram",
        compute_precision=compute_precision,
    )

    # Metadata
    mlmodel.author = "Picode"
    mlmodel.short_description = "StegaStamp decoder for iOS"
    mlmodel.version = "1.0"

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    print(f"Saved: {output_path}")

    # Validate
    if args.validate:
        print("\nValidating...")
        import time

        import numpy as np
        from PIL import Image

        model = ct.models.MLModel(str(output_path))
        test_img = Image.fromarray(
            (np.random.rand(input_size, input_size, 3) * 255).astype(np.uint8)
        )

        start = time.perf_counter()
        result = model.predict({"image": test_img})
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Inference: {elapsed:.1f}ms")
        print(f"  Output shape: {np.array(result['logits']).shape}")


if __name__ == "__main__":
    main()
