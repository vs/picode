# picode/detection/export/coreml.py
"""Export FastDetector to Core ML format for iOS deployment."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from picode.detection.fast_detector import FastDetectorModel


@dataclass
class CoreMLExportConfig:
    """Configuration for Core ML export.

    Attributes:
        input_size: Input image size (default 320)
        compute_units: Core ML compute units ('ALL', 'CPU_AND_GPU', 'CPU_AND_NE', 'CPU_ONLY')
        minimum_deployment_target: Minimum iOS version (e.g., 'iOS15', 'iOS16', 'iOS17')
        convert_to_fp16: Whether to convert weights to float16 (reduces size)
        quantize_to_int8: Whether to quantize to int8 (further reduces size)
        model_name: Name for the exported model
        model_description: Description for the model metadata
        author: Author name for metadata
        version: Version string for metadata
    """

    input_size: int = 320
    compute_units: str = "ALL"
    minimum_deployment_target: str = "iOS15"
    convert_to_fp16: bool = True
    quantize_to_int8: bool = False
    model_name: str = "FastDetector"
    model_description: str = "Watermark detection model for real-time mobile inference"
    author: str = "Picode"
    version: str = "1.0"
    input_names: list[str] = field(default_factory=lambda: ["image"])
    output_names: list[str] = field(
        default_factory=lambda: ["is_watermark", "corners", "corner_confidence"]
    )


class _FastDetectorWrapper(nn.Module):
    """Wrapper to convert dict output to tuple for Core ML compatibility."""

    def __init__(self, model: FastDetectorModel | nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass returning tuple instead of dict."""
        out = self.model(x)
        return (
            out["is_watermark"],
            out["corners"],
            out["corner_confidence"],
        )


def convert_to_coreml(
    model: FastDetectorModel | nn.Module,
    output_path: str | Path,
    config: CoreMLExportConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Convert FastDetector model to Core ML format.

    Args:
        model: FastDetectorModel instance or any nn.Module with compatible output
        output_path: Path for the output .mlpackage file
        config: Export configuration (uses defaults if not provided)
        checkpoint_path: Optional path to load model weights from

    Returns:
        Path to the exported .mlpackage file

    Raises:
        ImportError: If coremltools is not installed
        ValueError: If model conversion fails

    Example:
        >>> model = FastDetectorModel(input_size=320)
        >>> model.load_state_dict(torch.load("detector.pt")["model_state_dict"])
        >>> path = convert_to_coreml(model, "FastDetector.mlpackage")
        >>> print(f"Exported to: {path}")
    """
    ct = import_coremltools()

    config = config or CoreMLExportConfig()
    output_path = Path(output_path)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

    # Wrap model for tuple output
    wrapped_model = _FastDetectorWrapper(model)
    wrapped_model.eval()

    # Create example input
    example_input = torch.rand(1, 3, config.input_size, config.input_size)

    # Trace the model
    traced_model = torch.jit.trace(wrapped_model, example_input)  # type: ignore[no-untyped-call]

    # Convert to Core ML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name=config.input_names[0],
                shape=(1, 3, config.input_size, config.input_size),
                scale=1.0 / 255.0,  # Normalize from [0, 255] to [0, 1]
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[
            ct.TensorType(name=config.output_names[0]),  # is_watermark
            ct.TensorType(name=config.output_names[1]),  # corners
            ct.TensorType(name=config.output_names[2]),  # corner_confidence
        ],
        compute_units=_get_compute_units(config.compute_units),
        minimum_deployment_target=_get_deployment_target(config.minimum_deployment_target),
        convert_to="mlprogram",
    )

    # Set metadata
    mlmodel.author = config.author
    mlmodel.short_description = config.model_description
    mlmodel.version = config.version

    # Add input/output descriptions
    spec = mlmodel.get_spec()
    _set_io_descriptions(spec, config)

    # Apply optimizations
    if config.convert_to_fp16:
        mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
            mlmodel, nbits=16
        )

    if config.quantize_to_int8:
        # Note: INT8 quantization requires calibration data for best results
        # This uses post-training quantization without calibration
        mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
            mlmodel, nbits=8
        )

    # Save the model
    mlmodel.save(str(output_path))

    return output_path



def import_coremltools() -> Any:
    """Import coremltools, explaining the Python-version limit when it is unusable.

    coremltools publishes native wheels only up to CPython 3.13; on newer interpreters
    pip falls back to a build without its native library, which imports but cannot export.

    Raises:
        ImportError: If coremltools is missing or unusable on this interpreter.
    """
    if sys.version_info >= (3, 14):
        raise ImportError(
            f"Core ML export needs Python 3.10-3.13 (running {sys.version.split()[0]}); "
            "coremltools has no native build for newer versions. Create a separate "
            "environment, e.g. `uv venv -p 3.13 && pip install -e '.[export-ios]'`."
        )
    try:
        import coremltools as ct
    except Exception as e:  # coremltools can fail with non-ImportErrors (native libs)
        raise ImportError(
            "coremltools is required for Core ML export: pip install -e '.[export-ios]'"
        ) from e
    return ct

def _get_compute_units(units_str: str) -> Any:
    """Convert string to coremltools compute units enum."""
    import coremltools as ct

    mapping = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }
    return mapping.get(units_str.upper(), ct.ComputeUnit.ALL)


def _get_deployment_target(target_str: str) -> Any:
    """Convert string to coremltools deployment target enum."""
    import coremltools as ct

    mapping = {
        "iOS13": ct.target.iOS13,
        "iOS14": ct.target.iOS14,
        "iOS15": ct.target.iOS15,
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
    }
    return mapping.get(target_str, ct.target.iOS15)


def _set_io_descriptions(spec: Any, config: CoreMLExportConfig) -> None:
    """Set input/output descriptions in the model spec."""
    # Input description
    for inp in spec.description.input:
        if inp.name == config.input_names[0]:
            inp.shortDescription = "RGB image normalized to [0, 1]"

    # Output descriptions
    output_descriptions = {
        config.output_names[0]: "Watermark presence logit (apply sigmoid for probability)",
        config.output_names[1]: "Normalized corner coordinates [x1,y1,x2,y2,x3,y3,x4,y4]",
        config.output_names[2]: "Corner prediction confidence [0, 1]",
    }

    for out in spec.description.output:
        if out.name in output_descriptions:
            out.shortDescription = output_descriptions[out.name]


def validate_coreml_model(
    mlpackage_path: str | Path,
    test_input: Tensor | None = None,
    input_size: int = 320,
) -> dict[str, Any]:
    """Validate an exported Core ML model.

    Args:
        mlpackage_path: Path to the .mlpackage file
        test_input: Optional test input tensor (creates random if not provided)
        input_size: Input size if creating random test input

    Returns:
        Dictionary with validation results including output shapes and inference time

    Example:
        >>> results = validate_coreml_model("FastDetector.mlpackage")
        >>> print(f"Inference time: {results['inference_time_ms']:.2f}ms")
    """
    try:
        import coremltools as ct
    except ImportError as e:
        raise ImportError(
            "coremltools is required for validation. "
            "Install it with: pip install coremltools"
        ) from e

    import time

    import numpy as np
    from PIL import Image

    mlpackage_path = Path(mlpackage_path)
    model = ct.models.MLModel(str(mlpackage_path))

    # Create test input
    if test_input is None:
        test_input = torch.rand(1, 3, input_size, input_size)

    # Convert to PIL Image (Core ML expects this for ImageType input)
    img_np = (test_input.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_image = Image.fromarray(img_np)

    # Run inference and measure time
    start_time = time.perf_counter()
    outputs = model.predict({"image": pil_image})
    inference_time = (time.perf_counter() - start_time) * 1000  # ms

    return {
        "valid": True,
        "inference_time_ms": inference_time,
        "output_keys": list(outputs.keys()),
        "output_shapes": {k: np.array(v).shape for k, v in outputs.items()},
        "model_path": str(mlpackage_path),
    }
