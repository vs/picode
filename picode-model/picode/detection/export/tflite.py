# picode/detection/export/tflite.py
"""Export FastDetector to TFLite format for Android deployment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from picode.detection.fast_detector import FastDetectorModel


@dataclass
class TFLiteExportConfig:
    """Configuration for TFLite export.

    Attributes:
        input_size: Input image size (default 320)
        quantize_to_fp16: Whether to quantize to float16 (reduces size, maintains accuracy)
        quantize_to_int8: Whether to quantize to int8 (smallest size, may reduce accuracy)
        use_dynamic_range_quantization: Use dynamic range quantization for int8
        model_name: Name for the exported model
        add_metadata: Whether to add TFLite metadata
    """

    input_size: int = 320
    quantize_to_fp16: bool = True
    quantize_to_int8: bool = False
    use_dynamic_range_quantization: bool = True
    model_name: str = "FastDetector"
    add_metadata: bool = True


class _FastDetectorTFLiteWrapper(nn.Module):
    """Wrapper for TFLite-compatible output format."""

    def __init__(self, model: FastDetectorModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass returning tuple for TFLite compatibility."""
        out = self.model(x)
        return (
            out["is_watermark"],
            out["corners"],
            out["corner_confidence"],
        )


def convert_to_tflite(
    model: FastDetectorModel | nn.Module,
    output_path: str | Path,
    config: TFLiteExportConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Convert FastDetector model to TFLite format.

    This function supports two export methods:
    1. ai-edge-torch (preferred): Direct PyTorch to TFLite conversion
    2. ONNX + tf2onnx (fallback): PyTorch -> ONNX -> TFLite

    Args:
        model: FastDetectorModel instance or any nn.Module with compatible output
        output_path: Path for the output .tflite file
        config: Export configuration (uses defaults if not provided)
        checkpoint_path: Optional path to load model weights from

    Returns:
        Path to the exported .tflite file

    Raises:
        ImportError: If required export libraries are not installed
        ValueError: If model conversion fails

    Example:
        >>> model = FastDetectorModel(input_size=320)
        >>> model.load_state_dict(torch.load("detector.pt")["model_state_dict"])
        >>> path = convert_to_tflite(model, "FastDetector.tflite")
        >>> print(f"Exported to: {path}")
    """
    config = config or TFLiteExportConfig()
    output_path = Path(output_path)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

    # Wrap model for tuple output
    wrapped_model = _FastDetectorTFLiteWrapper(model)
    wrapped_model.eval()

    # Try ai-edge-torch first (preferred method)
    try:
        return _convert_with_ai_edge_torch(wrapped_model, output_path, config)
    except ImportError:
        pass

    # Fall back to ONNX -> TFLite conversion
    try:
        return _convert_via_onnx(wrapped_model, output_path, config)
    except ImportError as e:
        raise ImportError(
            "TFLite export requires either:\n"
            "  1. ai-edge-torch: pip install ai-edge-torch\n"
            "  2. onnx + tf2onnx + tensorflow: pip install onnx tf2onnx tensorflow"
        ) from e


def _convert_with_ai_edge_torch(
    model: nn.Module,
    output_path: Path,
    config: TFLiteExportConfig,
) -> Path:
    """Convert using ai-edge-torch (Google's official tool)."""
    import ai_edge_torch

    # Create example input
    example_input = torch.rand(1, 3, config.input_size, config.input_size)

    # Convert to TFLite
    edge_model = ai_edge_torch.convert(model, (example_input,))

    # Apply quantization if requested
    if config.quantize_to_int8:
        # Note: Full int8 quantization requires representative dataset
        # Using dynamic range quantization as default
        edge_model = ai_edge_torch.quantize(
            edge_model,
            mode=ai_edge_torch.quantization.QuantizationMode.DYNAMIC_RANGE,
        )
    elif config.quantize_to_fp16:
        edge_model = ai_edge_torch.quantize(
            edge_model,
            mode=ai_edge_torch.quantization.QuantizationMode.FLOAT16,
        )

    # Export to file
    edge_model.export(str(output_path))

    return output_path


def _convert_via_onnx(
    model: nn.Module,
    output_path: Path,
    config: TFLiteExportConfig,
) -> Path:
    """Convert via ONNX -> TFLite pipeline (fallback method)."""
    import tempfile

    import onnx

    # Create example input
    example_input = torch.rand(1, 3, config.input_size, config.input_size)

    # Export to ONNX first
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name

    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        input_names=["image"],
        output_names=["is_watermark", "corners", "corner_confidence"],
        dynamic_axes=None,  # Fixed batch size for mobile
        opset_version=13,
    )

    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Convert ONNX to TFLite
    _onnx_to_tflite(onnx_path, output_path, config)

    # Clean up ONNX file
    Path(onnx_path).unlink()

    return output_path


def _onnx_to_tflite(
    onnx_path: str,
    output_path: Path,
    config: TFLiteExportConfig,
) -> None:
    """Convert ONNX model to TFLite."""
    import onnx
    import tensorflow as tf
    from onnx_tf.backend import prepare

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)

    # Convert to TensorFlow
    tf_rep = prepare(onnx_model)

    # Save as SavedModel
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_path = Path(tmpdir) / "saved_model"
        tf_rep.export_graph(str(saved_model_path))

        # Convert SavedModel to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))

        # Apply optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if config.quantize_to_fp16:
            converter.target_spec.supported_types = [tf.float16]

        if config.quantize_to_int8:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]

        # Convert
        tflite_model = converter.convert()

        # Save TFLite model
        with open(output_path, "wb") as f:
            f.write(tflite_model)


def validate_tflite_model(
    tflite_path: str | Path,
    test_input: Tensor | None = None,
    input_size: int = 320,
) -> dict[str, Any]:
    """Validate an exported TFLite model.

    Args:
        tflite_path: Path to the .tflite file
        test_input: Optional test input tensor (creates random if not provided)
        input_size: Input size if creating random test input

    Returns:
        Dictionary with validation results including output shapes and inference time

    Example:
        >>> results = validate_tflite_model("FastDetector.tflite")
        >>> print(f"Inference time: {results['inference_time_ms']:.2f}ms")
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "tensorflow is required for TFLite validation. "
            "Install it with: pip install tensorflow"
        ) from e

    import time

    import numpy as np

    tflite_path = Path(tflite_path)

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create test input
    if test_input is None:
        test_input = torch.rand(1, 3, input_size, input_size)

    # Convert to numpy and set correct format
    input_data = test_input.numpy().astype(np.float32)

    # Run inference and measure time
    interpreter.set_tensor(input_details[0]["index"], input_data)

    start_time = time.perf_counter()
    interpreter.invoke()
    inference_time = (time.perf_counter() - start_time) * 1000  # ms

    # Get outputs
    outputs = {}
    for detail in output_details:
        outputs[detail["name"]] = interpreter.get_tensor(detail["index"])

    return {
        "valid": True,
        "inference_time_ms": inference_time,
        "input_shape": input_details[0]["shape"].tolist(),
        "output_shapes": {k: v.shape for k, v in outputs.items()},
        "model_size_mb": tflite_path.stat().st_size / (1024 * 1024),
        "model_path": str(tflite_path),
    }


def get_tflite_model_info(tflite_path: str | Path) -> dict[str, Any]:
    """Get information about a TFLite model without running inference.

    Args:
        tflite_path: Path to the .tflite file

    Returns:
        Dictionary with model information
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "tensorflow is required for model info. "
            "Install it with: pip install tensorflow"
        ) from e

    tflite_path = Path(tflite_path)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return {
        "model_path": str(tflite_path),
        "model_size_mb": tflite_path.stat().st_size / (1024 * 1024),
        "inputs": [
            {
                "name": d["name"],
                "shape": d["shape"].tolist(),
                "dtype": str(d["dtype"]),
            }
            for d in input_details
        ],
        "outputs": [
            {
                "name": d["name"],
                "shape": d["shape"].tolist(),
                "dtype": str(d["dtype"]),
            }
            for d in output_details
        ],
    }
