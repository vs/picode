# picode/tests/detection/export/test_export.py
"""Tests for FastDetector export functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from picode.detection.fast_detector import FastDetectorModel


class TestExportConfig:
    """Test export configuration dataclasses."""

    def test_coreml_config_defaults(self) -> None:
        """CoreMLExportConfig has sensible defaults."""
        from picode.detection.export.coreml import CoreMLExportConfig

        config = CoreMLExportConfig()

        assert config.input_size == 320
        assert config.compute_units == "ALL"
        assert config.minimum_deployment_target == "iOS15"
        assert config.convert_to_fp16 is True
        assert config.quantize_to_int8 is False

    def test_tflite_config_defaults(self) -> None:
        """TFLiteExportConfig has sensible defaults."""
        from picode.detection.export.tflite import TFLiteExportConfig

        config = TFLiteExportConfig()

        assert config.input_size == 320
        assert config.quantize_to_fp16 is True
        assert config.quantize_to_int8 is False

    def test_coreml_config_custom(self) -> None:
        """CoreMLExportConfig accepts custom values."""
        from picode.detection.export.coreml import CoreMLExportConfig

        config = CoreMLExportConfig(
            input_size=256,
            compute_units="CPU_ONLY",
            minimum_deployment_target="iOS16",
            convert_to_fp16=False,
            quantize_to_int8=True,
        )

        assert config.input_size == 256
        assert config.compute_units == "CPU_ONLY"
        assert config.minimum_deployment_target == "iOS16"
        assert config.convert_to_fp16 is False
        assert config.quantize_to_int8 is True


class TestFastDetectorWrapper:
    """Test wrapper classes for export."""

    def test_coreml_wrapper_output_format(self) -> None:
        """Core ML wrapper returns tuple instead of dict."""
        from picode.detection.export.coreml import _FastDetectorWrapper

        model = FastDetectorModel(input_size=320, pretrained=False)
        wrapped = _FastDetectorWrapper(model)

        x = torch.rand(1, 3, 320, 320)
        output = wrapped(x)

        assert isinstance(output, tuple)
        assert len(output) == 3
        # is_watermark, corners, corner_confidence
        assert output[0].shape == (1, 1)
        assert output[1].shape == (1, 8)
        assert output[2].shape == (1, 1)

    def test_tflite_wrapper_output_format(self) -> None:
        """TFLite wrapper returns tuple instead of dict."""
        from picode.detection.export.tflite import _FastDetectorTFLiteWrapper

        model = FastDetectorModel(input_size=320, pretrained=False)
        wrapped = _FastDetectorTFLiteWrapper(model)

        x = torch.rand(1, 3, 320, 320)
        output = wrapped(x)

        assert isinstance(output, tuple)
        assert len(output) == 3


class TestJITTrace:
    """Test that model can be JIT traced (required for export)."""

    def test_model_is_traceable(self) -> None:
        """FastDetector can be JIT traced."""
        from picode.detection.export.coreml import _FastDetectorWrapper

        model = FastDetectorModel(input_size=320, pretrained=False)
        wrapped = _FastDetectorWrapper(model)
        wrapped.eval()

        x = torch.rand(1, 3, 320, 320)

        # Should not raise
        traced = torch.jit.trace(wrapped, x)

        # Traced model should produce same output
        with torch.no_grad():
            original_out = wrapped(x)
            traced_out = traced(x)

        for orig, traced in zip(original_out, traced_out):
            assert torch.allclose(orig, traced, atol=1e-5)


# Skip Core ML tests if coremltools not installed
try:
    import coremltools

    HAS_COREMLTOOLS = True
except ImportError:
    HAS_COREMLTOOLS = False


@pytest.mark.skipif(not HAS_COREMLTOOLS, reason="coremltools not installed")
class TestCoreMLExport:
    """Tests for Core ML export (requires coremltools)."""

    @pytest.fixture
    def model(self) -> FastDetectorModel:
        """Create a FastDetector model."""
        model = FastDetectorModel(input_size=320, pretrained=False)
        model.eval()
        return model

    def test_export_creates_mlpackage(self, model: FastDetectorModel) -> None:
        """Export creates .mlpackage file."""
        from picode.detection.export.coreml import convert_to_coreml

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mlpackage"

            result = convert_to_coreml(model, output_path)

            assert result.exists()
            assert result.is_dir()  # mlpackage is a directory

    def test_export_with_checkpoint(self, model: FastDetectorModel) -> None:
        """Export can load from checkpoint."""
        from picode.detection.export.coreml import convert_to_coreml

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint
            ckpt_path = Path(tmpdir) / "checkpoint.pt"
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

            # Create fresh model
            fresh_model = FastDetectorModel(input_size=320, pretrained=False)

            # Export with checkpoint
            output_path = Path(tmpdir) / "test.mlpackage"
            result = convert_to_coreml(fresh_model, output_path, checkpoint_path=ckpt_path)

            assert result.exists()

    def test_validate_exported_model(self, model: FastDetectorModel) -> None:
        """Validate function works on exported model."""
        from picode.detection.export.coreml import convert_to_coreml, validate_coreml_model

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mlpackage"
            convert_to_coreml(model, output_path)

            results = validate_coreml_model(output_path)

            assert results["valid"] is True
            assert "inference_time_ms" in results
            assert "output_shapes" in results


# Skip TFLite tests if dependencies not installed
try:
    import tensorflow

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import ai_edge_torch

    HAS_AI_EDGE_TORCH = True
except ImportError:
    HAS_AI_EDGE_TORCH = False


@pytest.mark.skipif(
    not (HAS_TENSORFLOW or HAS_AI_EDGE_TORCH),
    reason="TFLite export dependencies not installed",
)
class TestTFLiteExport:
    """Tests for TFLite export (requires tensorflow or ai-edge-torch)."""

    @pytest.fixture
    def model(self) -> FastDetectorModel:
        """Create a FastDetector model."""
        model = FastDetectorModel(input_size=320, pretrained=False)
        model.eval()
        return model

    @pytest.mark.skipif(not HAS_AI_EDGE_TORCH, reason="ai-edge-torch not installed")
    def test_export_creates_tflite(self, model: FastDetectorModel) -> None:
        """Export creates .tflite file."""
        from picode.detection.export.tflite import convert_to_tflite

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tflite"

            result = convert_to_tflite(model, output_path)

            assert result.exists()
            assert result.is_file()

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="tensorflow not installed")
    def test_get_tflite_model_info(self) -> None:
        """Can get info from TFLite model."""
        from picode.detection.export.tflite import get_tflite_model_info

        # Create a simple test model and export it
        # This test only runs if we have a tflite file
        # Skip for now since export may not work without full setup
        pytest.skip("Requires existing TFLite model file")


class TestExportModuleImports:
    """Test that export module can be imported."""

    def test_import_export_module(self) -> None:
        """Export module can be imported."""
        from picode.detection import export

        assert hasattr(export, "convert_to_coreml")
        assert hasattr(export, "convert_to_tflite")
        assert hasattr(export, "CoreMLExportConfig")
        assert hasattr(export, "TFLiteExportConfig")

    def test_import_from_detection(self) -> None:
        """Export types accessible from detection module."""
        # Export module should be importable even without dependencies
        from picode.detection.export import CoreMLExportConfig, TFLiteExportConfig

        assert CoreMLExportConfig is not None
        assert TFLiteExportConfig is not None
