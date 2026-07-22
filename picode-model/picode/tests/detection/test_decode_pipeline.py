"""Tests for multi-model decode pipeline."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from picode.detection.decode_pipeline import (
    DecodePipeline,
    DecodeResult,
    ModelSpec,
)


@pytest.fixture
def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(name="b72", num_bits=72, checkpoint_path="b72.pt"),
        ModelSpec(name="b96", num_bits=96, checkpoint_path="b96.pt"),
        ModelSpec(name="b48", num_bits=48, checkpoint_path="b48.pt"),
    ]


class TestDecodeResult:
    def test_success_result(self) -> None:
        result = DecodeResult(
            success=True,
            model_name="b72",
            message=torch.ones(38),
            raw_bits=torch.ones(72),
        )
        assert result.success
        assert result.model_name == "b72"

    def test_failure_result(self) -> None:
        result = DecodeResult(success=False)
        assert not result.success
        assert result.model_name is None
        assert result.message is None


class TestModelSpec:
    def test_creation(self) -> None:
        spec = ModelSpec(name="b72", num_bits=72, checkpoint_path="b72.pt")
        assert spec.name == "b72"
        assert spec.num_bits == 72


class TestDecodePipelineStrategyC:
    def test_first_decoder_succeeds(self, model_specs: list[ModelSpec]) -> None:
        pipeline = DecodePipeline(model_specs=model_specs, strategy="C")
        fake_result = DecodeResult(
            success=True,
            model_name="b72",
            message=torch.ones(38),
            raw_bits=torch.ones(72),
        )
        with patch.object(pipeline, "_try_decode", return_value=fake_result):
            result = pipeline._try_all_decoders(torch.rand(3, 512, 512))
            assert result.success
            assert result.model_name == "b72"

    def test_no_decoder_succeeds(self, model_specs: list[ModelSpec]) -> None:
        pipeline = DecodePipeline(model_specs=model_specs, strategy="C")
        fail = DecodeResult(success=False)
        with patch.object(pipeline, "_try_decode", return_value=fail):
            result = pipeline._try_all_decoders(torch.rand(3, 512, 512))
            assert not result.success
