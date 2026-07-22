"""Tests for lightweight model classifier (Strategy B)."""

from __future__ import annotations

import torch

from picode.detection.model_classifier import ModelClassifierModel


class TestModelClassifierModel:
    def test_forward_shape(self) -> None:
        model = ModelClassifierModel(
            num_classes=4,
            class_names=["b30", "b48", "b72", "b96"],
            input_size=320,
        )
        x = torch.rand(2, 3, 320, 320)
        output = model(x)
        assert output.shape == (2, 4)

    def test_predict_returns_class_name(self) -> None:
        model = ModelClassifierModel(
            num_classes=4,
            class_names=["b30", "b48", "b72", "b96"],
            input_size=320,
        )
        model.eval()
        x = torch.rand(1, 3, 320, 320)
        name = model.predict(x)
        assert name in ["b30", "b48", "b72", "b96"]

    def test_predict_batch(self) -> None:
        model = ModelClassifierModel(
            num_classes=4,
            class_names=["b30", "b48", "b72", "b96"],
            input_size=320,
        )
        model.eval()
        x = torch.rand(4, 3, 320, 320)
        names = model.predict_batch(x)
        assert len(names) == 4
        assert all(n in ["b30", "b48", "b72", "b96"] for n in names)
