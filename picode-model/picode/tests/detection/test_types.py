# picode/tests/detection/test_types.py
"""Tests for detection types."""

import numpy as np
import pytest
import torch

from picode.detection.types import Point, Quadrilateral


class TestPoint:
    def test_point_creation(self) -> None:
        p = Point(10.5, 20.3)
        assert p.x == 10.5
        assert p.y == 20.3

    def test_point_to_tuple(self) -> None:
        p = Point(10.0, 20.0)
        assert p.to_tuple() == (10.0, 20.0)

    def test_point_scaled(self) -> None:
        p = Point(0.5, 0.25)
        scaled = p.scaled(100.0, 200.0)
        assert scaled.x == 50.0
        assert scaled.y == 50.0


def test_public_exports() -> None:
    """Test that all public classes are exported from detection module."""
    from picode.detection import (
        Detection,
        FastDetector,
        FastDetectorModel,
        Point,
        Quadrilateral,
        Rectifier,
    )

    assert Detection is not None
    assert FastDetector is not None
    assert FastDetectorModel is not None
    assert Point is not None
    assert Quadrilateral is not None
    assert Rectifier is not None


class TestQuadrilateral:
    @pytest.fixture
    def unit_quad(self) -> Quadrilateral:
        """Unit square quadrilateral."""
        return Quadrilateral(
            top_left=Point(0.0, 0.0),
            top_right=Point(1.0, 0.0),
            bottom_right=Point(1.0, 1.0),
            bottom_left=Point(0.0, 1.0),
        )

    def test_quad_creation(self, unit_quad: Quadrilateral) -> None:
        assert unit_quad.top_left.x == 0.0
        assert unit_quad.bottom_right.x == 1.0

    def test_quad_to_tensor(self, unit_quad: Quadrilateral) -> None:
        t = unit_quad.to_tensor()
        assert t.shape == (8,)
        assert torch.allclose(t, torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]))

    def test_quad_from_tensor(self) -> None:
        t = torch.tensor([0.1, 0.2, 0.9, 0.2, 0.9, 0.8, 0.1, 0.8])
        quad = Quadrilateral.from_tensor(t)
        assert quad.top_left.x == pytest.approx(0.1)
        assert quad.top_left.y == pytest.approx(0.2)
        assert quad.bottom_right.x == pytest.approx(0.9)

    def test_quad_to_numpy(self, unit_quad: Quadrilateral) -> None:
        arr = unit_quad.to_numpy()
        assert arr.shape == (4, 2)
        assert arr.dtype == np.float32
        expected = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        np.testing.assert_array_almost_equal(arr, expected)

    def test_quad_round_trip(self) -> None:
        original = torch.tensor([0.2, 0.3, 0.8, 0.25, 0.85, 0.75, 0.15, 0.8])
        quad = Quadrilateral.from_tensor(original)
        recovered = quad.to_tensor()
        assert torch.allclose(original, recovered)
