# picode/detection/types.py
"""Type definitions for detection module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor


@dataclass
class Point:
    """2D point in pixel or normalized coordinates."""

    x: float
    y: float

    def to_tuple(self) -> tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)

    def scaled(self, scale_x: float, scale_y: float) -> Point:
        """Scale coordinates by given factors."""
        return Point(self.x * scale_x, self.y * scale_y)


@dataclass
class Quadrilateral:
    """Four corners in clockwise order: TL, TR, BR, BL."""

    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point

    def to_tensor(self) -> Tensor:
        """Convert to (8,) tensor of [x1,y1, x2,y2, x3,y3, x4,y4]."""
        return torch.tensor(
            [
                self.top_left.x,
                self.top_left.y,
                self.top_right.x,
                self.top_right.y,
                self.bottom_right.x,
                self.bottom_right.y,
                self.bottom_left.x,
                self.bottom_left.y,
            ]
        )

    @classmethod
    def from_tensor(cls, t: Tensor) -> Quadrilateral:
        """Create from (8,) tensor."""
        return cls(
            top_left=Point(t[0].item(), t[1].item()),
            top_right=Point(t[2].item(), t[3].item()),
            bottom_right=Point(t[4].item(), t[5].item()),
            bottom_left=Point(t[6].item(), t[7].item()),
        )

    def to_numpy(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Convert to (4, 2) array for OpenCV."""
        return np.array(
            [
                [self.top_left.x, self.top_left.y],
                [self.top_right.x, self.top_right.y],
                [self.bottom_right.x, self.bottom_right.y],
                [self.bottom_left.x, self.bottom_left.y],
            ],
            dtype=np.float32,
        )


@dataclass
class Detection:
    """Detection result from FastDetector or SlowDetector.

    Attributes:
        corners: Quadrilateral corners in pixel coordinates
        confidence: Detection confidence [0, 1]
        detector_type: Which detector produced this result
        message_bits: Decoded bits (SlowDetector only)
        message_probs: Bit probabilities (SlowDetector only)
    """

    corners: Quadrilateral
    confidence: float
    detector_type: Literal["slow", "fast"]
    message_bits: Tensor | None = None
    message_probs: Tensor | None = None

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Bounding box (x, y, width, height) for backward compatibility."""
        xs = [
            self.corners.top_left.x,
            self.corners.top_right.x,
            self.corners.bottom_right.x,
            self.corners.bottom_left.x,
        ]
        ys = [
            self.corners.top_left.y,
            self.corners.top_right.y,
            self.corners.bottom_right.y,
            self.corners.bottom_left.y,
        ]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)
        return (x, y, w, h)
