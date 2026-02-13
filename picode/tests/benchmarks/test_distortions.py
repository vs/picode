"""Benchmark tests for distortion backends.

Run with: pytest picode/tests/benchmarks/ --benchmark-only -v
"""

from typing import Any

import pytest
import torch
from torch import Tensor

from picode.distortions import kornia, native

DISTORTIONS: list[tuple[str, dict[str, Any]]] = [
    ("GaussianBlur", {"intensity": 0.5, "kernel_size": 7, "sigma": 2.0}),
    ("MotionBlur", {"intensity": 0.5, "kernel_size": 7}),
    ("GaussianNoise", {"intensity": 0.5, "std": 0.02}),
    ("Contrast", {"intensity": 0.5}),
    ("Saturation", {"intensity": 0.5}),
    ("Rotation", {"intensity": 0.5, "max_angle": 30.0}),
    ("Scale", {"intensity": 0.5}),
    ("Crop", {"intensity": 0.5}),
    ("JPEGCompression", {"intensity": 0.5, "quality": 50}),
]


def get_module(backend: str):
    """Get distortion module by backend name."""
    if backend == "native":
        return native
    return kornia


@pytest.mark.parametrize("distortion_name,kwargs", DISTORTIONS)
def test_distortion_benchmark(
    benchmark,
    backend: str,
    batch_size: int,
    benchmark_image: Tensor,
    distortion_name: str,
    kwargs: dict[str, Any],
) -> None:
    """Benchmark a single distortion across backends and batch sizes."""
    module = get_module(backend)
    device = benchmark_image.device
    distortion = getattr(module, distortion_name)(**kwargs).to(device)

    def run() -> Tensor:
        result = distortion(benchmark_image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return result

    # Set benchmark group for nice comparison tables
    benchmark.group = distortion_name
    benchmark.extra_info["backend"] = backend
    benchmark.extra_info["batch_size"] = batch_size

    benchmark(run)
