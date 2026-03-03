"""Tests for window generator."""

from picode.detection.window import Window, WindowGenerator


class TestWindow:
    """Tests for Window dataclass."""

    def test_window_attributes(self) -> None:
        """Window has required attributes."""
        w = Window(x=10, y=20, w=100, h=80, scale=0.5)
        assert w.x == 10
        assert w.y == 20
        assert w.w == 100
        assert w.h == 80
        assert w.scale == 0.5


class TestWindowGenerator:
    """Tests for WindowGenerator."""

    def test_generates_windows(self) -> None:
        """Generator produces windows."""
        gen = WindowGenerator(scales=[0.5], stride_ratio=0.25)
        windows = list(gen.generate(frame_h=100, frame_w=100))
        assert len(windows) > 0
        assert all(isinstance(w, Window) for w in windows)

    def test_windows_within_bounds(self) -> None:
        """All windows stay within frame boundaries."""
        gen = WindowGenerator(scales=[0.25, 0.5, 0.75], stride_ratio=0.15)
        windows = list(gen.generate(frame_h=1080, frame_w=1920))
        for w in windows:
            assert w.x >= 0
            assert w.y >= 0
            assert w.x + w.w <= 1920
            assert w.y + w.h <= 1080

    def test_window_sizes_match_scales(self) -> None:
        """Windows at each scale have correct size."""
        gen = WindowGenerator(scales=[0.5], stride_ratio=0.2)
        windows = list(gen.generate(frame_h=100, frame_w=200))
        # At 0.5 scale, window should be 100x50 (half of frame)
        for w in windows:
            assert w.w == 100
            assert w.h == 50
            assert w.scale == 0.5

    def test_multiple_scales_different_sizes(self) -> None:
        """Different scales produce different window sizes."""
        gen = WindowGenerator(scales=[0.25, 0.5], stride_ratio=0.5)
        windows = list(gen.generate(frame_h=100, frame_w=100))
        sizes = set((w.w, w.h) for w in windows)
        assert len(sizes) == 2  # Two different sizes

    def test_stride_affects_window_count(self) -> None:
        """Smaller stride produces more windows."""
        gen_sparse = WindowGenerator(scales=[0.5], stride_ratio=0.5)
        gen_dense = WindowGenerator(scales=[0.5], stride_ratio=0.1)
        sparse = list(gen_sparse.generate(frame_h=100, frame_w=100))
        dense = list(gen_dense.generate(frame_h=100, frame_w=100))
        assert len(dense) > len(sparse)
