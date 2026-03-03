"""Window generation for sliding window detection."""

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class Window:
    """A candidate detection window.

    Attributes:
        x: Top-left x coordinate.
        y: Top-left y coordinate.
        w: Window width.
        h: Window height.
        scale: Scale factor relative to frame size.
    """

    x: int
    y: int
    w: int
    h: int
    scale: float


class WindowGenerator:
    """Generates sliding windows at multiple scales.

    Args:
        scales: List of scale factors (0-1) relative to frame size.
        stride_ratio: Stride as fraction of window size (e.g., 0.15 = 15% overlap).
    """

    def __init__(
        self,
        scales: list[float] | None = None,
        stride_ratio: float = 0.15,
    ) -> None:
        self.scales = scales or [0.25, 0.35, 0.5, 0.65, 0.75]
        self.stride_ratio = stride_ratio

    def generate(self, frame_h: int, frame_w: int) -> Iterator[Window]:
        """Generate windows for a frame.

        Args:
            frame_h: Frame height in pixels.
            frame_w: Frame width in pixels.

        Yields:
            Window objects covering the frame at each scale.
        """
        for scale in self.scales:
            # Window size at this scale
            win_w = int(frame_w * scale)
            win_h = int(frame_h * scale)

            if win_w < 1 or win_h < 1:
                continue

            # Stride in pixels
            stride_x = max(1, int(win_w * self.stride_ratio))
            stride_y = max(1, int(win_h * self.stride_ratio))

            # Slide window across frame
            y = 0
            while y + win_h <= frame_h:
                x = 0
                while x + win_w <= frame_w:
                    yield Window(x=x, y=y, w=win_w, h=win_h, scale=scale)
                    x += stride_x
                y += stride_y
