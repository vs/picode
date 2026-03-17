"""Graceful shutdown handling for harvest workers."""

import signal
from collections.abc import Callable
from types import FrameType
from typing import Any

# Type alias for signal handlers returned by signal.signal()
_SignalHandler = Callable[[int, FrameType | None], Any] | int | None


class ShutdownHandler:
    """Handle SIGTERM/SIGINT for graceful shutdown.

    Usage:
        handler = ShutdownHandler()
        handler.register()

        while not handler.should_shutdown:
            # do work
            pass

        # cleanup
    """

    def __init__(self) -> None:
        """Initialize shutdown handler."""
        self.should_shutdown = False
        self._original_sigterm: _SignalHandler = signal.SIG_DFL
        self._original_sigint: _SignalHandler = signal.SIG_DFL

    def register(self) -> None:
        """Register signal handlers for SIGTERM and SIGINT."""
        self._original_sigterm = signal.signal(signal.SIGTERM, self.handle_signal)
        self._original_sigint = signal.signal(signal.SIGINT, self.handle_signal)

    def unregister(self) -> None:
        """Restore original signal handlers."""
        signal.signal(signal.SIGTERM, self._original_sigterm)
        signal.signal(signal.SIGINT, self._original_sigint)

    def handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """Handle shutdown signal by setting flag.

        Args:
            signum: Signal number received
            frame: Current stack frame (unused)
        """
        self.should_shutdown = True
