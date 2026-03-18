"""Structured logging configuration for picode-scraper."""

import logging
import sys

import structlog


def configure_logging(json_format: bool = False, level: str = "INFO") -> None:
    """Configure structured logging.

    Args:
        json_format: If True, output JSON logs (for containers). If False, pretty console.
        level: Log level (DEBUG, INFO, WARNING, ERROR).
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
        force=True,  # Allow reconfiguration
    )

    # Configure structlog processors
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_format:
        # JSON output for containers
        structlog.configure(
            processors=shared_processors + [structlog.processors.JSONRenderer()],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper())
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Pretty console output for development
        structlog.configure(
            processors=shared_processors + [structlog.dev.ConsoleRenderer()],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper())
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Optional logger name for context.

    Returns:
        Configured structlog logger.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger()
    if name:
        logger = logger.bind(logger=name)
    return logger
