"""Database connection management."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from picode_scraper.config import DatabaseConfig

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def init_db(config: DatabaseConfig) -> Engine:
    """Initialize database connection.

    Args:
        config: Database configuration containing URL and pool settings.

    Returns:
        The initialized SQLAlchemy engine.
    """
    global _engine, _session_factory

    _engine = create_engine(
        str(config.url),
        pool_size=config.pool_size,
        pool_pre_ping=True,
    )
    _session_factory = sessionmaker(bind=_engine)
    return _engine


def get_engine() -> Engine:
    """Get the database engine.

    Returns:
        The SQLAlchemy engine.

    Raises:
        RuntimeError: If database not initialized.
    """
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get a database session as context manager.

    Yields:
        A database session that commits on success, rolls back on error.

    Raises:
        RuntimeError: If database not initialized.
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
