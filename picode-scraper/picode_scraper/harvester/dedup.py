"""Image deduplication utilities."""

import hashlib

import cv2
import numpy as np
from sqlalchemy.orm import Session

from picode_scraper.db.models import Image
from picode_scraper.harvester.utils import compute_phash
from picode_scraper.storage.base import StorageBackend, detect_format


def get_or_create_image(
    db: Session,
    image_data: bytes,
    source_url: str,
    storage: StorageBackend,
) -> Image:
    """Store image with deduplication based on content hash.

    Args:
        db: Database session
        image_data: Raw image bytes
        source_url: Original download URL
        storage: Storage backend for saving image

    Returns:
        Image record (existing or newly created)

    Raises:
        ValueError: If image cannot be decoded
    """
    sha256 = hashlib.sha256(image_data).digest()

    # Check for exact duplicate
    existing = db.query(Image).filter(Image.sha256 == sha256).first()
    if existing:
        return existing

    # Decode image for dimensions and phash
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image from {source_url}")

    phash = compute_phash(img)

    # Store image
    format = detect_format(image_data)
    uri = storage.save(image_data, format)

    # Create record
    image = Image(
        source_url=source_url,
        phash=phash,
        sha256=sha256,
        storage_uri=uri,
        format=format,
        width=img.shape[1],
        height=img.shape[0],
    )
    db.add(image)

    return image
