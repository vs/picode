"""Harvest worker for collecting image pairs."""

import time
import uuid

import cv2
import httpx
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from picode_scraper.config import Config
from picode_scraper.db.connection import get_session
from picode_scraper.db.models import HarvestTask, Pair, Source
from picode_scraper.harvester.dedup import get_or_create_image
from picode_scraper.harvester.pair_finder import PairFinder
from picode_scraper.harvester.rate_limiter import DomainRateLimiter
from picode_scraper.harvester.validator import ValidationResult
from picode_scraper.sources import get_source
from picode_scraper.sources.base import CandidateImage, PageContent
from picode_scraper.storage import create_storage_backend


class HarvestWorker:
    """Worker that harvests image pairs from sources.

    The worker runs a loop that:
    1. Claims pending tasks from the database using FOR UPDATE SKIP LOCKED
    2. Fetches the page content using the appropriate source plugin
    3. Downloads all candidate images from the page
    4. Uses PairFinder to validate all image combinations via SIFT matching
    5. Stores valid pairs with deduplication
    """

    def __init__(
        self,
        config: Config,
        worker_id: str | None = None,
        source_filter: list[str] | None = None,
    ) -> None:
        """Initialize harvest worker.

        Args:
            config: Application configuration
            worker_id: Unique identifier for this worker (auto-generated if None)
            source_filter: Only process these source types (None = all)
        """
        self.config = config
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.source_filter = source_filter

        # Configure proxy if specified in config
        proxy = config.scraping.proxy_url if config.scraping.proxy_url else None

        self.http = httpx.Client(
            timeout=config.scraping.timeout,
            follow_redirects=True,
            headers={"User-Agent": config.scraping.user_agent},
            proxy=proxy,
            trust_env=False,  # Don't inherit proxy from environment
        )
        self.storage = create_storage_backend(config.storage)
        self.pair_finder = PairFinder(config.validation)
        self.rate_limiter = DomainRateLimiter(config.scraping.request_delay)

    def run(self) -> None:
        """Main worker loop.

        Continuously claims and processes tasks until no tasks are available
        for several consecutive polls. Uses exponential backoff when no tasks
        are found.
        """
        consecutive_empty = 0

        while consecutive_empty < 5:  # Exit after 5 empty polls
            with get_session() as db:
                task = self._claim_task(db)

                if task is None:
                    consecutive_empty += 1
                    time.sleep(2**consecutive_empty)  # Exponential backoff
                    continue

                consecutive_empty = 0

                try:
                    self._process_task(db, task)
                    self._mark_completed(db, task.id)
                except Exception as e:
                    self._mark_failed(db, task.id, str(e))

    def _claim_task(self, db: Session) -> HarvestTask | None:
        """Claim next available task using FOR UPDATE SKIP LOCKED.

        This ensures multiple workers can safely claim tasks concurrently
        without conflicts.

        Args:
            db: Database session

        Returns:
            Claimed HarvestTask or None if no pending tasks
        """
        result = db.execute(
            text("""
            UPDATE harvest_tasks
            SET status = 'claimed', claimed_by = :worker, claimed_at = NOW()
            WHERE id = (
                SELECT id FROM harvest_tasks
                WHERE status = 'pending'
                ORDER BY id ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING *
        """),
            {"worker": self.worker_id},
        )
        row = result.fetchone()
        if row is None:
            return None

        return db.get(HarvestTask, row.id)

    def _process_task(self, db: Session, task: HarvestTask) -> None:
        """Process a single harvest task.

        Args:
            db: Database session
            task: Task to process
        """
        # Get source type from task's source
        source = db.get(Source, task.source_id) if task.source_id else None
        source_type = source.source_type if source else "mock"

        source_plugin = get_source(source_type)

        # Respect rate limits
        self.rate_limiter.wait(task.url)

        # Fetch and parse page
        response = self.http.get(task.url)
        response.raise_for_status()
        content = source_plugin.extract_images(task.url, response.text)

        if len(content.candidate_images) < 2:
            return  # Need at least 2 images to form a pair

        # Download all candidate images
        images = self._download_candidates(content.candidate_images)
        if len(images) < 2:
            return

        # Find valid pairs using SIFT validation on all combinations
        pairs = self.pair_finder.find_pairs(images)

        # Store each valid pair
        for original, capture, validation in pairs:
            self._store_pair(db, task, original, capture, validation, content)

    def _download_candidates(
        self, candidates: list[CandidateImage]
    ) -> list[tuple[CandidateImage, bytes, np.ndarray]]:
        """Download candidate images, respecting rate limits.

        Args:
            candidates: List of candidate images to download

        Returns:
            List of (candidate, raw_bytes, decoded_array) tuples for
            successfully downloaded images meeting size requirements
        """
        results: list[tuple[CandidateImage, bytes, np.ndarray]] = []

        for candidate in candidates:
            self.rate_limiter.wait(candidate.url)

            try:
                response = self.http.get(candidate.url)
                response.raise_for_status()
                data = response.content

                # Decode image
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None and min(img.shape[:2]) >= self.config.validation.min_image_size:
                    results.append((candidate, data, img))

            except Exception:
                continue  # Skip failed downloads

        return results

    def _store_pair(
        self,
        db: Session,
        task: HarvestTask,
        original: tuple[CandidateImage, bytes, np.ndarray],
        capture: tuple[CandidateImage, bytes, np.ndarray],
        validation: ValidationResult,
        content: PageContent,
    ) -> None:
        """Store validated pair to database.

        Args:
            db: Database session
            task: Source task
            original: Tuple of (candidate, raw_bytes, array) for original image
            capture: Tuple of (candidate, raw_bytes, array) for capture image
            validation: ValidationResult from pair finder
            content: PageContent with metadata
        """
        orig_candidate, orig_data, _ = original
        cap_candidate, cap_data, _ = capture

        # Store images with deduplication
        orig_image = get_or_create_image(db, orig_data, orig_candidate.url, self.storage)
        cap_image = get_or_create_image(db, cap_data, cap_candidate.url, self.storage)

        # Create pair record
        pair = Pair(
            task_id=task.id,
            original_image_id=orig_image.id,
            capture_image_id=cap_image.id,
            capture_type=content.likely_capture_type,
            quality_score=validation.quality_score,
            corners=validation.corners,
            extra_data={
                "source_url": task.url,
                **content.metadata,
            },
        )
        db.add(pair)

    def _mark_completed(self, db: Session, task_id: int) -> None:
        """Mark task as completed.

        Args:
            db: Database session
            task_id: ID of task to mark
        """
        db.execute(
            text("""
            UPDATE harvest_tasks
            SET status = 'completed', completed_at = NOW()
            WHERE id = :id
        """),
            {"id": task_id},
        )

    def _mark_failed(self, db: Session, task_id: int, error: str) -> None:
        """Mark task as failed.

        Args:
            db: Database session
            task_id: ID of task to mark
            error: Error message to record
        """
        db.execute(
            text("""
            UPDATE harvest_tasks
            SET status = 'failed', error_message = :error, retry_count = retry_count + 1
            WHERE id = :id
        """),
            {"id": task_id, "error": error},
        )

    def close(self) -> None:
        """Clean up resources."""
        self.http.close()
