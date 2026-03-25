#!/usr/bin/env python
"""Run local test for picode-scraper with mocked HTTP.

This script bypasses the need for an HTTP server by mocking the HTTP client
to serve files from the local filesystem.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class MockResponse:
    """Mock HTTP response that reads from local files."""

    def __init__(self, content: bytes, text: str = "", status_code: int = 200):
        self.content = content
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class LocalFileHTTPClient:
    """Mock HTTP client that reads from local filesystem."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.is_closed = False

    def get(self, url: str) -> MockResponse:
        """Map URL to local file and return response."""
        # Extract path from URL (assumes localhost:8765)
        if "localhost:8765/" in url:
            path = url.split("localhost:8765/")[1]
        else:
            # For other URLs, try to find matching file
            path = url.rsplit("/", 1)[-1]

        local_path = self.base_dir / path

        if not local_path.exists():
            print(f"  [MOCK HTTP] File not found: {local_path}")
            return MockResponse(b"", "", 404)

        content = local_path.read_bytes()

        # Determine if text or binary
        if local_path.suffix in (".html", ".htm", ".txt"):
            text = content.decode("utf-8")
        else:
            text = ""

        print(f"  [MOCK HTTP] Served: {path} ({len(content)} bytes)")
        return MockResponse(content, text)

    def close(self):
        self.is_closed = True


def run_test():
    """Run the local test with mocked HTTP."""
    from picode_scraper.config import load_config
    from picode_scraper.db import Base, HarvestTask, Pair, get_engine, get_session, init_db
    from picode_scraper.harvester.worker import HarvestWorker

    os.chdir(Path(__file__).parent.parent)

    config_path = "configs/local_test.yaml"
    data_dir = Path("test_data")

    print("=" * 60)
    print("Picode Scraper Local Test Runner (Mocked HTTP)")
    print("=" * 60)

    # Check if setup was run
    if not (data_dir / "sample_images" / "original_1.jpg").exists():
        print("Error: Test data not found. Run local_test_setup.py first.")
        sys.exit(1)

    # Load config and init database
    config = load_config(config_path)
    engine = init_db(config.database)

    # Check pending tasks
    with get_session() as db:
        pending = db.query(HarvestTask).filter_by(status="pending").count()
        print(f"Pending tasks: {pending}")

        if pending == 0:
            print("No pending tasks. Re-running setup...")
            # Reset tasks to pending
            db.query(HarvestTask).update({"status": "pending", "error_message": None})
            pending = db.query(HarvestTask).count()
            print(f"Reset {pending} tasks to pending")

    # Create worker with mocked HTTP client
    print()
    print("Starting harvest worker with mocked HTTP...")
    print("-" * 60)

    worker = HarvestWorker(
        config=config,
        worker_id="local-test-worker",
    )

    # Replace HTTP client with mock
    mock_client = LocalFileHTTPClient(data_dir)
    worker.http.close()
    worker.http = mock_client

    try:
        # Run worker (will process all tasks then exit)
        worker.run()
    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        worker.close()

    # Show results
    print()
    print("-" * 60)
    print("Results:")
    print("-" * 60)

    with get_session() as db:
        # Task status
        for status in ["pending", "claimed", "completed", "failed", "dead_letter"]:
            count = db.query(HarvestTask).filter_by(status=status).count()
            if count > 0:
                print(f"  Tasks {status}: {count}")

        # Failed tasks details
        failed_tasks = db.query(HarvestTask).filter(
            HarvestTask.status.in_(["failed", "dead_letter"])
        ).all()
        for task in failed_tasks:
            print(f"    - {task.url}: {task.error_message}")

        # Pairs found
        pairs = db.query(Pair).all()
        print(f"  Pairs found: {len(pairs)}")

        for pair in pairs:
            print(f"    - Pair {pair.id}: quality={pair.quality_score:.2f}, type={pair.capture_type}")

    print()
    print("=" * 60)
    print("Test complete!")
    print("=" * 60)

    # Show storage
    images_dir = data_dir / "images"
    if images_dir.exists():
        stored_files = list(images_dir.glob("**/*"))
        stored_images = [f for f in stored_files if f.is_file()]
        print(f"Stored images: {len(stored_images)}")
        for img in stored_images[:5]:
            print(f"  - {img.relative_to(images_dir)}")
        if len(stored_images) > 5:
            print(f"  ... and {len(stored_images) - 5} more")


if __name__ == "__main__":
    run_test()
