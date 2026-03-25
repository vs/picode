#!/usr/bin/env python
"""Local test setup for picode-scraper.

Creates sample test data, initializes database, and prepares for local testing.
"""

import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from picode_scraper.config import load_config
from picode_scraper.db import Base, HarvestTask, Source, get_engine, init_db


def create_sample_images(base_dir: Path) -> None:
    """Create sample original and capture image pairs for testing."""
    images_dir = base_dir / "sample_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating sample images in {images_dir}")

    # Create distinctive "original" images with recognizable patterns
    for i in range(3):
        # Original image with distinctive features
        original = np.zeros((400, 400, 3), dtype=np.uint8)

        # Add colored rectangles as features
        colors = [
            [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)],
            [(255, 128, 0), (128, 0, 255), (0, 255, 128), (255, 0, 128)],
            [(64, 64, 255), (255, 64, 64), (64, 255, 64), (128, 128, 128)],
        ][i]

        # Four quadrants with different colors
        original[50:150, 50:150] = colors[0]
        original[50:150, 250:350] = colors[1]
        original[250:350, 50:150] = colors[2]
        original[250:350, 250:350] = colors[3]

        # Add some text/patterns for SIFT features
        cv2.putText(original, f"IMG{i+1}", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.circle(original, (200, 200), 30, (200, 200, 200), 3)

        # Save original
        cv2.imwrite(str(images_dir / f"original_{i+1}.jpg"), original)
        print(f"  Created original_{i+1}.jpg")

        # Create "capture" - original embedded in larger image with some noise/transform
        capture = np.ones((600, 600, 3), dtype=np.uint8) * 40  # Dark gray background

        # Embed original with slight offset and scale
        # Simulate "photograph of screen"
        y_off, x_off = 100, 100
        capture[y_off:y_off + 400, x_off:x_off + 400] = original

        # Add some noise to simulate real capture
        noise = np.random.normal(0, 5, capture.shape).astype(np.int16)
        capture = np.clip(capture.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Slight blur to simulate camera capture
        capture = cv2.GaussianBlur(capture, (3, 3), 0)

        cv2.imwrite(str(images_dir / f"capture_{i+1}.jpg"), capture)
        print(f"  Created capture_{i+1}.jpg")

    # Create an unrelated image (should not match)
    unrelated = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "unrelated.jpg"), unrelated)
    print("  Created unrelated.jpg")


def create_test_html(base_dir: Path, server_port: int = 8765) -> None:
    """Create test HTML pages that reference the sample images."""
    html_dir = base_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating test HTML pages in {html_dir}")

    # Thread page 1 - has matching original and capture
    thread1 = f"""<!DOCTYPE html>
<html>
<head><title>Monitor vs Print Test Thread 1</title></head>
<body>
<div class="thread-container">
    <h1 class="thread-title">Test: Monitor vs Print comparison</h1>
    <div class="post" id="post-1">
        <div class="post-header">
            <img class="avatar" src="http://localhost:{server_port}/sample_images/avatar.jpg" width="50" height="50">
            <span class="username">TestUser1</span>
        </div>
        <div class="post-content">
            <p>Here's my original image:</p>
            <img src="http://localhost:{server_port}/sample_images/original_1.jpg" width="400" height="400" alt="Original image">
        </div>
    </div>
    <div class="post" id="post-2">
        <div class="post-header">
            <span class="username">TestUser2</span>
        </div>
        <div class="post-content">
            <p>Here's the photo I took of my screen showing that image:</p>
            <img src="http://localhost:{server_port}/sample_images/capture_1.jpg" width="600" height="600" alt="Screen capture">
        </div>
    </div>
</div>
</body>
</html>"""

    (html_dir / "thread1.html").write_text(thread1)
    print("  Created thread1.html")

    # Thread page 2 - multiple images including matching pairs
    thread2 = f"""<!DOCTYPE html>
<html>
<head><title>Soft Proofing Results Thread</title></head>
<body>
<div class="thread-container">
    <h1 class="thread-title">Soft proofing comparison test</h1>
    <div class="post" id="post-1">
        <div class="post-content">
            <p>Testing different color profiles. Original:</p>
            <img src="http://localhost:{server_port}/sample_images/original_2.jpg" width="400" height="400" alt="Test original">
            <p>Capture from monitor:</p>
            <img src="http://localhost:{server_port}/sample_images/capture_2.jpg" width="600" height="600" alt="Monitor capture">
        </div>
    </div>
    <div class="post" id="post-2">
        <div class="post-content">
            <p>Another test with third image:</p>
            <img src="http://localhost:{server_port}/sample_images/original_3.jpg" width="400" height="400" alt="Third original">
            <img src="http://localhost:{server_port}/sample_images/capture_3.jpg" width="600" height="600" alt="Third capture">
        </div>
    </div>
</div>
</body>
</html>"""

    (html_dir / "thread2.html").write_text(thread2)
    print("  Created thread2.html")

    # Thread page 3 - no valid pairs (unrelated images)
    thread3 = f"""<!DOCTYPE html>
<html>
<head><title>Random Images Thread</title></head>
<body>
<div class="thread-container">
    <h1 class="thread-title">Random discussion</h1>
    <div class="post" id="post-1">
        <div class="post-content">
            <img src="http://localhost:{server_port}/sample_images/original_1.jpg" width="400" height="400">
            <img src="http://localhost:{server_port}/sample_images/unrelated.jpg" width="400" height="400">
        </div>
    </div>
</div>
</body>
</html>"""

    (html_dir / "thread3.html").write_text(thread3)
    print("  Created thread3.html")


def init_database(config_path: str) -> None:
    """Initialize database and create schema."""
    print(f"Initializing database from config: {config_path}")

    config = load_config(config_path)
    engine = init_db(config.database)

    # Create all tables
    Base.metadata.create_all(engine)
    print("  Database schema created")


def create_test_tasks(config_path: str, server_port: int = 8765) -> None:
    """Create test harvest tasks in the database."""
    from picode_scraper.db import get_session

    print("Creating test harvest tasks")

    config = load_config(config_path)
    init_db(config.database)

    with get_session() as db:
        # Create source
        source = Source(
            source_type="mock",
            url=f"http://localhost:{server_port}/html/",
            title="Local Test DPReview Archive",
            priority=1,
        )
        db.add(source)
        db.flush()

        # Create tasks for each test page
        for i, page in enumerate(["thread1.html", "thread2.html", "thread3.html"]):
            task = HarvestTask(
                source_id=source.id,
                url=f"http://localhost:{server_port}/html/{page}",
                status="pending",
            )
            db.add(task)
            print(f"  Created task for {page}")

    print("  Tasks committed to database")


def main():
    """Run the local test setup."""
    import argparse

    parser = argparse.ArgumentParser(description="Setup local test environment")
    parser.add_argument(
        "--config",
        default="configs/local_test.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP server port",
    )
    parser.add_argument(
        "--data-dir",
        default="./test_data",
        help="Directory for test data",
    )
    args = parser.parse_args()

    # Change to scraper directory
    os.chdir(Path(__file__).parent.parent)

    base_dir = Path(args.data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create images directory for storage
    (base_dir / "images").mkdir(exist_ok=True)

    print("=" * 60)
    print("Picode Scraper Local Test Setup")
    print("=" * 60)

    # Step 1: Create sample images
    create_sample_images(base_dir)

    # Step 2: Create test HTML pages
    create_test_html(base_dir, args.port)

    # Step 3: Initialize database
    init_database(args.config)

    # Step 4: Create test tasks
    create_test_tasks(args.config, args.port)

    print()
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. Start HTTP server:  python -m http.server {args.port} --directory {base_dir}")
    print(f"  2. Run harvester:      picode-scraper harvest -c {args.config}")
    print(f"  3. Check status:       picode-scraper status -c {args.config}")
    print()


if __name__ == "__main__":
    main()
