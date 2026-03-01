"""Modal script to extract tarballs in a volume."""

import modal
import tarfile

app = modal.App("picode-extract")
volume = modal.Volume.from_name("picode-data")


@app.function(volumes={"/vol": volume}, timeout=3600)
def extract(tarball: str, dest: str) -> None:
    """Extract a tarball within the Modal volume.

    Args:
        tarball: Path to tarball within volume (e.g., /picode_upload_train2017.tar)
        dest: Destination directory within volume (e.g., /)
    """
    import os

    tarball_path = f"/vol{tarball}"
    dest_path = f"/vol{dest}"

    if not os.path.exists(tarball_path):
        raise FileNotFoundError(f"Tarball not found: {tarball_path}")

    print(f"Extracting {tarball_path} to {dest_path}...")

    with tarfile.open(tarball_path, "r") as tar:
        # Get total members for progress
        members = tar.getmembers()
        total = len(members)
        print(f"Total files to extract: {total}")

        for i, member in enumerate(members):
            tar.extract(member, dest_path)
            if (i + 1) % 10000 == 0:
                print(f"Extracted {i + 1}/{total} files...")

    print(f"Extraction complete! Files in {dest_path}")
    volume.commit()


@app.local_entrypoint()
def main(tarball: str, dest: str = "/") -> None:
    """Entry point for modal run."""
    extract.remote(tarball, dest)
