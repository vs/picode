#!/usr/bin/env python3
"""
Lightning.ai SDK runner for remote training control.

This script allows you to:
- Create/start/stop Lightning.ai Studios remotely
- Run training jobs on GPU
- Monitor training progress
- Download checkpoints

Requirements:
    pip install lightning-sdk

Environment variables:
    LIGHTNING_API_KEY - Your Lightning.ai API key
    LIGHTNING_USER_ID - Your Lightning.ai user ID (optional)
    LIGHTNING_USERNAME - Your Lightning.ai username

Get your API key at: https://lightning.ai/<username>/home?settings=keys

Usage:
    # Start training (creates studio if needed)
    python scripts/lightning_runner.py train

    # Resume training
    python scripts/lightning_runner.py train --resume

    # Check status
    python scripts/lightning_runner.py status

    # Stop studio (preserves data)
    python scripts/lightning_runner.py stop

    # Download latest checkpoint
    python scripts/lightning_runner.py download

    # Delete studio (WARNING: deletes all data)
    python scripts/lightning_runner.py delete
"""

import argparse
import os
import sys


def get_studio(name: str = "picode-training", teamspace: str = "default"):
    """Get or create a Lightning.ai Studio."""
    from lightning_sdk import Studio, Machine

    username = os.environ.get("LIGHTNING_USERNAME")
    if not username:
        print("ERROR: Set LIGHTNING_USERNAME environment variable")
        sys.exit(1)

    # Check for API key
    if not os.environ.get("LIGHTNING_API_KEY"):
        print("ERROR: Set LIGHTNING_API_KEY environment variable")
        print("Get your key at: https://lightning.ai/{}/home?settings=keys".format(username))
        sys.exit(1)

    studio = Studio(name=name, teamspace=teamspace, user=username, create_ok=True)
    return studio


def cmd_train(args):
    """Start or resume training."""
    from lightning_sdk import Machine

    studio = get_studio(args.studio)
    
    # Select GPU machine
    machine_map = {
        "t4": Machine.T4,
        "l4": Machine.L4,
        "a10g": Machine.A10G,
        "a100": Machine.A100,
    }
    machine = machine_map.get(args.gpu.lower(), Machine.T4)
    
    print(f"Starting studio '{args.studio}' with {args.gpu.upper()} GPU...")
    studio.start(machine)
    
    print(f"Studio status: {studio.status}")
    print(f"Machine: {studio.machine}")
    
    # Clone/update repo
    print("\nSetting up repository...")
    studio.run("git clone https://github.com/vs/code.git /teamspace/studios/this_studio/code 2>/dev/null || (cd /teamspace/studios/this_studio/code && git pull)")
    
    # Install dependencies
    print("Installing dependencies...")
    studio.run("cd /teamspace/studios/this_studio/code/picode-model && pip install -q -e '.[lpips,kornia]'")
    
    # Check for training data
    print("\nChecking training data...")
    result = studio.run("ls /teamspace/studios/this_studio/data/train2017/*.jpg 2>/dev/null | wc -l")
    num_images = int(result.strip() or "0")
    
    if num_images == 0:
        print("WARNING: No training images found!")
        print("Upload images to: /teamspace/studios/this_studio/data/train2017/")
        print("Or run: python scripts/lightning_runner.py setup-data")
        if not args.force:
            print("\nUse --force to start training anyway")
            return
    else:
        print(f"Found {num_images} training images")
    
    # Build training command
    train_cmd = "cd /teamspace/studios/this_studio/code/picode-model && python scripts/lightning_train.py"
    if args.resume:
        train_cmd += " --resume"
    if args.batch_size:
        train_cmd += f" --batch-size {args.batch_size}"
    
    print(f"\nStarting training...")
    print(f"Command: {train_cmd}")
    print("-" * 60)
    
    # Run training (this will stream output)
    try:
        result = studio.run_with_exit_code(train_cmd)
        print(f"\nTraining finished with exit code: {result}")
    except KeyboardInterrupt:
        print("\n\nInterrupted! Studio is still running.")
        print("Use 'python scripts/lightning_runner.py stop' to stop it.")


def cmd_status(args):
    """Check studio status."""
    studio = get_studio(args.studio)
    
    print(f"Studio: {args.studio}")
    print(f"Status: {studio.status}")
    print(f"Machine: {studio.machine}")
    
    if str(studio.status) == "Status.Running":
        print("\nChecking training progress...")
        result = studio.run("ls -la /teamspace/studios/this_studio/checkpoints/*/checkpoint_*.pt 2>/dev/null | tail -5 || echo 'No checkpoints yet'")
        print(result)


def cmd_stop(args):
    """Stop the studio (preserves data)."""
    studio = get_studio(args.studio)
    
    print(f"Stopping studio '{args.studio}'...")
    studio.stop()
    print("Studio stopped. Data is preserved.")


def cmd_delete(args):
    """Delete the studio (WARNING: deletes all data)."""
    studio = get_studio(args.studio)
    
    if not args.force:
        confirm = input(f"DELETE studio '{args.studio}' and ALL its data? Type 'yes' to confirm: ")
        if confirm != "yes":
            print("Aborted.")
            return
    
    print(f"Deleting studio '{args.studio}'...")
    studio.delete()
    print("Studio deleted.")


def cmd_download(args):
    """Download checkpoint from studio."""
    studio = get_studio(args.studio)
    
    print("Finding latest checkpoint...")
    result = studio.run("ls -t /teamspace/studios/this_studio/checkpoints/*/checkpoint_*.pt 2>/dev/null | head -1")
    checkpoint = result.strip()
    
    if not checkpoint:
        # Try best.pt
        result = studio.run("ls /teamspace/studios/this_studio/checkpoints/*/best.pt 2>/dev/null | head -1")
        checkpoint = result.strip()
    
    if not checkpoint:
        print("No checkpoints found.")
        return
    
    print(f"Found: {checkpoint}")
    
    # Create local directory
    local_dir = args.output or "checkpoints"
    os.makedirs(local_dir, exist_ok=True)
    
    filename = os.path.basename(checkpoint)
    local_path = os.path.join(local_dir, filename)
    
    print(f"Downloading to {local_path}...")
    # Use scp-like download (Lightning SDK supports this)
    studio.run(f"cat {checkpoint} | base64 > /tmp/checkpoint.b64")
    b64_data = studio.run("cat /tmp/checkpoint.b64")
    
    import base64
    with open(local_path, "wb") as f:
        f.write(base64.b64decode(b64_data))
    
    print(f"Downloaded: {local_path}")


def cmd_setup_data(args):
    """Download COCO training data to studio."""
    studio = get_studio(args.studio)
    
    print("This will download COCO train2017 (~18GB) to the studio.")
    print("This only needs to be done once - data persists across sessions.")
    
    if not args.force:
        confirm = input("Continue? [y/N]: ")
        if confirm.lower() != "y":
            print("Aborted.")
            return
    
    print("\nDownloading COCO train2017 (this may take 20-30 minutes)...")
    studio.run("mkdir -p /teamspace/studios/this_studio/data")
    studio.run("cd /teamspace/studios/this_studio/data && wget -q http://images.cocodataset.org/zips/train2017.zip")
    studio.run("cd /teamspace/studios/this_studio/data && unzip -q train2017.zip && rm train2017.zip")
    
    result = studio.run("ls /teamspace/studios/this_studio/data/train2017/*.jpg | wc -l")
    print(f"Downloaded {result.strip()} images")


def cmd_logs(args):
    """Show recent training logs."""
    studio = get_studio(args.studio)
    
    print("Recent training output:")
    print("-" * 60)
    result = studio.run("tail -50 /teamspace/studios/this_studio/checkpoints/*/train.log 2>/dev/null || echo 'No logs found'")
    print(result)


def main():
    parser = argparse.ArgumentParser(description="Lightning.ai remote training runner")
    parser.add_argument(
        "--studio",
        default="picode-training",
        help="Studio name (default: picode-training)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # train
    train_parser = subparsers.add_parser("train", help="Start/resume training")
    train_parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    train_parser.add_argument("--gpu", default="t4", choices=["t4", "l4", "a10g", "a100"], help="GPU type")
    train_parser.add_argument("--batch-size", type=int, help="Override batch size")
    train_parser.add_argument("--force", action="store_true", help="Start even without data")
    
    # status
    subparsers.add_parser("status", help="Check studio status")
    
    # stop
    subparsers.add_parser("stop", help="Stop studio (preserves data)")
    
    # delete
    delete_parser = subparsers.add_parser("delete", help="Delete studio (WARNING: deletes data)")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")
    
    # download
    download_parser = subparsers.add_parser("download", help="Download latest checkpoint")
    download_parser.add_argument("--output", "-o", help="Output directory")
    
    # setup-data
    data_parser = subparsers.add_parser("setup-data", help="Download COCO training data")
    data_parser.add_argument("--force", action="store_true", help="Skip confirmation")
    
    # logs
    subparsers.add_parser("logs", help="Show recent training logs")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    commands = {
        "train": cmd_train,
        "status": cmd_status,
        "stop": cmd_stop,
        "delete": cmd_delete,
        "download": cmd_download,
        "setup-data": cmd_setup_data,
        "logs": cmd_logs,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
