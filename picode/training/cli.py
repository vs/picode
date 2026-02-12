"""CLI for training Picode models."""

from typing import Any

import click


def parse_overrides(args: list[str]) -> dict[str, Any]:
    """Parse key=value overrides into nested dict.

    Example: ["training.lr=0.001", "data.batch_size=8"]
    Returns: {"training": {"lr": 0.001}, "data": {"batch_size": 8}}

    Args:
        args: List of key=value strings with dot-separated keys.

    Returns:
        Nested dictionary of parsed overrides.
    """
    overrides: dict[str, Any] = {}
    for arg in args:
        if "=" not in arg:
            continue
        key, value_str = arg.split("=", 1)

        # Auto-convert types
        value: Any
        if value_str.lower() in ("true", "false"):
            value = value_str.lower() == "true"
        elif value_str.replace(".", "").replace("-", "").isdigit():
            value = float(value_str) if "." in value_str else int(value_str)
        else:
            value = value_str

        # Build nested dict
        parts = key.split(".")
        d = overrides
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value

    return overrides


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--resume", type=click.Path(exists=True), help="Resume from checkpoint")
@click.option("--eval-only", is_flag=True, help="Run evaluation only")
@click.option("--checkpoint", type=click.Path(exists=True), help="Checkpoint for eval-only mode")
@click.option("--robustness", is_flag=True, help="Run robustness sweep")
@click.argument("overrides", nargs=-1)
def main(
    config: str,
    resume: str | None,
    eval_only: bool,
    checkpoint: str | None,
    robustness: bool,
    overrides: tuple[str, ...],
) -> None:
    """Train Picode steganography models.

    CONFIG is the path to a YAML config file.

    Additional key=value pairs can be passed to override config values.

    Examples:

        picode-train config.yaml

        picode-train config.yaml training.lr=0.0003

        picode-train config.yaml --resume checkpoints/exp/checkpoint_00050000.pt

        picode-train config.yaml --eval-only --checkpoint checkpoints/exp/best.pt
    """
    import torch

    from picode.training.trainer import Trainer

    override_dict = parse_overrides(list(overrides))

    # Create or resume trainer
    if resume:
        click.echo(f"Resuming from {resume}")
        trainer = Trainer.from_checkpoint(resume)
    else:
        trainer = Trainer.from_config(config, override_dict)

    # Eval-only mode
    if eval_only:
        if checkpoint:
            trainer = Trainer.from_checkpoint(checkpoint)

        metrics = trainer.evaluate()
        click.echo("\nEvaluation Results:")
        click.echo(f"  Bit Accuracy:     {metrics.bit_accuracy:.4f}")
        click.echo(f"  Message Accuracy: {metrics.message_accuracy:.4f}")
        click.echo(f"  PSNR:             {metrics.psnr:.2f} dB")
        click.echo(f"  SSIM:             {metrics.ssim:.4f}")
        if metrics.lpips is not None:
            click.echo(f"  LPIPS:            {metrics.lpips:.4f}")

        if robustness:
            click.echo("\nRobustness Sweep:")
            images = next(iter(trainer.dataloader)).to(trainer.device)
            messages = torch.randint(
                0,
                2,
                (images.shape[0], trainer.config.training.num_bits),
                device=trainer.device,
            ).float()
            results = trainer.robustness_sweep(images, messages)

            current_dist = None
            for r in results:
                if r.distortion != current_dist:
                    current_dist = r.distortion
                    click.echo(f"\n  {r.distortion}:")
                click.echo(
                    f"    strength={r.strength:<6} "
                    f"bit_acc={r.bit_accuracy:.4f}  "
                    f"msg_acc={r.message_accuracy:.4f}"
                )

        return

    # Training mode
    click.echo(f"Starting training: {trainer.config.experiment_name}")
    click.echo(f"  Device: {trainer.device}")
    click.echo(f"  Steps:  {trainer.config.training.num_steps}")
    click.echo(f"  LR:     {trainer.config.training.lr}")
    click.echo()

    trainer.fit()
    click.echo("\nTraining complete!")


if __name__ == "__main__":
    main()
