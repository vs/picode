"""Command-line interface for distortions."""

from pathlib import Path

import click
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from distortions import (
    BrightnessHue,
    Contrast,
    Crop,
    GaussianBlur,
    GaussianNoise,
    JPEGCompression,
    MotionBlur,
    PerspectiveWarp,
    RandomBlur,
    Rotation,
    Saturation,
    Scale,
)
from distortions.visualization import create_comparison, create_diff, create_intensity_grid

# Registry of available distortions (using correct class names)
DISTORTIONS = {
    "gaussian-noise": GaussianNoise,
    "gaussian-blur": GaussianBlur,
    "motion-blur": MotionBlur,
    "random-blur": RandomBlur,
    "brightness-hue": BrightnessHue,
    "contrast": Contrast,
    "saturation": Saturation,
    "perspective-warp": PerspectiveWarp,
    "rotation": Rotation,
    "scale": Scale,
    "crop": Crop,
    "jpeg-compression": JPEGCompression,
}


def load_image(path: Path) -> torch.Tensor:
    """Load image as tensor (1, C, H, W) in [0, 1]."""
    img = Image.open(path).convert("RGB")
    tensor = pil_to_tensor(img).float() / 255.0
    return tensor.unsqueeze(0)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Save tensor (1, C, H, W) as image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = to_pil_image(tensor.squeeze(0).clamp(0, 1))
    img.save(path)


def apply_and_save(
    image: torch.Tensor,
    distortion_cls: type,
    output_dir: Path,
    intensity: float,
) -> None:
    """Apply distortion and save all visualization outputs."""
    distortion = distortion_cls(intensity=intensity)

    torch.manual_seed(42)
    distorted = distortion(image)

    save_image(distorted, output_dir / "distorted.png")
    save_image(create_diff(image, distorted), output_dir / "diff.png")
    save_image(create_comparison(image, distorted), output_dir / "comparison.png")
    save_image(create_intensity_grid(image, distortion), output_dir / "grid.png")


@click.group(invoke_without_command=True)
@click.option("--list", "list_distortions", is_flag=True, help="List available distortions")
@click.pass_context
def main(ctx: click.Context, list_distortions: bool) -> None:
    """Apply differentiable distortions to images."""
    if list_distortions:
        click.echo("Available distortions:")
        for name in sorted(DISTORTIONS.keys()):
            click.echo(f"  {name}")
        ctx.exit(0)

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("input", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(path_type=Path))
@click.option("--intensity", default=0.5, type=float, help="Distortion intensity (0-1)")
def all(input: Path, output: Path, intensity: float) -> None:
    """Apply all distortions, saving each to a subfolder."""
    image = load_image(input)

    for name, distortion_cls in DISTORTIONS.items():
        click.echo(f"Applying {name}...")
        apply_and_save(image, distortion_cls, output / name, intensity)

    click.echo(f"Done! Outputs saved to {output}")


# Create commands for each distortion
def create_distortion_command(name: str, distortion_cls: type) -> click.Command:
    """Create CLI command for a distortion."""

    @click.argument("input", type=click.Path(exists=True, path_type=Path))
    @click.option("-o", "--output", required=True, type=click.Path(path_type=Path))
    @click.option("--intensity", default=0.5, type=float, help="Distortion intensity (0-1)")
    def command(input: Path, output: Path, intensity: float) -> None:
        image = load_image(input)
        apply_and_save(image, distortion_cls, output, intensity)
        click.echo(f"Done! Outputs saved to {output}")

    command.__doc__ = f"Apply {name} distortion."
    return click.command(name=name)(command)


# Register distortion commands
for name, cls in DISTORTIONS.items():
    main.add_command(create_distortion_command(name, cls))


if __name__ == "__main__":
    main()
