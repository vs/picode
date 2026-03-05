"""CLI entry point for picode-scraper."""

from pathlib import Path

import click

from picode_scraper import __version__
from picode_scraper.config import Config, load_config


@click.group()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to config file",
)
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None) -> None:
    """Picode dataset scraper - collect original/capture image pairs."""
    ctx.ensure_object(dict)

    if config_path:
        try:
            ctx.obj["config"] = load_config(config_path)
        except (ValueError, FileNotFoundError) as e:
            raise click.ClickException(str(e))
    else:
        ctx.obj["config"] = None


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show harvest progress and statistics."""
    config: Config | None = ctx.obj.get("config")

    if config is None:
        click.echo("Status: No config file provided")
        click.echo("Use --config to specify a configuration file")
        return

    hosts = config.database.url.hosts()
    host_str = hosts[0]["host"] if hosts else "unknown"
    click.echo(f"Status: Connected to {host_str}")
    click.echo(f"Storage backend: {config.storage.backend}")


@cli.command()
@click.option("--source", "-s", multiple=True, help="Source types to search")
@click.option("--terms", "-t", multiple=True, help="Search terms")
@click.pass_context
def discover(ctx: click.Context, source: tuple[str, ...], terms: tuple[str, ...]) -> None:
    """Discover sources and populate harvest queue."""
    config: Config | None = ctx.obj.get("config")

    if config is None:
        click.echo("Error: Config file required for discover command")
        raise SystemExit(1)

    click.echo("Discover command not yet implemented")


@cli.command()
@click.option("--worker-id", default=None, help="Unique worker identifier")
@click.option("--source", "-s", multiple=True, help="Limit to specific sources")
@click.pass_context
def harvest(ctx: click.Context, worker_id: str | None, source: tuple[str, ...]) -> None:
    """Run a harvest worker to collect image pairs."""
    config: Config | None = ctx.obj.get("config")

    if config is None:
        click.echo("Error: Config file required for harvest command")
        raise SystemExit(1)

    click.echo("Harvest command not yet implemented")


if __name__ == "__main__":
    cli()
