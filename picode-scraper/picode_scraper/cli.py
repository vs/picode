"""Command-line interface for picode-scraper."""

import click

from picode_scraper import __version__


@click.group()
@click.version_option(version=__version__, prog_name="picode-scraper")
def cli() -> None:
    """Picode dataset scraper for original/capture image pairs."""
    pass


@cli.command()
def status() -> None:
    """Show the current status of the scraper."""
    click.echo("Status: No database configured")


if __name__ == "__main__":
    cli()
