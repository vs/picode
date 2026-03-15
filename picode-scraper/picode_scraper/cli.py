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
@click.option("--list", "list_sources_flag", is_flag=True, help="List available sources")
@click.pass_context
def discover(
    ctx: click.Context,
    source: tuple[str, ...],
    terms: tuple[str, ...],
    list_sources_flag: bool,
) -> None:
    """Discover sources and populate harvest queue."""
    from picode_scraper.db import HarvestTask, get_session, init_db
    from picode_scraper.db import Source as DBSource
    from picode_scraper.sources import get_source, list_sources

    # Handle --list flag (doesn't require config)
    if list_sources_flag:
        available = list_sources()
        click.echo("Available source plugins:")
        for name in available:
            plugin = get_source(name)
            click.echo(f"  - {name}: {plugin.display_name}")
        return

    config: Config | None = ctx.obj.get("config")

    if config is None:
        click.echo("Error: Config file required for discover command")
        raise SystemExit(1)

    # Initialize database
    init_db(config.database)

    # Get sources to search
    available = list_sources()
    source_types = list(source) if source else available

    # Validate source types
    invalid = set(source_types) - set(available)
    if invalid:
        click.echo(f"Error: Unknown sources: {invalid}")
        click.echo(f"Available: {available}")
        raise SystemExit(1)

    # Get search terms
    search_terms = list(terms) if terms else ["screen capture comparison"]

    click.echo(f"Discovering from sources: {source_types}")
    click.echo(f"Search terms: {search_terms}")

    total_discovered = 0
    total_tasks = 0

    with get_session() as db:
        for source_type in source_types:
            plugin = get_source(source_type)
            click.echo(f"\nSearching {plugin.display_name}...")

            for discovered in plugin.discover(search_terms):
                # Create or get Source record
                db_source = db.query(DBSource).filter_by(url=discovered.url).first()
                if not db_source:
                    db_source = DBSource(
                        source_type=discovered.source_type,
                        url=discovered.url,
                        title=discovered.title,
                        priority=discovered.priority,
                        extra_data=discovered.metadata,
                    )
                    db.add(db_source)
                    db.flush()  # Get the ID
                    total_discovered += 1

                # Create HarvestTask for the source
                existing_task = db.query(HarvestTask).filter_by(url=discovered.url).first()
                if not existing_task:
                    task = HarvestTask(
                        source_id=db_source.id,
                        url=discovered.url,
                        status="pending",
                    )
                    db.add(task)
                    total_tasks += 1

                click.echo(f"  Found: {discovered.title[:60]}...")

    click.echo("\nDiscovery complete:")
    click.echo(f"  New sources: {total_discovered}")
    click.echo(f"  New tasks: {total_tasks}")


@cli.command()
@click.option("--worker-id", default=None, help="Unique worker identifier")
@click.option("--source", "-s", multiple=True, help="Limit to specific sources")
@click.pass_context
def harvest(ctx: click.Context, worker_id: str | None, source: tuple[str, ...]) -> None:
    """Run a harvest worker to collect image pairs."""
    from picode_scraper.db import init_db
    from picode_scraper.harvester import HarvestWorker

    config: Config | None = ctx.obj.get("config")

    if config is None:
        click.echo("Error: Config file required for harvest command")
        raise SystemExit(1)

    # Initialize database
    init_db(config.database)

    # Create worker with optional source filter
    source_filter = list(source) if source else None

    click.echo("Starting harvest worker...")
    if worker_id:
        click.echo(f"Worker ID: {worker_id}")
    if source_filter:
        click.echo(f"Source filter: {source_filter}")

    worker = HarvestWorker(
        config=config,
        worker_id=worker_id,
        source_filter=source_filter,
    )

    try:
        click.echo("Worker running. Press Ctrl+C to stop.")
        worker.run()
        click.echo("Worker finished (no more tasks)")
    except KeyboardInterrupt:
        click.echo("\nWorker stopped")
    finally:
        worker.close()


if __name__ == "__main__":
    cli()
