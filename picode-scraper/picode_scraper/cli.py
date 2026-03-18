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
@click.option("--json-logs", is_flag=True, help="Output JSON formatted logs")
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None, json_logs: bool, log_level: str) -> None:
    """Picode dataset scraper - collect original/capture image pairs."""
    from picode_scraper.logging import configure_logging

    configure_logging(json_format=json_logs, level=log_level)

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
    from sqlalchemy import text

    from picode_scraper.db import get_session, init_db

    config: Config | None = ctx.obj.get("config")

    if config is None:
        click.echo("Error: Config file required for status command")
        raise SystemExit(1)

    # Initialize database
    init_db(config.database)

    with get_session() as db:
        # Task status counts
        task_stats = db.execute(
            text("""
                SELECT status, COUNT(*) as count
                FROM harvest_tasks
                GROUP BY status
                ORDER BY status
            """)
        ).fetchall()

        # Pair count
        pair_count = db.execute(text("SELECT COUNT(*) FROM pairs")).scalar() or 0

        # Image count
        image_count = db.execute(text("SELECT COUNT(*) FROM images")).scalar() or 0

        # Active workers (claimed in last 5 minutes)
        active_workers = db.execute(
            text("""
                SELECT COUNT(DISTINCT claimed_by)
                FROM harvest_tasks
                WHERE status = 'claimed'
                AND claimed_at > NOW() - INTERVAL '5 minutes'
            """)
        ).scalar() or 0

    # Display results
    click.echo("Tasks:")
    total_tasks = 0
    for row in task_stats:
        status = row[0]
        count = int(row[1])
        click.echo(f"  {status}: {count}")
        total_tasks += count

    if total_tasks == 0:
        click.echo("  (no tasks)")

    click.echo(f"\nPairs collected: {pair_count}")
    click.echo(f"Images stored: {image_count}")
    click.echo(f"Active workers: {active_workers}")


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
@click.option("--output", "-o", required=True, type=click.Path(), help="Output directory")
@click.option("--train-ratio", default=0.8, type=float, help="Training set ratio")
@click.option("--val-ratio", default=0.1, type=float, help="Validation set ratio")
@click.option("--test-ratio", default=0.1, type=float, help="Test set ratio")
@click.option("--min-quality", default=0.0, type=float, help="Minimum quality score filter")
@click.option("--no-symlinks", is_flag=True, help="Skip creating by_type symlinks")
@click.option("--seed", type=int, default=None, help="Random seed for reproducible splits")
@click.pass_context
def export(
    ctx: click.Context,
    output: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_quality: float,
    no_symlinks: bool,
    seed: int | None,
) -> None:
    """Export collected pairs to training dataset format."""
    from picode_scraper.db import init_db
    from picode_scraper.export import ExportConfig, ExportService

    config: Config | None = ctx.obj.get("config")

    if config is None:
        click.echo("Error: Config file required for export command")
        raise SystemExit(1)

    # Initialize database
    init_db(config.database)

    # Create export config from CLI options
    export_config = ExportConfig(
        output_dir=output,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_quality_score=min_quality,
        create_symlinks=not no_symlinks,
    )

    click.echo(f"Exporting dataset to: {output}")
    click.echo(f"Split ratios - train: {train_ratio}, val: {val_ratio}, test: {test_ratio}")
    if min_quality > 0:
        click.echo(f"Minimum quality score: {min_quality}")
    if seed is not None:
        click.echo(f"Random seed: {seed}")

    # Run export
    service = ExportService(export_config)
    result = service.export(seed=seed)

    # Display results
    click.echo("\nExport complete:")
    click.echo(f"  Total pairs: {result['total_pairs']}")
    click.echo(f"  Train: {result['train_count']}")
    click.echo(f"  Val: {result['val_count']}")
    click.echo(f"  Test: {result['test_count']}")

    if result["total_pairs"] == 0:
        click.echo("\nWarning: No pairs found to export")


@cli.command()
@click.option("--status", "-s", default="failed", help="Status to requeue (failed, dead_letter)")
@click.option("--limit", "-l", type=int, default=None, help="Max tasks to requeue")
@click.pass_context
def requeue(ctx: click.Context, status: str, limit: int | None) -> None:
    """Requeue failed or dead_letter tasks for retry."""
    from sqlalchemy import text

    from picode_scraper.db import get_session, init_db

    config: Config | None = ctx.obj.get("config")
    if config is None:
        click.echo("Error: Config file required for requeue command")
        raise SystemExit(1)

    init_db(config.database)

    valid_statuses = ("failed", "dead_letter")
    if status not in valid_statuses:
        click.echo(f"Error: Status must be one of {valid_statuses}")
        raise SystemExit(1)

    with get_session() as db:
        if limit:
            # Requeue only up to limit tasks
            result = db.execute(
                text("""
                    UPDATE harvest_tasks
                    SET status = 'pending', error_message = NULL,
                        claimed_by = NULL, claimed_at = NULL, retry_count = 0
                    WHERE id IN (
                        SELECT id FROM harvest_tasks
                        WHERE status = :status
                        LIMIT :limit
                    )
                """),
                {"status": status, "limit": limit},
            )
        else:
            result = db.execute(
                text("""
                    UPDATE harvest_tasks
                    SET status = 'pending', error_message = NULL,
                        claimed_by = NULL, claimed_at = NULL, retry_count = 0
                    WHERE status = :status
                """),
                {"status": status},
            )
        count = result.rowcount  # type: ignore[attr-defined]

    click.echo(f"Requeued {count} tasks from '{status}' to 'pending'")


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
