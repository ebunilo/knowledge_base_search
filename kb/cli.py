"""
Command-line entry point.

Usage:
    python -m kb.cli ingest --source src_sample_docs
    python -m kb.cli ingest --source src_sample_docs --output-dir ./out --limit 5
    python -m kb.cli list-sources
    python -m kb.cli inspect --source src_sample_docs --doc-index 0
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from kb.orchestration import ingest_source, load_source_inventory


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Enterprise knowledge-base search — ingestion CLI."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)


@cli.command("list-sources")
@click.option(
    "--inventory",
    default="data/source_inventory.json",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
def list_sources(inventory: str) -> None:
    """List sources defined in the inventory."""
    sources = load_source_inventory(inventory)
    for s in sources:
        click.echo(
            f"- {s['source_id']:28s}"
            f"  type={s.get('connector_type','?'):10s}"
            f"  lane={s.get('sensitivity_lane','?'):20s}"
            f"  vis={s.get('default_visibility','?')}"
        )


@cli.command("ingest")
@click.option("--source", required=True, help="source_id from the inventory.")
@click.option(
    "--inventory",
    default="data/source_inventory.json",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Write ChunkedDocuments as JSONL here (omit to discard).",
)
@click.option("--limit", type=int, default=None, help="Stop after N documents.")
def ingest(source: str, inventory: str, output_dir: str | None, limit: int | None) -> None:
    """Ingest a single source (connector → parse → chunk)."""
    stats = ingest_source(
        source_id=source,
        inventory_path=inventory,
        output_dir=output_dir,
        limit=limit,
    )
    click.echo("")
    click.echo("─" * 60)
    click.echo(f"source:             {stats.source_id}")
    click.echo(f"documents_seen:     {stats.documents_seen}")
    click.echo(f"documents_processed:{stats.documents_processed}")
    click.echo(f"documents_failed:   {stats.documents_failed}")
    click.echo(f"parents_written:    {stats.parents_written}")
    click.echo(f"children_written:   {stats.children_written}")
    click.echo(f"elapsed_s:          {stats.elapsed_s}")
    if stats.errors:
        click.echo("\nerrors:")
        for e in stats.errors[:10]:
            click.echo(f"  - {e}")
        if len(stats.errors) > 10:
            click.echo(f"  ... ({len(stats.errors) - 10} more)")
    if stats.documents_failed:
        sys.exit(1)


@cli.command("inspect")
@click.option("--source", required=True)
@click.option(
    "--inventory",
    default="data/source_inventory.json",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--doc-index", type=int, default=0)
@click.option("--show-children", is_flag=True)
def inspect(source: str, inventory: str, doc_index: int, show_children: bool) -> None:
    """Ingest a source and print one document's chunk tree (debug helper)."""
    from kb.chunking import chunk_document
    from kb.connectors import get_connector
    from kb.orchestration.pipeline import get_source
    from kb.parsers import parse_document

    inv = load_source_inventory(inventory)
    src = get_source(source, inv)
    connector = get_connector(source, src)

    for idx, raw in enumerate(connector.iter_documents()):
        if idx < doc_index:
            continue
        parsed = parse_document(raw)
        chunked = chunk_document(parsed)

        click.echo(f"document_id:   {chunked.document_id}")
        click.echo(f"source_uri:    {chunked.source_uri}")
        click.echo(f"title:         {chunked.title}")
        click.echo(f"format:        {chunked.format}")
        click.echo(f"language:      {chunked.language}")
        click.echo(f"sensitivity:   {chunked.sensitivity}")
        click.echo(f"visibility:    {chunked.visibility}")
        click.echo(f"parents:       {len(chunked.parents)}")
        click.echo(f"children:      {len(chunked.children)}")
        click.echo("")
        for p in chunked.parents:
            click.echo(f"  PARENT [{p.ord}] tokens={p.token_count} path={p.section_path}")
            snippet = p.content[:160].replace("\n", " ")
            click.echo(f"    {snippet}{'…' if len(p.content) > 160 else ''}")
            if show_children:
                kids = [c for c in chunked.children if c.parent_id == p.parent_id]
                for c in kids:
                    snip = c.content[:120].replace("\n", " ")
                    click.echo(
                        f"      child [{c.ord}] tok={c.token_count} "
                        f"tags={c.metadata.get('sensitivity_tags', [])}  "
                        f"{snip}{'…' if len(c.content) > 120 else ''}"
                    )
        return

    click.echo(f"No document at index {doc_index} for source {source}", err=True)
    sys.exit(2)


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
