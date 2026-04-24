"""
Command-line entry point.

Usage:
    kb list-sources
    kb ingest --source src_sample_public
    kb ingest --source src_sample_public --stage index
    kb ingest --source src_sample_public --stage embed --limit 5
    kb inspect --source src_sample_public --doc-index 0 --show-children
    kb health
"""

from __future__ import annotations

import logging
import sys

import click

from kb.orchestration import ingest_source, load_source_inventory
from kb.orchestration.pipeline import _STAGE_ORDER


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


# --------------------------------------------------------------------------- #
# list-sources
# --------------------------------------------------------------------------- #

@cli.command("list-sources")
@click.option(
    "--inventory",
    default="data/source_inventory.json",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
def list_sources(inventory: str) -> None:
    sources = load_source_inventory(inventory)
    for s in sources:
        click.echo(
            f"- {s['source_id']:28s}"
            f"  type={s.get('connector_type','?'):10s}"
            f"  lane={s.get('sensitivity_lane','?'):20s}"
            f"  vis={s.get('default_visibility','?')}"
        )


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #

@cli.command("ingest")
@click.option("--source", required=True, help="source_id from the inventory.")
@click.option(
    "--inventory",
    default="data/source_inventory.json",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--stage",
    type=click.Choice(_STAGE_ORDER),
    default="chunk",
    show_default=True,
    help="Run the pipeline up to this stage.",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Write per-document JSONL here (omit to discard).",
)
@click.option("--limit", type=int, default=None, help="Stop after N documents.")
@click.option(
    "--force-reindex",
    is_flag=True,
    help="Ignore the record manager and re-index every document.",
)
def ingest(
    source: str,
    inventory: str,
    stage: str,
    output_dir: str | None,
    limit: int | None,
    force_reindex: bool,
) -> None:
    """Ingest a single source up to the chosen stage."""
    stats = ingest_source(
        source_id=source,
        stage=stage,  # type: ignore[arg-type]
        inventory_path=inventory,
        output_dir=output_dir,
        limit=limit,
        force_reindex=force_reindex,
    )
    click.echo("")
    click.echo("─" * 60)
    click.echo(f"source:                  {stats.source_id}")
    click.echo(f"stage:                   {stage}")
    click.echo(f"documents_seen:          {stats.documents_seen}")
    click.echo(f"documents_processed:     {stats.documents_processed}")
    click.echo(f"documents_skipped:       {stats.documents_skipped_unchanged}")
    click.echo(f"documents_failed:        {stats.documents_failed}")
    click.echo(f"parents_written:         {stats.parents_written}")
    click.echo(f"children_written:        {stats.children_written}")
    click.echo(f"vectors_written:         {stats.vectors_written}")
    click.echo(f"elapsed_s:               {stats.elapsed_s}")
    if stats.errors:
        click.echo("\nerrors:")
        for e in stats.errors[:10]:
            click.echo(f"  - {e}")
        if len(stats.errors) > 10:
            click.echo(f"  ... ({len(stats.errors) - 10} more)")
    if stats.documents_failed:
        sys.exit(1)


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #

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
@click.option("--enrich", is_flag=True, help="Also run enrichment on this document.")
def inspect(
    source: str,
    inventory: str,
    doc_index: int,
    show_children: bool,
    enrich: bool,
) -> None:
    """Ingest a source and print one document's chunk tree (debug helper)."""
    from kb.chunking import chunk_document
    from kb.connectors import get_connector
    from kb.enrichment import Enricher
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
        if enrich:
            chunked = Enricher().enrich(chunked)  # type: ignore[assignment]

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
                        f"tags={c.metadata.get('sensitivity_tags', [])}"
                    )
                    click.echo(f"        text: {snip}{'…' if len(c.content) > 120 else ''}")
                    if enrich:
                        qs = getattr(c, "hypothetical_questions", None) or []
                        summ = getattr(c, "summary", None)
                        if summ:
                            click.echo(f"        summary: {summ}")
                        for i, q in enumerate(qs):
                            click.echo(f"        q{i}: {q}")
        return

    click.echo(f"No document at index {doc_index} for source {source}", err=True)
    sys.exit(2)


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #

@cli.command("health")
def health() -> None:
    """Check connectivity to the data plane (Postgres, Qdrant, BM25 dir)."""
    from kb.indexing import PostgresWriter, QdrantWriter
    from kb.indexing.bm25_writer import BM25Writer

    pg_ok = PostgresWriter().health()
    qd_ok = QdrantWriter().health()
    try:
        BM25Writer().stats("__healthcheck__")
        bm25_ok = True
    except Exception as exc:  # noqa: BLE001
        click.echo(f"BM25  FAIL: {exc}", err=True)
        bm25_ok = False

    click.echo(f"postgres : {'OK' if pg_ok else 'FAIL'}")
    click.echo(f"qdrant   : {'OK' if qd_ok else 'FAIL'}")
    click.echo(f"bm25 dir : {'OK' if bm25_ok else 'FAIL'}")
    if not (pg_ok and qd_ok and bm25_ok):
        sys.exit(1)


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
