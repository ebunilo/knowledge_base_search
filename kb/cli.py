"""
Command-line entry point.

Usage:
    kb list-sources
    kb ingest --source src_sample_public
    kb ingest --source src_sample_public --stage index
    kb ingest --source src_sample_public --stage embed --limit 5
    kb inspect --source src_sample_public --doc-index 0 --show-children
    kb search --query "how does the API gateway authenticate?"
    kb search --query "..." --as-user u_001
    kb ask --query "..." --as-user u_001 --stream
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
# search
# --------------------------------------------------------------------------- #

@cli.command("search")
@click.option("--query", "-q", required=True, help="Natural-language query.")
@click.option(
    "--as-user",
    "as_user",
    default="anonymous",
    show_default=True,
    help="user_id / email from staff_directory.json, or 'anonymous'.",
)
@click.option(
    "--role",
    default=None,
    help="Override role (default: taken from staff_directory entry).",
)
@click.option(
    "--dept",
    default=None,
    help="Override department (default: taken from staff_directory entry).",
)
@click.option("--top-k", type=int, default=10, show_default=True)
@click.option("--top-k-dense", type=int, default=30, show_default=True)
@click.option("--top-k-sparse", type=int, default=30, show_default=True)
@click.option(
    "--dense-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="RRF weight for dense retriever.",
)
@click.option(
    "--sparse-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="RRF weight for sparse retriever.",
)
@click.option(
    "--rerank/--no-rerank",
    default=True,
    show_default=True,
    help="Cross-encoder rerank the fused candidates (bge-reranker-v2-m3).",
)
@click.option(
    "--rerank-top-n",
    type=int,
    default=30,
    show_default=True,
    help="How many fused candidates to send to the reranker.",
)
@click.option(
    "--rewrite",
    type=click.Choice(["off", "multi_query", "hyde", "both"]),
    default="off",
    show_default=True,
    help="Query rewriting strategy (LLM-backed).",
)
@click.option(
    "--multi-query-k",
    type=int,
    default=2,
    show_default=True,
    help="Number of paraphrases when --rewrite includes multi_query.",
)
@click.option(
    "--no-parent",
    is_flag=True,
    help="Skip parent content in output (faster, fewer DB reads).",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit JSON instead of the formatted text view.",
)
def search(
    query: str,
    as_user: str,
    role: str | None,
    dept: str | None,
    top_k: int,
    top_k_dense: int,
    top_k_sparse: int,
    dense_weight: float,
    sparse_weight: float,
    rerank: bool,
    rerank_top_n: int,
    rewrite: str,
    multi_query_k: int,
    no_parent: bool,
    as_json: bool,
) -> None:
    """Run a hybrid (dense + BM25) search against the indexed corpus."""
    from kb.retrieval import Retriever, RetrievalConfig, UserContext
    from kb.retrieval.acl import load_user

    try:
        user = load_user(as_user)
    except (FileNotFoundError, KeyError):
        user = UserContext(user_id=as_user or "anonymous")

    if role is not None:
        user = user.model_copy(update={"role": role})
    if dept is not None:
        user = user.model_copy(update={"department": dept})

    cfg = RetrievalConfig(
        top_k_dense=top_k_dense,
        top_k_sparse=top_k_sparse,
        top_k_final=top_k,
        rrf_dense_weight=dense_weight,
        rrf_sparse_weight=sparse_weight,
        rerank=rerank,
        rerank_top_n=rerank_top_n,
        rewrite_strategy=rewrite,  # type: ignore[arg-type]
        multi_query_k=multi_query_k,
        include_parent_content=not no_parent,
    )

    result = Retriever().retrieve(query=query, user=user, config=cfg)

    if as_json:
        click.echo(result.model_dump_json(indent=2))
        return

    click.echo("")
    click.echo(f"query: {result.query!r}")
    click.echo(
        f"user:  {user.user_id} (tenant={user.tenant_id or '∅'} "
        f"dept={user.department or '∅'} role={user.role})"
    )
    click.echo(f"collections: {', '.join(result.collections_searched)}")
    click.echo(
        f"rewrite:     {result.rewrite_strategy}"
        + (f"  variants={len(result.query_variants)}"
           if result.rewrite_strategy != "off" else "")
        + (f"  hyde={'yes' if result.hyde_passage else 'no'}"
           if result.rewrite_strategy in {"hyde", "both"} else "")
        + (f"  FALLBACK({result.rewrite_fallback})"
           if result.rewrite_fallback else "")
    )
    click.echo(
        f"rerank:      {'on' if result.rerank_applied else 'off'}"
        + (f"  reranked={result.reranked_candidates}"
           if result.rerank_applied else "")
        + (f"  FALLBACK({result.rerank_fallback})"
           if result.rerank_fallback else "")
    )
    click.echo(
        f"candidates:  dense={result.dense_candidates} "
        f"sparse={result.sparse_candidates} "
        f"fused={result.fused_candidates} "
        f"final={result.final_hits}"
    )
    click.echo(
        f"timing_ms:   rewrite={result.rewrite_ms} embed={result.embed_ms} "
        f"dense={result.dense_ms} sparse={result.sparse_ms} "
        f"fuse={result.fusion_ms} rerank={result.rerank_ms} "
        f"parents={result.parent_ms} total={result.total_ms}"
    )
    click.echo("─" * 60)

    if not result.hits:
        click.echo("(no hits)")
        return

    for i, h in enumerate(result.hits, start=1):
        click.echo(
            f"[{i}] score={h.score:.4f} "
            f"rr={_fmt(h.rerank_score)}/{h.rerank_rank or '-'} "
            f"rrf={_fmt(h.rrf_score)} "
            f"d={h.dense_rank}/{_fmt(h.dense_score)} "
            f"s={h.sparse_rank}/{_fmt(h.sparse_score)}  vis={h.visibility}"
        )
        click.echo(f"    title:     {h.title or '(no title)'}")
        click.echo(f"    section:   {h.section_path or '-'}")
        click.echo(f"    source:    {h.source_id}  ({h.source_uri})")
        for mv in h.matched_via:
            if mv.kind == "question" and mv.text:
                click.echo(f"    matched:   question '{mv.text}'")
            else:
                click.echo(f"    matched:   {mv.kind} rank={mv.rank}")
        snippet = (h.content or "").strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "…"
        click.echo(f"    snippet:   {snippet}")
        if not no_parent and h.parent_content:
            parent = h.parent_content.strip().replace("\n", " ")
            if len(parent) > 300:
                parent = parent[:300] + "…"
            click.echo(f"    parent:    {parent}")
        click.echo("")


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


# --------------------------------------------------------------------------- #
# ask — Phase 3 generation
# --------------------------------------------------------------------------- #

@cli.command("ask")
@click.option("--query", "-q", required=True, help="Natural-language question.")
@click.option(
    "--as-user",
    "as_user",
    default="anonymous",
    show_default=True,
    help="user_id / email from staff_directory.json, or 'anonymous'.",
)
@click.option("--role", default=None, help="Override role.")
@click.option("--dept", default=None, help="Override department.")
@click.option("--top-k", type=int, default=8, show_default=True,
              help="Number of context blocks to feed the LLM.")
@click.option(
    "--rerank/--no-rerank",
    default=True, show_default=True,
    help="Cross-encoder rerank the retrieved candidates.",
)
@click.option(
    "--rewrite",
    type=click.Choice(["off", "multi_query", "hyde", "both"]),
    default="off", show_default=True,
    help="Query rewriting strategy.",
)
@click.option(
    "--multi-query-k", type=int, default=2, show_default=True,
    help="Number of paraphrases when --rewrite includes multi_query.",
)
@click.option(
    "--context-budget", type=int, default=None,
    help="Max tokens for the CONTEXT block (default: settings).",
)
@click.option(
    "--max-tokens", type=int, default=None,
    help="Max tokens for the LLM answer (default: settings).",
)
@click.option(
    "--temperature", type=float, default=None,
    help="LLM sampling temperature (default: settings).",
)
@click.option(
    "--stream/--no-stream",
    default=True, show_default=True,
    help="Stream tokens to the terminal as they arrive.",
)
@click.option(
    "--json", "as_json", is_flag=True,
    help="Emit the full GenerationResult as JSON (implies --no-stream).",
)
def ask(
    query: str,
    as_user: str,
    role: str | None,
    dept: str | None,
    top_k: int,
    rerank: bool,
    rewrite: str,
    multi_query_k: int,
    context_budget: int | None,
    max_tokens: int | None,
    temperature: float | None,
    stream: bool,
    as_json: bool,
) -> None:
    """Ask the knowledge base. Retrieves, generates, and streams a cited answer."""
    from kb.generation import Generator, GenerationConfig
    from kb.retrieval import RetrievalConfig, UserContext
    from kb.retrieval.acl import load_user
    from kb.settings import get_settings

    try:
        user = load_user(as_user)
    except (FileNotFoundError, KeyError):
        user = UserContext(user_id=as_user or "anonymous")
    if role is not None:
        user = user.model_copy(update={"role": role})
    if dept is not None:
        user = user.model_copy(update={"department": dept})

    settings = get_settings()
    rcfg = RetrievalConfig(
        top_k_final=top_k,
        rerank=rerank,
        rewrite_strategy=rewrite,  # type: ignore[arg-type]
        multi_query_k=multi_query_k,
    )
    gcfg = GenerationConfig(
        context_budget_tokens=(
            context_budget
            if context_budget is not None
            else settings.generation_context_budget_tokens
        ),
        max_answer_tokens=(
            max_tokens
            if max_tokens is not None
            else settings.generation_max_answer_tokens
        ),
        temperature=(
            temperature
            if temperature is not None
            else settings.generation_temperature
        ),
        stream=stream and not as_json,
        include_summaries_in_context=settings.generation_include_summaries_in_context,
        min_score_threshold=settings.generation_min_score_threshold,
    )

    gen = Generator(settings=settings)

    if as_json or not stream:
        result = gen.ask(
            query=query, user=user,
            retrieval_config=rcfg, generation_config=gcfg,
        )
        if as_json:
            click.echo(result.model_dump_json(indent=2))
            if result.refused:
                sys.exit(2)
            return
        _render_ask_header(result, user)
        click.echo(result.answer)
        click.echo("")
        _render_ask_footer(result)
        if result.refused:
            sys.exit(2)
        return

    final_result = None
    started = False
    for ev in gen.ask_stream(
        query=query, user=user,
        retrieval_config=rcfg, generation_config=gcfg,
    ):
        if ev.kind == "start":
            assert ev.result is not None
            _render_ask_header(ev.result, user)
            started = True
        elif ev.kind == "token":
            click.echo(ev.text, nl=False)
        elif ev.kind == "refused":
            if not started and ev.result is not None:
                _render_ask_header(ev.result, user)
            click.echo(ev.text)
            final_result = ev.result
        elif ev.kind == "done":
            final_result = ev.result
            click.echo("")

    click.echo("")
    if final_result is not None:
        _render_ask_footer(final_result)
        if final_result.refused:
            sys.exit(2)


def _render_ask_header(result, user) -> None:
    click.echo("")
    click.echo(f"query: {result.query!r}")
    click.echo(
        f"user:  {user.user_id} (tenant={user.tenant_id or '∅'} "
        f"dept={user.department or '∅'} role={user.role})"
    )
    if result.retrieval is not None:
        r = result.retrieval
        click.echo(
            f"retrieval: collections={','.join(r.collections_searched)} "
            f"hits={len(r.hits)} fused={r.fused_candidates} "
            f"rerank={'on' if r.rerank_applied else 'off'}"
        )
    click.echo(
        f"generation: lane={result.lane or '-'} "
        f"context_tokens={result.context_tokens} "
        f"used_hits={result.used_hit_count}"
    )
    click.echo("─" * 60)


def _render_ask_footer(result) -> None:
    click.echo("─" * 60)
    if result.refused:
        click.echo(f"REFUSED: {result.refusal_reason}")
    if result.citations:
        click.echo("citations:")
        for c in result.citations:
            click.echo(
                f"  [{c.marker}] {c.title or '(no title)'}  "
                f"({c.source_id} · {c.section_path or '/'})"
            )
            click.echo(f"      {c.source_uri}")
    if result.invalid_markers:
        click.echo(
            f"WARNING: {len(result.invalid_markers)} invalid citation marker(s): "
            f"{result.invalid_markers}"
        )
    if result.uncited_hits:
        click.echo(f"uncited context blocks: {result.uncited_hits}")
    click.echo(
        f"timing_ms: generation={result.generation_ms} total={result.total_ms} "
        f"answer_chars={result.answer_chars} "
        f"provider={result.provider or '-'} model={result.model or '-'}"
    )


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
