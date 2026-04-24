"""
Ingestion pipeline.

Stages (each strictly after the previous):
    chunk   — connector → parse → chunk
    enrich  — + hypothetical questions + summaries
    embed   — + content vectors + question vectors
    index   — + write to Postgres + Qdrant + BM25 (idempotent)

The default stage for the CLI is `chunk` (fast, no network). `index` runs
the full pipeline and requires the data plane to be reachable.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Literal

from kb.chunking import chunk_document
from kb.classifier.sensitivity import SensitivityClassifier
from kb.connectors import get_connector
from kb.embeddings import EmbeddingClient, embed_document
from kb.enrichment import Enricher
from kb.indexing import MultiIndexWriter, RecordManager
from kb.indexing.record_manager import RecordAction
from kb.parsers import ParserError, parse_document
from kb.types import (
    ChunkedDocument,
    EmbeddedDocument,
    EnrichedDocument,
    IngestStats,
)


logger = logging.getLogger(__name__)


Stage = Literal["chunk", "enrich", "embed", "index"]
_STAGE_ORDER: list[Stage] = ["chunk", "enrich", "embed", "index"]


def load_source_inventory(path: str | Path = "data/source_inventory.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "sources" in data and isinstance(data["sources"], list):
            return data["sources"]
        return [{"source_id": k, **v} for k, v in data.items() if isinstance(v, dict)]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected shape in {path}: {type(data).__name__}")


def get_source(source_id: str, inventory: list[dict]) -> dict:
    for s in inventory:
        if s.get("source_id") == source_id:
            return s
    raise KeyError(f"source_id {source_id!r} not found in inventory")


def _stage_reaches(stage: Stage, target: Stage) -> bool:
    return _STAGE_ORDER.index(stage) <= _STAGE_ORDER.index(target)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def ingest_source(
    source_id: str,
    *,
    stage: Stage = "chunk",
    inventory_path: str | Path = "data/source_inventory.json",
    output_dir: str | Path | None = None,
    limit: int | None = None,
    on_chunked: Callable[[ChunkedDocument], None] | None = None,
    on_enriched: Callable[[EnrichedDocument], None] | None = None,
    on_embedded: Callable[[EmbeddedDocument], None] | None = None,
    force_reindex: bool = False,
    enricher: Enricher | None = None,
    embedder: EmbeddingClient | None = None,
    writer: MultiIndexWriter | None = None,
) -> IngestStats:
    """
    Run the ingestion pipeline for one source up to `stage`.

    If `stage == "index"`, the writer's record manager short-circuits
    unchanged documents unless `force_reindex=True`.
    """
    if stage not in _STAGE_ORDER:
        raise ValueError(f"stage must be one of {_STAGE_ORDER}, got {stage!r}")

    inventory = load_source_inventory(inventory_path)
    source = get_source(source_id, inventory)
    connector = get_connector(source_id, source)
    classifier = SensitivityClassifier()

    # Lazy-init expensive clients only if their stage is reached.
    if _stage_reaches("enrich", stage) and enricher is None:
        enricher = Enricher()
    if _stage_reaches("embed", stage) and embedder is None:
        embedder = EmbeddingClient()
    if _stage_reaches("index", stage) and writer is None:
        writer = MultiIndexWriter()

    record_manager = writer.record_manager if writer else None

    stats = IngestStats(source_id=source_id)
    started = time.monotonic()

    out_path: Path | None = None
    out_file = None
    if output_dir:
        out_path = Path(output_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = (out_path / f"{source_id}.{stage}.jsonl").open("w", encoding="utf-8")

    try:
        for raw in connector.iter_documents():
            stats.documents_seen += 1
            if limit is not None and stats.documents_processed >= limit:
                break

            # ---------- short-circuit on unchanged docs when indexing ----------
            if record_manager is not None and not force_reindex:
                decision = record_manager.decide(
                    source_id=source_id,
                    document_id=raw.document_id,
                    content_hash=raw.content_hash,
                )
                if decision.action == RecordAction.SKIP:
                    stats.documents_skipped_unchanged += 1
                    continue

            # ---------- parse ----------
            try:
                parsed = parse_document(raw)
            except ParserError as exc:
                stats.documents_failed += 1
                msg = f"parse_failed doc={raw.document_id} uri={raw.source_uri}: {exc}"
                logger.warning(msg)
                stats.errors.append(msg)
                continue
            except Exception as exc:  # noqa: BLE001
                stats.documents_failed += 1
                msg = f"parse_crashed doc={raw.document_id} uri={raw.source_uri}: {exc}"
                logger.exception(msg)
                stats.errors.append(msg)
                continue

            # ---------- chunk ----------
            try:
                chunked = chunk_document(parsed, classifier=classifier)
            except Exception as exc:  # noqa: BLE001
                stats.documents_failed += 1
                msg = f"chunk_failed doc={raw.document_id}: {exc}"
                logger.exception(msg)
                stats.errors.append(msg)
                continue

            stats.parents_written += len(chunked.parents)
            stats.children_written += len(chunked.children)

            if on_chunked is not None:
                on_chunked(chunked)

            current_doc: ChunkedDocument | EnrichedDocument | EmbeddedDocument = chunked

            # ---------- enrich ----------
            if _stage_reaches("enrich", stage):
                try:
                    current_doc = enricher.enrich(chunked)  # type: ignore[union-attr]
                except Exception as exc:  # noqa: BLE001
                    stats.documents_failed += 1
                    msg = f"enrich_failed doc={raw.document_id}: {exc}"
                    logger.exception(msg)
                    stats.errors.append(msg)
                    continue
                if on_enriched is not None:
                    on_enriched(current_doc)  # type: ignore[arg-type]

            # ---------- embed ----------
            if _stage_reaches("embed", stage):
                try:
                    current_doc = embed_document(current_doc, client=embedder)  # type: ignore[arg-type]
                except Exception as exc:  # noqa: BLE001
                    stats.documents_failed += 1
                    msg = f"embed_failed doc={raw.document_id}: {exc}"
                    logger.exception(msg)
                    stats.errors.append(msg)
                    continue
                stats.vectors_written += sum(
                    int(bool(c.content_vector)) + len(c.question_vectors or [])
                    for c in current_doc.children  # type: ignore[attr-defined]
                )
                if on_embedded is not None:
                    on_embedded(current_doc)  # type: ignore[arg-type]

            # ---------- index ----------
            if _stage_reaches("index", stage):
                try:
                    wrote = writer.upsert(current_doc)  # type: ignore[arg-type]
                    if not wrote:
                        stats.documents_skipped_unchanged += 1
                        continue
                except Exception as exc:  # noqa: BLE001
                    stats.documents_failed += 1
                    msg = f"index_failed doc={raw.document_id}: {exc}"
                    logger.exception(msg)
                    stats.errors.append(msg)
                    continue

            stats.documents_processed += 1

            if out_file is not None:
                payload = current_doc.model_dump(exclude={"children": {"__all__": {"content_vector", "question_vectors"}}})
                out_file.write(json.dumps(payload, default=str) + "\n")
    finally:
        if out_file is not None:
            out_file.close()

    stats.elapsed_s = round(time.monotonic() - started, 3)
    logger.info(
        "ingest_source done source=%s stage=%s seen=%d processed=%d "
        "skipped=%d failed=%d parents=%d children=%d vectors=%d elapsed=%.2fs",
        source_id, stage, stats.documents_seen, stats.documents_processed,
        stats.documents_skipped_unchanged, stats.documents_failed,
        stats.parents_written, stats.children_written, stats.vectors_written,
        stats.elapsed_s,
    )

    if writer is not None:
        logger.info(
            "writer stats: pg_parents=%d pg_children=%d qdrant_points=%d "
            "bm25_children=%d skip=%d replace=%d write=%d rollbacks=%d",
            writer.stats.postgres_parents, writer.stats.postgres_children,
            writer.stats.qdrant_points, writer.stats.bm25_children,
            writer.stats.decided_skip, writer.stats.decided_replace,
            writer.stats.decided_write, writer.stats.rollbacks,
        )

    return stats
