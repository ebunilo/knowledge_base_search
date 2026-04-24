"""
Ingestion pipeline (Phase-1 slice).

This is the glue that runs:
    connector → parser → chunker → (next session: enrich → embed → index)

For now the pipeline stops at chunking and either:
    * returns the ChunkedDocument objects in memory, or
    * writes them as JSONL to a directory for inspection.

Enrichment, embeddings, and multi-index writers are wired in the next
session — at that point we'll add:
    * RecordManager.upsert(chunked_doc.children) for idempotent indexing
    * Qdrant, Postgres, and BM25 writers behind a single MultiIndexWriter
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

from kb.chunking import chunk_document
from kb.classifier.sensitivity import SensitivityClassifier
from kb.connectors import get_connector
from kb.parsers import ParserError, parse_document
from kb.types import ChunkedDocument, IngestStats


logger = logging.getLogger(__name__)


def load_source_inventory(path: str | Path = "data/source_inventory.json") -> list[dict]:
    """Load source definitions from the inventory JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Two accepted shapes:
        #   {"sources": [ {...}, {...} ]}
        #   { "src_internal_docs": {...}, "src_public_kb": {...} }
        if "sources" in data and isinstance(data["sources"], list):
            return data["sources"]
        return [
            {"source_id": k, **v}
            for k, v in data.items()
            if isinstance(v, dict)
        ]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected shape in {path}: {type(data).__name__}")


def get_source(source_id: str, inventory: list[dict]) -> dict:
    for s in inventory:
        if s.get("source_id") == source_id:
            return s
    raise KeyError(f"source_id {source_id!r} not found in inventory")


# --------------------------------------------------------------------------- #
# Ingestion entry point
# --------------------------------------------------------------------------- #

def ingest_source(
    source_id: str,
    *,
    inventory_path: str | Path = "data/source_inventory.json",
    output_dir: str | Path | None = None,
    on_chunked: Callable[[ChunkedDocument], None] | None = None,
    limit: int | None = None,
) -> IngestStats:
    """
    Run the Phase-1 ingestion slice for a single source.

    Args:
        source_id: identifier matching an entry in source_inventory.json.
        inventory_path: path to the inventory JSON.
        output_dir: if given, write each ChunkedDocument as JSONL to this dir.
        on_chunked: optional callback invoked for every successful chunk.
        limit: stop after this many documents (debug helper).

    Returns:
        IngestStats for the run.
    """
    inventory = load_source_inventory(inventory_path)
    source = get_source(source_id, inventory)
    connector = get_connector(source_id, source)

    classifier = SensitivityClassifier()
    stats = IngestStats(source_id=source_id)
    started = time.monotonic()

    out_path: Path | None = None
    out_file = None
    if output_dir:
        out_path = Path(output_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = (out_path / f"{source_id}.jsonl").open("w", encoding="utf-8")

    try:
        for raw in connector.iter_documents():
            stats.documents_seen += 1
            if limit is not None and stats.documents_processed >= limit:
                break

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

            try:
                chunked = chunk_document(parsed, classifier=classifier)
            except Exception as exc:  # noqa: BLE001
                stats.documents_failed += 1
                msg = f"chunk_failed doc={raw.document_id}: {exc}"
                logger.exception(msg)
                stats.errors.append(msg)
                continue

            stats.documents_processed += 1
            stats.parents_written += len(chunked.parents)
            stats.children_written += len(chunked.children)

            if on_chunked is not None:
                on_chunked(chunked)

            if out_file is not None:
                out_file.write(
                    chunked.model_dump_json(exclude={"metadata": {"size_bytes"}}) + "\n"
                )
    finally:
        if out_file is not None:
            out_file.close()

    stats.elapsed_s = round(time.monotonic() - started, 3)
    logger.info(
        "ingest_source done source=%s seen=%d processed=%d failed=%d "
        "parents=%d children=%d elapsed=%.2fs",
        source_id, stats.documents_seen, stats.documents_processed,
        stats.documents_failed, stats.parents_written, stats.children_written,
        stats.elapsed_s,
    )
    return stats
