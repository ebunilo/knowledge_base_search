"""
MultiIndexWriter — Qdrant + Postgres + BM25, behind one idempotent API.

Write order is chosen so that on partial failure the system is left in the
least-confusing state:
    1. Decide (record manager) — SKIP fast for unchanged docs.
    2. Postgres write — authoritative doc store; transactional. If this
       fails, nothing else is touched.
    3. Qdrant write — if this fails after Postgres succeeded, roll back by
       deleting the Postgres rows and re-raising. Retrieval is then
       exactly as it was before this document.
    4. BM25 write — same rollback strategy if it fails after Qdrant.
    5. Record manager.mark_indexed — only after all three succeed.

The rollback is best-effort; if the rollback itself fails we log loudly and
leave the record manager unchanged (so a subsequent run will re-attempt).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from kb.indexing.bm25_writer import BM25Writer
from kb.indexing.postgres_writer import PostgresWriter
from kb.indexing.qdrant_writer import QdrantWriter
from kb.indexing.record_manager import RecordAction, RecordManager
from kb.types import EmbeddedDocument, SensitivityLane


logger = logging.getLogger(__name__)


@dataclass
class WriteStats:
    decided_write: int = 0
    decided_skip: int = 0
    decided_replace: int = 0
    postgres_parents: int = 0
    postgres_children: int = 0
    qdrant_points: int = 0
    bm25_children: int = 0
    rollbacks: int = 0
    errors: list[str] = field(default_factory=list)


class MultiIndexWriter:
    def __init__(
        self,
        *,
        postgres: PostgresWriter | None = None,
        qdrant: QdrantWriter | None = None,
        bm25: BM25Writer | None = None,
        record_manager: RecordManager | None = None,
    ) -> None:
        self.postgres = postgres or PostgresWriter()
        self.qdrant = qdrant or QdrantWriter()
        self.bm25 = bm25 or BM25Writer()
        self.record_manager = record_manager or RecordManager()
        self.stats = WriteStats()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def upsert(self, doc: EmbeddedDocument) -> bool:
        """
        Upsert one EmbeddedDocument. Returns True if work was performed,
        False if the document was skipped as unchanged.
        """
        decision = self.record_manager.decide(
            source_id=doc.source_id,
            document_id=doc.document_id,
            content_hash=doc.content_hash,
        )

        if decision.action == RecordAction.SKIP:
            self.stats.decided_skip += 1
            return False

        if decision.action == RecordAction.REPLACE:
            self.stats.decided_replace += 1
        else:
            self.stats.decided_write += 1

        collection = self.qdrant.collection_for(doc.sensitivity)

        # --- Stage 1: Postgres ------------------------------------------------
        try:
            parents, children = self.postgres.upsert_document(doc)
            self.stats.postgres_parents += parents
            self.stats.postgres_children += children
        except Exception as exc:
            self.stats.errors.append(f"postgres_failed:{doc.document_id}:{exc}")
            raise

        # --- Stage 2: Qdrant --------------------------------------------------
        try:
            points = self.qdrant.upsert_document(doc)
            self.stats.qdrant_points += points
        except Exception as exc:
            logger.exception("qdrant write failed for %s — rolling back postgres", doc.document_id)
            self._rollback_postgres(doc.document_id)
            self.stats.errors.append(f"qdrant_failed:{doc.document_id}:{exc}")
            raise

        # --- Stage 3: BM25 ----------------------------------------------------
        try:
            written = self.bm25.upsert_document(collection, doc)
            self.stats.bm25_children += written
        except Exception as exc:
            logger.exception("bm25 write failed for %s — rolling back pg + qdrant", doc.document_id)
            self._rollback_postgres(doc.document_id)
            self._rollback_qdrant(collection, doc.document_id)
            self.stats.errors.append(f"bm25_failed:{doc.document_id}:{exc}")
            raise

        # --- Stage 4: record manager -----------------------------------------
        self.record_manager.mark_indexed(
            source_id=doc.source_id,
            document_id=doc.document_id,
            content_hash=doc.content_hash,
        )
        return True

    def delete(self, *, source_id: str, document_id: str, lane: SensitivityLane) -> None:
        """Remove a document from every index."""
        collection = self.qdrant.collection_for(lane)
        try:
            self.postgres.delete_document(document_id)
        except Exception:
            logger.exception("postgres delete failed for %s", document_id)
        try:
            self.qdrant.delete_document(collection, document_id)
        except Exception:
            logger.exception("qdrant delete failed for %s", document_id)
        try:
            self.bm25.delete_document(collection, document_id)
        except Exception:
            logger.exception("bm25 delete failed for %s", document_id)
        self.record_manager.forget(source_id=source_id, document_id=document_id)

    # ------------------------------------------------------------------ #
    # Rollback helpers (best-effort)
    # ------------------------------------------------------------------ #

    def _rollback_postgres(self, document_id: str) -> None:
        try:
            self.postgres.delete_document(document_id)
            self.stats.rollbacks += 1
        except Exception:
            logger.error("rollback-postgres failed for %s", document_id, exc_info=True)

    def _rollback_qdrant(self, collection: str, document_id: str) -> None:
        try:
            self.qdrant.delete_document(collection, document_id)
            self.stats.rollbacks += 1
        except Exception:
            logger.error("rollback-qdrant failed for %s", document_id, exc_info=True)
