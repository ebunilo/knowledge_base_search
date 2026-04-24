"""
Record manager — idempotency gate for incremental indexing.

Backed by the `kb.upsertion_record` table created in init.sql. We use the
same shape LangChain's SQLRecordManager uses, so a future swap to LC's
incremental-index helper is a drop-in change.

Keys:
    namespace = source_id
    key       = document_id
    group_id  = content_hash

Decision table:
    no row                           → WRITE      (new document)
    row with same group_id           → SKIP       (unchanged, not re-indexed)
    row with different group_id      → REPLACE    (content changed, re-index)
    row that should be gone          → DELETE     (document removed upstream)

The manager itself never touches Qdrant / BM25 / document tables — it only
records what has been written. The MultiIndexWriter consults it before
doing work (short-circuit on SKIP) and updates it after a successful write.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import psycopg

from kb.settings import Settings, get_settings


logger = logging.getLogger(__name__)


class RecordAction(str, Enum):
    WRITE = "write"
    SKIP = "skip"
    REPLACE = "replace"


@dataclass
class RecordDecision:
    action: RecordAction
    previous_group_id: str | None = None


class RecordManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._dsn = _build_dsn(self.settings)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def decide(self, *, source_id: str, document_id: str, content_hash: str) -> RecordDecision:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT group_id
                FROM kb.upsertion_record
                WHERE namespace = %s AND key = %s
                """,
                (source_id, document_id),
            )
            row = cur.fetchone()

        if row is None:
            return RecordDecision(action=RecordAction.WRITE)

        prev_group_id = row[0]
        if prev_group_id == content_hash:
            return RecordDecision(action=RecordAction.SKIP, previous_group_id=prev_group_id)
        return RecordDecision(action=RecordAction.REPLACE, previous_group_id=prev_group_id)

    def mark_indexed(self, *, source_id: str, document_id: str, content_hash: str) -> None:
        """Record a successful (re-)indexing."""
        record_uuid = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"kb-upsert|{source_id}|{document_id}",
            )
        )
        with self._conn() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO kb.upsertion_record
                            (uuid, key, namespace, group_id, updated_at)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (namespace, key) DO UPDATE SET
                            group_id = EXCLUDED.group_id,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (
                            record_uuid,
                            document_id,
                            source_id,
                            content_hash,
                            time.time(),
                        ),
                    )

    def forget(self, *, source_id: str, document_id: str) -> None:
        """Record a deletion; MultiIndexWriter.delete() calls this after cleanup."""
        with self._conn() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        DELETE FROM kb.upsertion_record
                        WHERE namespace = %s AND key = %s
                        """,
                        (source_id, document_id),
                    )

    def known_document_ids(self, source_id: str) -> set[str]:
        """All doc ids we've previously indexed for this source."""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT key FROM kb.upsertion_record WHERE namespace = %s",
                (source_id,),
            )
            return {r[0] for r in cur.fetchall()}

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @contextmanager
    def _conn(self):
        conn = psycopg.connect(self._dsn, autocommit=False)
        try:
            yield conn
        finally:
            conn.close()


def _build_dsn(s: Settings) -> str:
    if s.postgres_url and s.postgres_url.startswith(("postgres://", "postgresql://")):
        return s.postgres_url
    return (
        f"host={s.postgres_host} port={s.postgres_port} "
        f"user={s.postgres_user} password={s.postgres_password} dbname={s.postgres_db}"
    )
