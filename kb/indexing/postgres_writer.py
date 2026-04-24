"""
Postgres writer — authoritative doc store for parent-child retrieval.

Writes to:
    kb.document        — one row per document (ACL, sensitivity, metadata)
    kb.parent_chunk    — one row per parent (section-sized)
    kb.child_chunk     — one row per child (question-vector text lives here)

All writes happen inside a single transaction per document so a crash can't
leave partial state. A replacement (re-ingestion of a changed doc) does:
    DELETE FROM kb.document WHERE document_id = ?
    (ON DELETE CASCADE drops parents + children)
    INSERT INTO kb.document ...
    INSERT INTO kb.parent_chunk ...
    INSERT INTO kb.child_chunk ...

Schema comes from docker/postgres/init.sql.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager

import psycopg
from psycopg import sql

from kb.settings import Settings, get_settings
from kb.types import EmbeddedDocument


logger = logging.getLogger(__name__)


class PostgresWriter:
    """Connection-per-call writer. Safe to share across threads."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._dsn = _build_dsn(self.settings)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def upsert_document(self, doc: EmbeddedDocument) -> tuple[int, int]:
        """
        Replace the document record and all its parents/children.
        Returns (parents_written, children_written).
        """
        acl_json = json.dumps(doc.acl.model_dump())
        metadata_json = json.dumps(doc.metadata or {})

        with self._conn() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    # CASCADE clears any existing parent/child rows for this doc
                    cur.execute(
                        "DELETE FROM kb.document WHERE document_id = %s",
                        (doc.document_id,),
                    )

                    cur.execute(
                        """
                        INSERT INTO kb.document (
                            document_id, source_id, source_uri, title,
                            content_hash, language, region,
                            sensitivity_lane, visibility, acl, metadata,
                            updated_at
                        )
                        VALUES (
                            %s, %s, %s, %s,
                            %s, %s, %s,
                            %s, %s, %s::jsonb, %s::jsonb,
                            now()
                        )
                        """,
                        (
                            doc.document_id,
                            doc.source_id,
                            doc.source_uri,
                            doc.title,
                            doc.content_hash,
                            doc.language,
                            doc.metadata.get("region") if doc.metadata else None,
                            doc.sensitivity.value,
                            doc.visibility.value,
                            acl_json,
                            metadata_json,
                        ),
                    )

                    parent_rows = [
                        (
                            p.parent_id,
                            doc.document_id,
                            p.ord,
                            p.content,
                            p.token_count,
                            " > ".join(p.section_path) if p.section_path else None,
                            json.dumps(p.metadata or {}),
                        )
                        for p in doc.parents
                    ]
                    if parent_rows:
                        cur.executemany(
                            """
                            INSERT INTO kb.parent_chunk (
                                parent_id, document_id, ord, content,
                                token_count, section_path, metadata
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                            """,
                            parent_rows,
                        )

                    child_rows = [
                        (
                            c.child_id,
                            c.parent_id,
                            doc.document_id,
                            c.ord,
                            c.content,
                            c.token_count,
                            json.dumps(c.hypothetical_questions or []),
                            c.summary,
                            json.dumps(c.metadata or {}),
                        )
                        for c in doc.children
                    ]
                    if child_rows:
                        cur.executemany(
                            """
                            INSERT INTO kb.child_chunk (
                                child_id, parent_id, document_id, ord,
                                content, token_count,
                                hypothetical_questions, summary, metadata
                            )
                            VALUES (
                                %s, %s, %s, %s,
                                %s, %s,
                                %s::jsonb, %s, %s::jsonb
                            )
                            """,
                            child_rows,
                        )

        return len(doc.parents), len(doc.children)

    def delete_document(self, document_id: str) -> int:
        """Remove a document and its chunks. Returns rowcount on kb.document."""
        with self._conn() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM kb.document WHERE document_id = %s",
                        (document_id,),
                    )
                    return cur.rowcount

    def health(self) -> bool:
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("postgres health check failed: %s", exc)
            return False

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
