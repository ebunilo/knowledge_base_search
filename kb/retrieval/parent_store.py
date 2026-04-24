"""
Parent store — bulk Postgres reads for small-to-big retrieval.

After fusion we have a set of child-level candidates. Each child points at
a parent (stored in `kb.parent_chunk`) and a document (stored in
`kb.document`). To build the final context we fetch:

    * The child's own content + summary + hypothetical_questions
      (useful for citation rendering and explainability).
    * The parent's content + section_path.
    * The document's title + source_uri + ACL fields (already present in
      Qdrant / BM25 payloads, but Postgres is the authoritative store).

One round trip per call using ANY(%s) so the DB can plan it as an index scan.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row

from kb.settings import Settings, get_settings


logger = logging.getLogger(__name__)


@dataclass
class ChildRecord:
    child_id: str
    parent_id: str
    document_id: str
    content: str
    summary: str | None
    hypothetical_questions: list[str]


@dataclass
class ParentRecord:
    parent_id: str
    document_id: str
    content: str
    section_path: str | None
    token_count: int | None


@dataclass
class DocumentRecord:
    document_id: str
    source_id: str
    source_uri: str
    title: str | None
    visibility: str
    sensitivity_lane: str
    acl: dict[str, Any]


class ParentStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._dsn = _build_dsn(self.settings)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fetch_children(self, child_ids: list[str]) -> dict[str, ChildRecord]:
        if not child_ids:
            return {}
        query = """
            SELECT child_id, parent_id, document_id, content, summary,
                   hypothetical_questions
            FROM kb.child_chunk
            WHERE child_id = ANY(%s)
        """
        with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, (child_ids,))
            rows = cur.fetchall()
        return {
            r["child_id"]: ChildRecord(
                child_id=r["child_id"],
                parent_id=r["parent_id"],
                document_id=r["document_id"],
                content=r["content"],
                summary=r.get("summary"),
                hypothetical_questions=r.get("hypothetical_questions") or [],
            )
            for r in rows
        }

    def fetch_parents(self, parent_ids: list[str]) -> dict[str, ParentRecord]:
        if not parent_ids:
            return {}
        query = """
            SELECT parent_id, document_id, content, section_path, token_count
            FROM kb.parent_chunk
            WHERE parent_id = ANY(%s)
        """
        with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, (parent_ids,))
            rows = cur.fetchall()
        return {
            r["parent_id"]: ParentRecord(
                parent_id=r["parent_id"],
                document_id=r["document_id"],
                content=r["content"],
                section_path=r.get("section_path"),
                token_count=r.get("token_count"),
            )
            for r in rows
        }

    def fetch_documents(self, document_ids: list[str]) -> dict[str, DocumentRecord]:
        if not document_ids:
            return {}
        query = """
            SELECT document_id, source_id, source_uri, title,
                   visibility, sensitivity_lane, acl
            FROM kb.document
            WHERE document_id = ANY(%s)
        """
        with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, (document_ids,))
            rows = cur.fetchall()
        return {
            r["document_id"]: DocumentRecord(
                document_id=r["document_id"],
                source_id=r["source_id"],
                source_uri=r["source_uri"],
                title=r.get("title"),
                visibility=r["visibility"],
                sensitivity_lane=r["sensitivity_lane"],
                acl=r.get("acl") or {},
            )
            for r in rows
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @contextmanager
    def _conn(self):
        conn = psycopg.connect(self._dsn, autocommit=True)
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
