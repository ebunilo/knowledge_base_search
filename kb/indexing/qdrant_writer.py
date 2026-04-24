"""
Qdrant writer — REST client.

Why REST (not qdrant-client): the official Python client hard-depends on
grpcio-tools which doesn't build on our target Python/macOS combination
(see scripts/smoke_test.py comments). REST is one HTTP call per operation,
simple to test, and plenty fast for ingestion throughput.

Per-document upsert flow:
    1. Ensure the collection exists with the expected vector dim.
    2. Delete any existing points where payload.document_id == doc.document_id.
    3. Build one point per vector:
         * content vector → payload.kind = "content"
         * each question vector → payload.kind = "question", question_ord
       All points share the (child_id, parent_id, document_id) triple so the
       retriever can group them back to the underlying child.
    4. POST to /collections/<name>/points.

Routing by sensitivity:
    * hosted_ok        → public collection   (APP_PUBLIC_COLLECTION)
    * self_hosted_only → private collection  (APP_PRIVATE_COLLECTION)

ACL payload shape (for retrieval-time filtering):
    acl_departments: ["engineering", "operations", ...]   or ["*"]
    acl_min_role:    "anonymous" | "employee" | "manager" | ...
    visibility:      "public" | "internal" | "restricted"
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import requests

from kb.settings import Settings, get_settings
from kb.types import EmbeddedChildChunk, EmbeddedDocument, SensitivityLane


logger = logging.getLogger(__name__)


class QdrantWriterError(Exception):
    """Raised on non-recoverable Qdrant failures."""


class QdrantWriter:
    TIMEOUT_S = 30
    MAX_BATCH_POINTS = 256

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._url = self.settings.qdrant_url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if self.settings.qdrant_api_key:
            self._headers["api-key"] = self.settings.qdrant_api_key
        self._ensured: set[str] = set()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def collection_for(self, lane: SensitivityLane) -> str:
        if lane == SensitivityLane.HOSTED_OK:
            return self.settings.app_public_collection
        return self.settings.app_private_collection

    def upsert_document(self, doc: EmbeddedDocument) -> int:
        """Upsert all vectors for this document. Returns count of points written."""
        collection = self.collection_for(doc.sensitivity)
        self._ensure_collection(collection, dim=doc.embed_dim)

        self.delete_document(collection, doc.document_id)

        points = list(self._build_points(doc))
        if not points:
            return 0

        for batch in _batched(points, self.MAX_BATCH_POINTS):
            self._put_points(collection, batch)
        return len(points)

    def delete_document(self, collection: str, document_id: str) -> None:
        """Delete all points where payload.document_id == document_id."""
        body = {
            "filter": {
                "must": [
                    {"key": "document_id", "match": {"value": document_id}}
                ]
            }
        }
        url = f"{self._url}/collections/{collection}/points/delete?wait=true"
        r = requests.post(url, headers=self._headers, json=body, timeout=self.TIMEOUT_S)
        if r.status_code == 404:
            return  # collection doesn't exist yet — nothing to delete
        if not r.ok:
            raise QdrantWriterError(
                f"Qdrant delete failed ({r.status_code}): {r.text[:300]}"
            )

    def health(self) -> bool:
        try:
            r = requests.get(
                f"{self._url}/collections",
                headers=self._headers, timeout=self.TIMEOUT_S,
            )
            return r.ok
        except Exception as exc:  # noqa: BLE001
            logger.warning("qdrant health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _ensure_collection(self, collection: str, *, dim: int) -> None:
        if collection in self._ensured:
            return

        r = requests.get(
            f"{self._url}/collections/{collection}",
            headers=self._headers, timeout=self.TIMEOUT_S,
        )
        if r.status_code == 200:
            self._ensured.add(collection)
            return
        if r.status_code != 404:
            raise QdrantWriterError(
                f"Qdrant collection probe failed ({r.status_code}): {r.text[:200]}"
            )

        # Create with cosine distance; bge-m3 is L2-normalised so cosine
        # equals dot-product, and cosine is the standard choice.
        body = {
            "vectors": {
                "size": dim,
                "distance": "Cosine",
            }
        }
        r = requests.put(
            f"{self._url}/collections/{collection}",
            headers=self._headers, json=body, timeout=self.TIMEOUT_S,
        )
        if not r.ok:
            raise QdrantWriterError(
                f"Qdrant create collection failed ({r.status_code}): {r.text[:300]}"
            )
        # Payload indexes — speed up document_id deletes and ACL filters.
        for field, schema in (
            ("document_id", "keyword"),
            ("source_id", "keyword"),
            ("visibility", "keyword"),
            ("sensitivity", "keyword"),
            ("acl_min_role", "keyword"),
            ("acl_departments", "keyword"),
        ):
            requests.put(
                f"{self._url}/collections/{collection}/index",
                headers=self._headers,
                json={"field_name": field, "field_schema": schema},
                timeout=self.TIMEOUT_S,
            )
        self._ensured.add(collection)

    def _build_points(self, doc: EmbeddedDocument):
        base_payload = self._doc_payload(doc)
        for child in doc.children:
            if child.content_vector:
                yield {
                    "id": _point_id(child.child_id, "content", 0),
                    "vector": child.content_vector,
                    "payload": {
                        **base_payload,
                        **self._child_payload(child),
                        "kind": "content",
                    },
                }
            for q_idx, q_vec in enumerate(child.question_vectors or []):
                yield {
                    "id": _point_id(child.child_id, "q", q_idx),
                    "vector": q_vec,
                    "payload": {
                        **base_payload,
                        **self._child_payload(child),
                        "kind": "question",
                        "question_ord": q_idx,
                        "question_text": (
                            child.hypothetical_questions[q_idx]
                            if q_idx < len(child.hypothetical_questions)
                            else None
                        ),
                    },
                }

    def _put_points(self, collection: str, points: list[dict[str, Any]]) -> None:
        url = f"{self._url}/collections/{collection}/points?wait=true"
        r = requests.put(
            url, headers=self._headers, json={"points": points}, timeout=self.TIMEOUT_S,
        )
        if not r.ok:
            raise QdrantWriterError(
                f"Qdrant upsert failed ({r.status_code}): {r.text[:300]}"
            )

    @staticmethod
    def _doc_payload(doc: EmbeddedDocument) -> dict[str, Any]:
        return {
            "document_id": doc.document_id,
            "source_id": doc.source_id,
            "source_uri": doc.source_uri,
            "title": doc.title,
            "visibility": doc.visibility.value,
            "sensitivity": doc.sensitivity.value,
            "language": doc.language,
            "acl_departments": doc.acl.departments,
            "acl_min_role": doc.acl.min_role,
            "acl_tags": doc.acl.tags,
        }

    @staticmethod
    def _child_payload(child: EmbeddedChildChunk) -> dict[str, Any]:
        return {
            "child_id": child.child_id,
            "parent_id": child.parent_id,
            "ord": child.ord,
            "section_path": " > ".join(child.section_path) if child.section_path else "",
            "token_count": child.token_count,
            "summary": child.summary,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _point_id(child_id: str, kind: str, q_idx: int) -> str:
    """
    Qdrant requires UUIDs or unsigned ints as point IDs. We derive a stable
    UUID-v5 from (child_id, kind, q_idx) so re-ingestion of the same chunk
    overwrites the same point deterministically.
    """
    seed = f"{child_id}|{kind}|{q_idx}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def _batched(seq, n):
    buf = []
    for item in seq:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
