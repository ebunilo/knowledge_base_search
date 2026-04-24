"""
In-process BM25 writer.

We keep per-collection indices on disk (pickled) under `data/bm25_index/`.
On ingestion we:
    1. Load the collection's token store (child_id → tokens).
    2. Remove any entries for the current document (supports re-ingestion).
    3. Add the new child chunks.
    4. Rebuild BM25Okapi from the token store.
    5. Persist atomically (write → fsync → rename).

For Phase-1 demo volumes this is plenty. For prod we'd swap to OpenSearch;
the `bm25_backend` setting is already wired in settings.py.

Thread-safety: a per-collection lock is acquired for the critical section
(load → mutate → save) so a concurrent ingest cannot corrupt the index.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from kb.types import EmbeddedDocument


logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "for",
    "to", "from", "on", "in", "with", "at", "by", "is", "are", "was", "were",
    "be", "been", "being", "it", "this", "that", "these", "those",
})


@dataclass
class _Store:
    """On-disk shape for one collection's BM25 data."""
    # child_id -> tokenized text (content + questions concatenated)
    tokens: dict[str, list[str]] = field(default_factory=dict)
    # child_id -> {document_id, parent_id, source_id, section_path, acl, ...}
    metadata: dict[str, dict] = field(default_factory=dict)
    # document_id -> list[child_id] (for fast delete-by-document)
    doc_to_children: dict[str, list[str]] = field(default_factory=dict)


class BM25Writer:
    """Write / delete children in a persisted BM25 index, per collection."""

    def __init__(self, base_dir: str | Path = "data/bm25_index") -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def upsert_document(self, collection: str, doc: EmbeddedDocument) -> int:
        """Insert or replace all children for this document. Returns count written."""
        with self._lock_for(collection):
            store = self._load(collection)
            self._delete_document_inplace(store, doc.document_id)

            child_ids: list[str] = []
            for child in doc.children:
                tokens = _tokenize(
                    " ".join(
                        [
                            child.content,
                            child.summary or "",
                            *(child.hypothetical_questions or []),
                        ]
                    )
                )
                store.tokens[child.child_id] = tokens
                store.metadata[child.child_id] = {
                    "document_id": doc.document_id,
                    "parent_id": child.parent_id,
                    "source_id": doc.source_id,
                    "section_path": child.section_path,
                    "acl": doc.acl.model_dump(),
                    "visibility": doc.visibility.value,
                    "sensitivity": doc.sensitivity.value,
                }
                child_ids.append(child.child_id)
            store.doc_to_children[doc.document_id] = child_ids

            self._save(collection, store)
            return len(child_ids)

    def delete_document(self, collection: str, document_id: str) -> int:
        """Remove all entries for this document. Returns count removed."""
        with self._lock_for(collection):
            store = self._load(collection)
            removed = self._delete_document_inplace(store, document_id)
            if removed:
                self._save(collection, store)
            return removed

    def stats(self, collection: str) -> dict:
        """Quick counts for debug / smoke tests."""
        with self._lock_for(collection):
            store = self._load(collection)
            return {
                "children": len(store.tokens),
                "documents": len(store.doc_to_children),
                "path": str(self._path(collection)),
            }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _lock_for(self, collection: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(collection)
            if lock is None:
                lock = threading.Lock()
                self._locks[collection] = lock
        return lock

    def _path(self, collection: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", collection)
        return self.base_dir / f"{safe}.pkl"

    def _load(self, collection: str) -> _Store:
        path = self._path(collection)
        if not path.exists():
            return _Store()
        try:
            with path.open("rb") as f:
                store = pickle.load(f)
            if not isinstance(store, _Store):
                logger.warning("BM25 store at %s has wrong type; resetting", path)
                return _Store()
            return store
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to load BM25 store %s: %s — resetting", path, exc)
            return _Store()

    def _save(self, collection: str, store: _Store) -> None:
        path = self._path(collection)
        # Atomic write: tmp → fsync → rename
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=str(self.base_dir)
        )
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @staticmethod
    def _delete_document_inplace(store: _Store, document_id: str) -> int:
        kids = store.doc_to_children.pop(document_id, [])
        for cid in kids:
            store.tokens.pop(cid, None)
            store.metadata.pop(cid, None)
        return len(kids)


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #

def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens = (t.lower() for t in _TOKEN_RE.findall(text))
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]
