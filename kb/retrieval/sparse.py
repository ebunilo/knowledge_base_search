"""
Sparse retriever — BM25 over the pickled stores written during ingestion.

Responsibilities:
    * Lazily load `_Store` from `data/bm25_index/<collection>.pkl`.
    * Tokenize the query using the same tokenizer used at index time
      (stopword filter, lower-cased alphanumerics).
    * Score with BM25Okapi; filter by ACL using the same predicate the
      Qdrant side uses (defence in depth).
    * Return the top-K deduped by child_id (each child appears at most once
      since BM25 indexes a single document per child).

Caching:
    The BM25 index (BM25Okapi instance) is rebuilt once per store load and
    cached in memory keyed by the store's file mtime, so subsequent queries
    skip the rebuild until a new ingestion write happens.
"""

from __future__ import annotations

import logging
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kb.indexing.bm25_writer import _Store  # shared on-disk schema
from kb.retrieval.acl import hit_allowed
from kb.retrieval.types import MatchVia, UserContext
from kb.settings import Settings, get_settings


logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "for",
    "to", "from", "on", "in", "with", "at", "by", "is", "are", "was", "were",
    "be", "been", "being", "it", "this", "that", "these", "those",
})


@dataclass
class SparseRawHit:
    child_id: str
    score: float
    rank: int
    collection: str
    payload: dict[str, Any]
    matched: MatchVia


class SparseRetriever:
    def __init__(
        self,
        *,
        base_dir: str | Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_dir = Path(base_dir or "data/bm25_index").expanduser().resolve()
        # collection -> (mtime, Store, BM25Okapi, ordered_child_ids)
        self._cache: dict[str, tuple[float, _Store, Any, list[str]]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def search(
        self,
        *,
        query: str,
        collections: list[str],
        user: UserContext,
        top_k: int = 30,
        source_allowlist: list[str] | None = None,
    ) -> list[SparseRawHit]:
        tokens = _tokenize(query)
        if not tokens:
            return []

        all_hits: list[SparseRawHit] = []
        for collection in collections:
            hits = self._search_one(
                collection=collection,
                tokens=tokens,
                top_k=top_k,
                user=user,
                source_allowlist=source_allowlist,
            )
            all_hits.extend(hits)

        return self._dedupe(all_hits)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _search_one(
        self,
        *,
        collection: str,
        tokens: list[str],
        top_k: int,
        user: UserContext,
        source_allowlist: list[str] | None,
    ) -> list[SparseRawHit]:
        loaded = self._load(collection)
        if loaded is None:
            return []

        store, bm25, ordered_child_ids = loaded
        if not ordered_child_ids:
            return []

        t0 = time.monotonic()
        scores = bm25.get_scores(tokens)
        # Argsort descending, take a healthy multiplier of top_k to allow
        # ACL rejects without exhausting results.
        k = min(len(ordered_child_ids), top_k * 4)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        hits: list[SparseRawHit] = []
        rank = 0
        for i in idxs:
            s = float(scores[i])
            if s <= 0.0:
                break
            child_id = ordered_child_ids[i]
            payload = store.metadata.get(child_id) or {}

            if source_allowlist and payload.get("source_id") not in source_allowlist:
                continue
            if not hit_allowed(user, payload):
                continue

            rank += 1
            hits.append(
                SparseRawHit(
                    child_id=child_id,
                    score=s,
                    rank=rank,
                    collection=collection,
                    payload=payload,
                    matched=MatchVia(kind="sparse", score=s, rank=rank),
                )
            )
            if rank >= top_k:
                break

        logger.debug(
            "sparse search collection=%s returned=%d (in %.1fms)",
            collection, len(hits), (time.monotonic() - t0) * 1000,
        )
        return hits

    def _load(self, collection: str):
        """Return (store, bm25, ordered_child_ids) or None if missing."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            logger.error("rank-bm25 is not installed: %s", exc)
            return None

        path = self._path(collection)
        if not path.exists():
            logger.debug("no BM25 store for collection=%s at %s", collection, path)
            return None

        mtime = path.stat().st_mtime
        cached = self._cache.get(collection)
        if cached is not None and cached[0] == mtime:
            return cached[1], cached[2], cached[3]

        try:
            with path.open("rb") as f:
                store: _Store = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to load BM25 store %s: %s", path, exc)
            return None

        if not isinstance(store, _Store):
            logger.error("unexpected BM25 store type at %s", path)
            return None

        ordered_child_ids = list(store.tokens.keys())
        corpus = [store.tokens[cid] for cid in ordered_child_ids]
        bm25 = BM25Okapi(corpus) if corpus else None
        if bm25 is None:
            return None

        self._cache[collection] = (mtime, store, bm25, ordered_child_ids)
        return store, bm25, ordered_child_ids

    def _path(self, collection: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", collection)
        return self.base_dir / f"{safe}.pkl"

    @staticmethod
    def _dedupe(hits: list[SparseRawHit]) -> list[SparseRawHit]:
        best: dict[str, SparseRawHit] = {}
        for h in hits:
            current = best.get(h.child_id)
            if current is None or h.score > current.score:
                best[h.child_id] = h
        ordered = sorted(best.values(), key=lambda h: h.score, reverse=True)
        for rank, h in enumerate(ordered, start=1):
            h.rank = rank
            h.matched.rank = rank
        return ordered


# --------------------------------------------------------------------------- #
# Tokenization — MUST match kb.indexing.bm25_writer._tokenize
# --------------------------------------------------------------------------- #

def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens = (t.lower() for t in _TOKEN_RE.findall(text))
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]
