"""
Dense retriever — Qdrant REST KNN.

Responsibilities:
    * Accept a query vector and a list of accessible collections.
    * Issue one KNN search per collection, pushing the ACL filter down.
    * Normalise hits into `_DenseRawHit` with a stable shape regardless of
      whether the match came from a content vector or a question vector.
    * Dedupe by child_id across the union of collections, keeping the
      highest-scoring match per child. Record `matched_via` diagnostics
      so the UI can explain *why* each result matched.

The dense retriever NEVER filters by ACL itself — it trusts the filter
pushed to Qdrant. The Python-side `hit_allowed` predicate is still
applied in the orchestrator as defence in depth.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from kb.retrieval.types import MatchVia, UserContext
from kb.retrieval.acl import build_qdrant_filter
from kb.settings import Settings, get_settings


logger = logging.getLogger(__name__)


class DenseRetrieverError(Exception):
    pass


@dataclass
class DenseRawHit:
    """A single Qdrant hit, pre-fusion."""
    child_id: str
    score: float
    rank: int                            # 1-based rank within its collection
    collection: str
    payload: dict[str, Any]
    matched: MatchVia


class DenseRetriever:
    TIMEOUT_S = 10

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._url = self.settings.qdrant_url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if self.settings.qdrant_api_key:
            self._headers["api-key"] = self.settings.qdrant_api_key

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def search(
        self,
        *,
        query_vector: list[float],
        collections: list[str],
        user: UserContext,
        top_k: int = 30,
        source_allowlist: list[str] | None = None,
        allowed_match_kinds: list[str] | None = None,
    ) -> list[DenseRawHit]:
        """
        Run KNN against every accessible collection, merge, and dedupe by
        child_id. The returned list is sorted by descending score.
        """
        raw_hits: list[DenseRawHit] = []
        for collection in collections:
            filt = build_qdrant_filter(
                user,
                collection,
                source_allowlist=source_allowlist,
                allowed_match_kinds=allowed_match_kinds,
                settings=self.settings,
            )
            hits = self._search_one(
                collection=collection,
                query_vector=query_vector,
                top_k=top_k,
                filt=filt,
            )
            raw_hits.extend(hits)

        return self._dedupe_by_child(raw_hits)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _search_one(
        self,
        *,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filt: dict[str, Any] | None,
    ) -> list[DenseRawHit]:
        body: dict[str, Any] = {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": True,
            "with_vector": False,
        }
        if filt is not None:
            body["filter"] = filt

        url = f"{self._url}/collections/{collection}/points/search"
        t0 = time.monotonic()
        try:
            r = requests.post(
                url, headers=self._headers, json=body, timeout=self.TIMEOUT_S,
            )
        except requests.RequestException as exc:
            logger.error("qdrant search failed (%s): %s", collection, exc)
            return []
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if r.status_code == 404:
            logger.debug("collection %s not present (skipping)", collection)
            return []
        if not r.ok:
            logger.error(
                "qdrant search %s failed %d: %s",
                collection, r.status_code, r.text[:300],
            )
            return []

        result = r.json().get("result", []) or []
        logger.debug(
            "dense search collection=%s returned=%d in %dms",
            collection, len(result), elapsed_ms,
        )
        return [
            self._to_hit(point=p, rank=idx + 1, collection=collection)
            for idx, p in enumerate(result)
        ]

    @staticmethod
    def _to_hit(*, point: dict[str, Any], rank: int, collection: str) -> DenseRawHit:
        payload = point.get("payload") or {}
        score = float(point.get("score", 0.0))
        kind = payload.get("kind", "content")
        matched = MatchVia(
            kind=kind,
            text=payload.get("question_text") if kind == "question" else None,
            score=score,
            rank=rank,
        )
        return DenseRawHit(
            child_id=payload.get("child_id", ""),
            score=score,
            rank=rank,
            collection=collection,
            payload=payload,
            matched=matched,
        )

    @staticmethod
    def _dedupe_by_child(hits: list[DenseRawHit]) -> list[DenseRawHit]:
        """
        Keep the best-scoring match per child_id across all collections and
        match kinds. Carries forward the matched-via info of the winning point.

        Returns the deduped list sorted by descending score; ranks are
        re-assigned (1-based) in that order so downstream RRF sees a clean
        ordering.
        """
        best: dict[str, DenseRawHit] = {}
        for h in hits:
            if not h.child_id:
                continue
            current = best.get(h.child_id)
            if current is None or h.score > current.score:
                best[h.child_id] = h

        ordered = sorted(best.values(), key=lambda h: h.score, reverse=True)
        for new_rank, h in enumerate(ordered, start=1):
            h.rank = new_rank
            h.matched.rank = new_rank
        return ordered
