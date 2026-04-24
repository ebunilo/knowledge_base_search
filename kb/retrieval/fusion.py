"""
Reciprocal Rank Fusion (RRF).

RRF is the standard hybrid-search fusion algorithm. Given two or more
ranked lists of candidates, the fused score for a candidate is:

    rrf_score(c) = sum_{r in retrievers}  w_r / (k + rank_r(c))

where:
    * rank_r(c) is the 1-based rank of c in retriever r (or ∞ if absent).
    * k is a smoothing constant; 60 is the value in the original paper and
      the de-facto default across the industry.
    * w_r is an optional per-retriever weight. Usually 1.0 for both
      retrievers; setting dense_weight > sparse_weight biases toward
      semantic matching on conversational queries.

This module is pure — no I/O, no network — so it's cheap to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kb.retrieval.dense import DenseRawHit
from kb.retrieval.sparse import SparseRawHit
from kb.retrieval.types import MatchVia


@dataclass
class FusedHit:
    """A candidate after RRF fusion, before parent expansion."""

    child_id: str
    score: float                              # RRF score
    payload: dict[str, Any]                   # merged payload (prefers dense)
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    matched_via: list[MatchVia] = None        # type: ignore[assignment]


def rrf_fuse(
    *,
    dense_hits: list[DenseRawHit],
    sparse_hits: list[SparseRawHit],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    top_k: int | None = None,
) -> list[FusedHit]:
    """
    Merge dense + sparse candidates with RRF. Returns a descending-scored
    list; ties are broken by dense rank, then sparse rank.
    """
    # Build rank maps, keyed by child_id.
    dense_by_id: dict[str, DenseRawHit] = {h.child_id: h for h in dense_hits if h.child_id}
    sparse_by_id: dict[str, SparseRawHit] = {h.child_id: h for h in sparse_hits if h.child_id}

    all_ids = set(dense_by_id) | set(sparse_by_id)
    fused: list[FusedHit] = []

    for child_id in all_ids:
        d = dense_by_id.get(child_id)
        s = sparse_by_id.get(child_id)

        score = 0.0
        if d is not None:
            score += dense_weight / (k + d.rank)
        if s is not None:
            score += sparse_weight / (k + s.rank)

        payload: dict[str, Any] = {}
        if s is not None:
            payload.update(s.payload)
        if d is not None:
            # Dense payload is canonical — overwrite any sparse-side values.
            payload.update(d.payload)

        matched_via: list[MatchVia] = []
        if d is not None:
            matched_via.append(d.matched)
        if s is not None:
            matched_via.append(s.matched)

        fused.append(
            FusedHit(
                child_id=child_id,
                score=score,
                payload=payload,
                dense_rank=d.rank if d else None,
                sparse_rank=s.rank if s else None,
                dense_score=d.score if d else None,
                sparse_score=s.score if s else None,
                matched_via=matched_via,
            )
        )

    fused.sort(
        key=lambda h: (
            -h.score,
            h.dense_rank if h.dense_rank is not None else 10**9,
            h.sparse_rank if h.sparse_rank is not None else 10**9,
        )
    )
    if top_k is not None:
        fused = fused[:top_k]
    return fused
