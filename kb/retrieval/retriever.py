"""
Hybrid retriever — orchestrates embed → dense + sparse → RRF → parent fetch.

Call shape:

    result = Retriever().retrieve(
        query="How does the API gateway authenticate requests?",
        user=UserContext(user_id="u_001", tenant_id="tenant_acme_staff",
                         department="engineering", role="manager"),
    )
    for hit in result.hits:
        print(hit.title, hit.section_path, hit.score)
        print(hit.parent_content)

Design notes:
    * One query embedding covers both collections (bge-m3 is shared).
    * Dense and sparse run sequentially here; a future slice will fan them
      out onto a thread pool once the retrieval path matters for latency.
    * RRF is weight-configurable via RetrievalConfig so operators can bias
      toward dense (conversational) or sparse (factoid/code) at runtime.
    * Parent expansion is optional; when on, we fold multiple children
      under the same parent into one hit, carrying forward the best child's
      match diagnostics.
    * The orchestrator applies the Python ACL predicate one more time on
      every final hit. This is belt-and-braces — if a bug in the filter
      builder lets something through, we still catch it here.
"""

from __future__ import annotations

import logging
import time

from kb.embeddings import EmbeddingClient
from kb.retrieval.acl import accessible_collections, hit_allowed
from kb.retrieval.dense import DenseRetriever
from kb.retrieval.fusion import FusedHit, rrf_fuse
from kb.retrieval.parent_store import ParentStore
from kb.retrieval.sparse import SparseRetriever
from kb.retrieval.types import (
    MatchVia,
    RetrievalConfig,
    RetrievalHit,
    RetrievalResult,
    UserContext,
)
from kb.settings import Settings, get_settings
from kb.types import AclPayload


logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        embedder: EmbeddingClient | None = None,
        dense: DenseRetriever | None = None,
        sparse: SparseRetriever | None = None,
        parents: ParentStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or EmbeddingClient(self.settings)
        self.dense = dense or DenseRetriever(self.settings)
        self.sparse = sparse or SparseRetriever(settings=self.settings)
        self.parents = parents or ParentStore(self.settings)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        user: UserContext | None = None,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        query = (query or "").strip()
        user = user or UserContext()
        config = config or RetrievalConfig()

        t_total_0 = time.monotonic()
        collections = accessible_collections(user, self.settings)
        logger.info(
            "retrieve user=%s tenant=%s role=%s collections=%s q=%r",
            user.user_id, user.tenant_id, user.role, collections, query[:80],
        )

        if not query:
            return _empty_result(query, user, collections)

        # 1. Embed
        t0 = time.monotonic()
        query_vector = self.embedder.embed([query])[0]
        embed_ms = int((time.monotonic() - t0) * 1000)

        # 2. Dense
        t0 = time.monotonic()
        dense_hits = self.dense.search(
            query_vector=query_vector,
            collections=collections,
            user=user,
            top_k=config.top_k_dense,
            source_allowlist=config.source_allowlist,
            allowed_match_kinds=config.allowed_match_kinds,
        )
        dense_ms = int((time.monotonic() - t0) * 1000)

        # 3. Sparse
        t0 = time.monotonic()
        sparse_hits = self.sparse.search(
            query=query,
            collections=collections,
            user=user,
            top_k=config.top_k_sparse,
            source_allowlist=config.source_allowlist,
        )
        sparse_ms = int((time.monotonic() - t0) * 1000)

        # 4. Fuse
        t0 = time.monotonic()
        fused = rrf_fuse(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            k=config.rrf_constant,
            dense_weight=config.rrf_dense_weight,
            sparse_weight=config.rrf_sparse_weight,
        )
        fusion_ms = int((time.monotonic() - t0) * 1000)

        # 5. Final ACL sweep (defence in depth)
        fused = [f for f in fused if hit_allowed(user, f.payload)]

        # 6. Dedupe by parent if requested (small-to-big)
        if config.dedupe_by_parent:
            fused = _pick_best_per_parent(fused)

        fused = fused[: config.top_k_final]

        # 7. Parent + child expansion from Postgres
        t0 = time.monotonic()
        hits = self._materialize(fused, include_parent=config.include_parent_content)
        parent_ms = int((time.monotonic() - t0) * 1000)

        total_ms = int((time.monotonic() - t_total_0) * 1000)
        return RetrievalResult(
            query=query,
            user_id=user.user_id,
            hits=hits,
            collections_searched=collections,
            dense_candidates=len(dense_hits),
            sparse_candidates=len(sparse_hits),
            fused_candidates=len(fused),
            final_hits=len(hits),
            embed_ms=embed_ms,
            dense_ms=dense_ms,
            sparse_ms=sparse_ms,
            fusion_ms=fusion_ms,
            parent_ms=parent_ms,
            total_ms=total_ms,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _materialize(
        self,
        fused: list[FusedHit],
        *,
        include_parent: bool,
    ) -> list[RetrievalHit]:
        if not fused:
            return []

        child_ids = [f.child_id for f in fused]
        children = self.parents.fetch_children(child_ids)

        parent_ids = [
            children[cid].parent_id for cid in child_ids if cid in children
        ]
        parent_recs = (
            self.parents.fetch_parents(parent_ids) if include_parent else {}
        )

        # Many of the fields we need are already in the payload; Postgres is
        # only needed for full content. We still fetch documents() lazily
        # only for the ids we actually lack a title for.
        hits: list[RetrievalHit] = []
        for f in fused:
            child = children.get(f.child_id)
            payload = f.payload

            parent_rec = (
                parent_recs.get(child.parent_id) if child and include_parent else None
            )

            section_path = payload.get("section_path") or ""
            if parent_rec and parent_rec.section_path:
                section_path = parent_rec.section_path

            acl = AclPayload(
                departments=payload.get("acl_departments", ["*"]),
                min_role=payload.get("acl_min_role", "anonymous"),
                tags=payload.get("acl_tags", []) or [],
            )

            hits.append(
                RetrievalHit(
                    child_id=f.child_id,
                    parent_id=(child.parent_id if child else payload.get("parent_id", "")),
                    document_id=(
                        child.document_id if child else payload.get("document_id", "")
                    ),
                    source_id=payload.get("source_id", ""),
                    source_uri=payload.get("source_uri", ""),
                    title=payload.get("title"),
                    section_path=section_path,
                    content=(child.content if child else ""),
                    summary=(child.summary if child else payload.get("summary")),
                    parent_content=(parent_rec.content if parent_rec else None),
                    score=f.score,
                    dense_rank=f.dense_rank,
                    sparse_rank=f.sparse_rank,
                    dense_score=f.dense_score,
                    sparse_score=f.sparse_score,
                    visibility=payload.get("visibility", "public"),
                    sensitivity=payload.get("sensitivity", "hosted_ok"),
                    acl=acl,
                    matched_via=_dedupe_matched(f.matched_via or []),
                )
            )
        return hits


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _pick_best_per_parent(fused: list[FusedHit]) -> list[FusedHit]:
    """
    Collapse candidates sharing the same parent into the best-scoring one.
    Carries forward the combined matched_via from ALL children under that
    parent so the UI can show "matched on content AND on question q3".
    """
    best: dict[str, FusedHit] = {}
    for f in fused:
        parent_id = f.payload.get("parent_id") or f.child_id
        current = best.get(parent_id)
        if current is None or f.score > current.score:
            if current is not None:
                # Merge matched_via from the losing sibling
                f.matched_via = (f.matched_via or []) + (current.matched_via or [])
            best[parent_id] = f
        else:
            current.matched_via = (current.matched_via or []) + (f.matched_via or [])

    return sorted(best.values(), key=lambda h: h.score, reverse=True)


def _dedupe_matched(vias: list[MatchVia]) -> list[MatchVia]:
    """Keep at most one entry per (kind, text) pair, preserve order."""
    seen: set[tuple[str, str | None]] = set()
    out: list[MatchVia] = []
    for v in vias:
        key = (v.kind, v.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _empty_result(query: str, user: UserContext, collections: list[str]) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        user_id=user.user_id,
        hits=[],
        collections_searched=collections,
    )


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #

def search(
    query: str,
    user: UserContext | None = None,
    config: RetrievalConfig | None = None,
) -> RetrievalResult:
    """Top-level convenience helper — builds a Retriever with default wiring."""
    return Retriever().retrieve(query=query, user=user, config=config)
