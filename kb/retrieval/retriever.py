"""
Hybrid retriever — orchestrates:

    rewrite  → embed (N variants)  → dense + sparse (parallel)
            → RRF fusion  → ACL sweep  → parent dedupe
            → cross-encoder rerank  → parent expansion  → hits

Design principles:
    * Every heavy-weight stage is optional and gated by `RetrievalConfig`.
      The default config turns rerank on and rewrite off — cheap for the
      demo, precision-first in production.
    * Every stage degrades gracefully. If rewrite LLM fails we fall back
      to the raw query; if rerank HF fails we fall back to RRF order.
      A single user query never breaks because an auxiliary service did.
    * ACL is enforced twice — once pushdown to Qdrant, once as a Python
      sweep over the fused list. The reranker ONLY sees post-ACL
      candidates so a vector-DB bug can't leak a doc through the
      cross-encoder's ranking either.
    * Postgres fetches are two-staged. First we fetch children for the
      rerank candidate pool (text needed for cross-encoder input). Then
      we fetch parents only for the final top-K survivors. This matches
      the query shape: many rows touched during rerank, few touched
      during final answer assembly.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import time
from dataclasses import dataclass

from kb.embeddings import EmbeddingClient
from kb.retrieval.acl import accessible_collections, hit_allowed
from kb.retrieval.dense import DenseRawHit, DenseRetriever
from kb.retrieval.fusion import FusedHit, rrf_fuse
from kb.retrieval.parent_store import ChildRecord, ParentStore
from kb.retrieval.rerank import CrossEncoderReranker, RerankerError
from kb.retrieval.rewrite import QueryRewriter, RewriteResult
from kb.retrieval.sparse import SparseRawHit, SparseRetriever
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


@dataclass
class _DensePoolResult:
    hits: list[DenseRawHit]
    per_variant_counts: list[int]


class Retriever:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        embedder: EmbeddingClient | None = None,
        dense: DenseRetriever | None = None,
        sparse: SparseRetriever | None = None,
        parents: ParentStore | None = None,
        rewriter: QueryRewriter | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or EmbeddingClient(self.settings)
        self.dense = dense or DenseRetriever(self.settings)
        self.sparse = sparse or SparseRetriever(settings=self.settings)
        self.parents = parents or ParentStore(self.settings)
        # Rewriter + reranker are lazily resolved on first need so a
        # retriever instance with rewrite/rerank turned off never
        # touches the LLM / HF reranker clients.
        self._rewriter_override = rewriter
        self._reranker_override = reranker
        self._rewriter: QueryRewriter | None = rewriter
        self._reranker: CrossEncoderReranker | None = reranker

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

        t_total = time.monotonic()
        collections = accessible_collections(user, self.settings)
        logger.info(
            "retrieve user=%s tenant=%s role=%s collections=%s "
            "rewrite=%s rerank=%s q=%r",
            user.user_id, user.tenant_id, user.role, collections,
            config.rewrite_strategy, config.rerank, query[:80],
        )

        if not query:
            return _empty_result(query, user, collections)

        # ------------------------------------------------------------ 1. rewrite
        t0 = time.monotonic()
        rewrite = self._run_rewrite(query, config)
        rewrite_ms = int((time.monotonic() - t0) * 1000)

        # The "canonical" query is the coref-resolved form when there
        # was a conversation history, otherwise the raw user query.
        # Sparse retrieval and the cross-encoder rerank both anchor on
        # the canonical text so pronouns in the raw query don't poison
        # BM25 / rerank pair scoring.
        canonical_query = rewrite.canonical or query

        # ------------------------------------------------------------ 2. embed
        # One batch for every dense query variant.
        dense_texts = list(rewrite.query_variants)
        if rewrite.hyde_passage:
            dense_texts.append(rewrite.hyde_passage)

        t0 = time.monotonic()
        query_vectors = self.embedder.embed(dense_texts)
        embed_ms = int((time.monotonic() - t0) * 1000)

        # ------------------------------------------------------------ 3. retrieve (parallel)
        dense_pool, sparse_hits, dense_ms, sparse_ms = self._parallel_retrieve(
            query_vectors=query_vectors,
            query=canonical_query,
            collections=collections,
            user=user,
            config=config,
        )

        # ------------------------------------------------------------ 4. fuse
        t0 = time.monotonic()
        fused = rrf_fuse(
            dense_hits=dense_pool,
            sparse_hits=sparse_hits,
            k=config.rrf_constant,
            dense_weight=config.rrf_dense_weight,
            sparse_weight=config.rrf_sparse_weight,
        )
        fusion_ms = int((time.monotonic() - t0) * 1000)

        # ------------------------------------------------------------ 5. ACL sweep (defence in depth)
        fused = [f for f in fused if hit_allowed(user, f.payload)]

        # ------------------------------------------------------------ 6. parent dedupe
        if config.dedupe_by_parent:
            fused = _pick_best_per_parent(fused)

        # ------------------------------------------------------------ 7. rerank
        rerank_pool = fused[: max(config.rerank_top_n, config.top_k_final)]
        rerank_applied = False
        rerank_fallback: str | None = None
        rerank_ms = 0
        children_cache: dict[str, ChildRecord] = {}

        if config.rerank and rerank_pool:
            t0 = time.monotonic()
            # Need child text as rerank input → bulk-fetch up front.
            children_cache = self.parents.fetch_children(
                [f.child_id for f in rerank_pool]
            )
            try:
                rerank_pool = self._apply_rerank(
                    query=canonical_query,
                    candidates=rerank_pool,
                    children=children_cache,
                )
                rerank_applied = True
            except RerankerError as exc:
                rerank_fallback = f"{type(exc).__name__}: {exc}"
                logger.warning("rerank failed — falling back to RRF order: %s", exc)
            rerank_ms = int((time.monotonic() - t0) * 1000)

        final = rerank_pool[: config.top_k_final]

        # ------------------------------------------------------------ 8. materialize hits
        t0 = time.monotonic()
        hits = self._materialize(
            fused=final,
            children_cache=children_cache,
            include_parent=config.include_parent_content,
        )
        parent_ms = int((time.monotonic() - t0) * 1000)

        total_ms = int((time.monotonic() - t_total) * 1000)
        return RetrievalResult(
            query=query,
            user_id=user.user_id,
            hits=hits,
            query_variants=dense_texts,
            hyde_passage=(rewrite.hyde_passage or None),
            rewrite_strategy=config.rewrite_strategy,
            rewrite_fallback=(rewrite.fallback_reason or None),
            resolved_query=(rewrite.resolved or None),
            stepback_query=(rewrite.stepback or None),
            rerank_applied=rerank_applied,
            rerank_fallback=rerank_fallback,
            collections_searched=collections,
            dense_candidates=len(dense_pool),
            sparse_candidates=len(sparse_hits),
            fused_candidates=len(fused),
            reranked_candidates=len(rerank_pool) if rerank_applied else 0,
            final_hits=len(hits),
            rewrite_ms=rewrite_ms,
            embed_ms=embed_ms,
            dense_ms=dense_ms,
            sparse_ms=sparse_ms,
            fusion_ms=fusion_ms,
            rerank_ms=rerank_ms,
            parent_ms=parent_ms,
            total_ms=total_ms,
        )

    # ================================================================== #
    # Stage: rewrite
    # ================================================================== #

    def _run_rewrite(self, query: str, config: RetrievalConfig) -> RewriteResult:
        # Skip the LLM only when nothing will be produced. Coref
        # resolution and stepback both require a call even with
        # rewrite_strategy="off", so we check all three signals.
        history = list(config.conversation_history or [])
        if (
            config.rewrite_strategy == "off"
            and not history
            and not config.stepback
        ):
            return RewriteResult(strategy="off", original=query)
        if self._rewriter is None:
            self._rewriter = QueryRewriter(settings=self.settings)
        return self._rewriter.rewrite(
            query,
            strategy=config.rewrite_strategy,
            k=config.multi_query_k,
            history=history,
            stepback=config.stepback,
        )

    # ================================================================== #
    # Stage: parallel dense + sparse
    # ================================================================== #

    def _parallel_retrieve(
        self,
        *,
        query_vectors: list[list[float]],
        query: str,
        collections: list[str],
        user: UserContext,
        config: RetrievalConfig,
    ) -> tuple[list[DenseRawHit], list[SparseRawHit], int, int]:
        """
        Run dense (once per query variant) and sparse (once, original query)
        in parallel. Dense hits from all variants are merged by child_id
        keeping the best rank.
        """
        t0_dense = time.monotonic()
        t0_sparse = time.monotonic()

        num_workers = max(1, min(
            config.max_parallel_workers,
            len(query_vectors) + 1,
        ))

        with cf.ThreadPoolExecutor(max_workers=num_workers) as ex:
            dense_futures = [
                ex.submit(
                    self.dense.search,
                    query_vector=vec,
                    collections=collections,
                    user=user,
                    top_k=config.top_k_dense,
                    source_allowlist=config.source_allowlist,
                    allowed_match_kinds=config.allowed_match_kinds,
                )
                for vec in query_vectors
            ]
            sparse_future = ex.submit(
                self.sparse.search,
                query=query,
                collections=collections,
                user=user,
                top_k=config.top_k_sparse,
                source_allowlist=config.source_allowlist,
            )

            variant_hits: list[list[DenseRawHit]] = []
            for fut in dense_futures:
                try:
                    variant_hits.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("dense variant failed: %s", exc)
                    variant_hits.append([])
            dense_ms = int((time.monotonic() - t0_dense) * 1000)

            try:
                sparse_hits = sparse_future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("sparse retrieval failed: %s", exc)
                sparse_hits = []
            sparse_ms = int((time.monotonic() - t0_sparse) * 1000)

        dense_pool = _merge_dense_variants(variant_hits)
        return dense_pool, sparse_hits, dense_ms, sparse_ms

    # ================================================================== #
    # Stage: rerank
    # ================================================================== #

    def _apply_rerank(
        self,
        *,
        query: str,
        candidates: list[FusedHit],
        children: dict[str, ChildRecord],
    ) -> list[FusedHit]:
        """Cross-encode (query, passage) and resort by rerank score."""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(settings=self.settings)

        passages = [_rerank_text(c, children) for c in candidates]
        results = self._reranker.rerank(query=query, passages=passages)

        # Attach scores + ranks onto the FusedHit payload; resort.
        for rank, r in enumerate(results, start=1):
            fh = candidates[r.index]
            fh.payload = dict(fh.payload)
            fh.payload["__rerank_score"] = r.score
            fh.payload["__rerank_rank"] = rank

        ordered = sorted(
            candidates,
            key=lambda f: (
                -float(f.payload.get("__rerank_score", 0.0)),
                int(f.payload.get("__rerank_rank", 10**9)),
            ),
        )
        return ordered

    # ================================================================== #
    # Stage: materialize
    # ================================================================== #

    def _materialize(
        self,
        *,
        fused: list[FusedHit],
        children_cache: dict[str, ChildRecord],
        include_parent: bool,
    ) -> list[RetrievalHit]:
        if not fused:
            return []

        # Fill in any children not already fetched (e.g. when rerank was off).
        missing = [f.child_id for f in fused if f.child_id not in children_cache]
        if missing:
            children_cache = {**children_cache, **self.parents.fetch_children(missing)}

        parent_ids = [
            children_cache[f.child_id].parent_id
            for f in fused
            if f.child_id in children_cache
        ]
        parent_recs = (
            self.parents.fetch_parents(parent_ids) if include_parent and parent_ids else {}
        )

        hits: list[RetrievalHit] = []
        for f in fused:
            child = children_cache.get(f.child_id)
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

            rerank_score = payload.get("__rerank_score")
            rerank_rank = payload.get("__rerank_rank")
            aggregate_score = (
                rerank_score if rerank_score is not None else f.score
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
                    score=aggregate_score,
                    rrf_score=f.score,
                    rerank_score=rerank_score,
                    rerank_rank=rerank_rank,
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

def _merge_dense_variants(
    variant_hits: list[list[DenseRawHit]],
) -> list[DenseRawHit]:
    """
    Merge dense hits from multiple query variants by child_id, keeping the
    one with the BEST rank (lowest rank number). Re-rank 1..N by ordering.

    Why best rank rather than best score? Scores from different queries
    (different embeddings) aren't directly comparable — a 0.82 from a
    rewrite is not the same 0.82 as the original. Ranks within each list
    are comparable and feed directly into RRF.
    """
    best: dict[str, DenseRawHit] = {}
    for hits in variant_hits:
        for h in hits:
            if not h.child_id:
                continue
            current = best.get(h.child_id)
            if current is None or h.rank < current.rank:
                best[h.child_id] = h
    ordered = sorted(best.values(), key=lambda h: (h.rank, -h.score))
    for new_rank, h in enumerate(ordered, start=1):
        h.rank = new_rank
        h.matched.rank = new_rank
    return ordered


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


def _rerank_text(f: FusedHit, children: dict[str, ChildRecord]) -> str:
    """
    Build the passage text the reranker scores against.

    Prefer the actual child content (what semantically matched). Prepend
    the summary when available — it gives the cross-encoder a concise
    topical framing, which helps on noisy / fragmented chunks. We
    deliberately do NOT include hypothetical questions here: the reranker
    should judge whether the *passage* answers the query, not whether a
    synthetic question lines up with it.
    """
    child = children.get(f.child_id)
    if child is None:
        # Fall back to payload-section to avoid sending an empty string.
        section = str(f.payload.get("section_path") or "").strip()
        return section or f.child_id

    parts: list[str] = []
    if child.summary:
        parts.append(child.summary.strip())
    if child.content:
        parts.append(child.content.strip())
    return "\n\n".join(p for p in parts if p) or child.content or f.child_id


def _empty_result(query: str, user: UserContext, collections: list[str]) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        user_id=user.user_id,
        hits=[],
        query_variants=[query] if query else [],
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
