"""
Retrieval-side data contracts.

`UserContext` is the subject of an access-control check. In production it
will be populated from a decoded JWT (see `data/staff_directory.json`'s
`jwt_claims_contract`). For local testing, users can be looked up by id.

`RetrievalConfig` captures the tunables that affect recall/precision/latency.
Defaults target an interactive workflow (~<300 ms dense + sparse + parent
fetch on the demo corpus).

`RetrievalHit` is one result, parent-level after small-to-big expansion.
It carries enough metadata to build a citation and to explain *why* it was
retrieved (`matched_via`).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from kb.types import AclPayload


# --------------------------------------------------------------------------- #
# User / request context
# --------------------------------------------------------------------------- #

class UserContext(BaseModel):
    """
    Identity + entitlements for a single query. In a live API this comes
    from the decoded JWT; for tests / CLI it can be built directly or loaded
    from `data/staff_directory.json`.
    """

    user_id: str = "anonymous"
    tenant_id: Optional[str] = None  # None → external / customer
    department: Optional[str] = None
    role: str = "anonymous"
    extra_grants: list[str] = Field(default_factory=list)
    region: Optional[str] = None

    @property
    def is_external(self) -> bool:
        return self.tenant_id is None

    @property
    def granted_source_ids(self) -> list[str]:
        """
        Source IDs the user has explicit extra-grant access to.

        Handles both bare IDs (`"src_github_private"`) and the permission-
        scoped form used in staff_directory.json
        (`"src_github_private:read"`).
        """
        out: list[str] = []
        for g in self.extra_grants:
            if not g:
                continue
            src, _, _perm = g.partition(":")
            if src.startswith("src_"):
                out.append(src)
        return out


# --------------------------------------------------------------------------- #
# Request config
# --------------------------------------------------------------------------- #

class RetrievalConfig(BaseModel):
    """Tunables for a single retrieve() call."""

    top_k_dense: int = 30
    top_k_sparse: int = 30
    top_k_final: int = 10

    rrf_constant: int = 60
    rrf_dense_weight: float = 1.0
    rrf_sparse_weight: float = 1.0

    # If True, fold children under the same parent into a single hit
    # (small-to-big retrieval). If False, return child-level granularity.
    dedupe_by_parent: bool = True
    include_parent_content: bool = True

    # Restrict to a subset of Qdrant point kinds. Default: both content
    # and question-vector matches are eligible.
    allowed_match_kinds: list[str] = Field(default_factory=lambda: ["content", "question"])

    # Restrict to these sources at query time (diagnostic / safety rail).
    # None = no restriction.
    source_allowlist: Optional[list[str]] = None

    # ------------- Query rewriting (Slice 2 + 2B) ------------- #
    # "off" keeps behaviour identical to Slice 1. "multi_query" generates
    # paraphrases; "hyde" generates a hypothetical answer passage; "both"
    # generates both in one LLM call.
    rewrite_strategy: Literal["off", "multi_query", "hyde", "both"] = "off"
    multi_query_k: int = 2

    # When non-empty, the rewriter will coref-resolve the current query
    # against this history before any other expansion. Each entry is a
    # `(question, answer)` tuple. The Generator populates this from the
    # session store; callers using the Retriever directly can pass it
    # explicitly.
    conversation_history: list[tuple[str, str]] = Field(default_factory=list)
    # Add a step-back (broader) variant to query_variants. Cheap when
    # combined with rewrite_strategy != "off" (one extra JSON field in
    # the same LLM call). When strategy="off" + stepback=True we still
    # make one LLM call. Off by default — Slice 2C will calibrate.
    stepback: bool = False

    # ------------- Cross-encoder reranking (Slice 2) ------------- #
    rerank: bool = True
    # Number of fused candidates sent to the reranker. Higher = better
    # precision ceiling but more reranker cost. The reranker is only ever
    # invoked on the post-ACL, parent-deduped list — never on raw RRF.
    rerank_top_n: int = 30

    # ------------- Parallelism ------------- #
    # Cap the thread pool used for dense/sparse fan-out.
    max_parallel_workers: int = 8


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

class MatchVia(BaseModel):
    """
    Describes how a hit was matched. Used in the UI to show *why* each
    result is relevant ("matched on 'How does the API gateway work?'").
    """

    kind: str                                # "content" | "question" | "sparse"
    text: Optional[str] = None               # question text (for kind="question")
    score: float
    rank: Optional[int] = None               # rank within its retriever (1-based)


class RetrievalHit(BaseModel):
    child_id: str
    parent_id: str
    document_id: str
    source_id: str
    source_uri: str
    title: Optional[str] = None
    section_path: str = ""

    # Chunk-level content (the best-matching child under this parent).
    content: str = ""
    summary: Optional[str] = None

    # Parent content, populated when RetrievalConfig.include_parent_content
    # is True (the default). This is what goes into the LLM context window.
    parent_content: Optional[str] = None

    # Aggregate score. When reranking is on this is the cross-encoder
    # score; otherwise it's the RRF score. `rrf_score` / `rerank_score`
    # are always populated independently so you can audit both.
    score: float = 0.0
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rerank_rank: Optional[int] = None          # 1-based rank from the reranker
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None

    # Everything we need to enforce ACL upstream and to log cleanly.
    visibility: str = "public"
    sensitivity: str = "hosted_ok"
    acl: AclPayload = Field(default_factory=AclPayload)

    # How this hit was matched — one entry per signal.
    matched_via: list[MatchVia] = Field(default_factory=list)

    # Raw Qdrant / BM25 payload — retained only for debugging; not returned
    # to external callers.
    debug: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    user_id: str
    hits: list[RetrievalHit]

    # The actual list of queries hitting the dense retriever (original +
    # rewrites + HyDE passage if any). Useful for tracing / evals.
    query_variants: list[str] = Field(default_factory=list)
    hyde_passage: Optional[str] = None
    rewrite_strategy: str = "off"
    rewrite_fallback: Optional[str] = None     # set when the rewriter silently failed
    # Canonical query actually used for sparse retrieval and as the
    # anchor for embedding. Equals `query` unless coref-resolution
    # rewrote a follow-up question into a self-contained one.
    resolved_query: Optional[str] = None
    stepback_query: Optional[str] = None
    rerank_applied: bool = False
    rerank_fallback: Optional[str] = None      # set when rerank silently failed

    # Counts + per-stage timing for tracing.
    collections_searched: list[str] = Field(default_factory=list)
    dense_candidates: int = 0
    sparse_candidates: int = 0
    fused_candidates: int = 0
    reranked_candidates: int = 0
    final_hits: int = 0

    rewrite_ms: int = 0
    embed_ms: int = 0
    dense_ms: int = 0
    sparse_ms: int = 0
    fusion_ms: int = 0
    rerank_ms: int = 0
    parent_ms: int = 0
    total_ms: int = 0
