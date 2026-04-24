"""
Tests for RRF fusion.

The numbers in these tests come from the standard formula:
    rrf(d) = 1/(k + rank_dense) + 1/(k + rank_sparse)
"""

from __future__ import annotations

from kb.retrieval.dense import DenseRawHit
from kb.retrieval.fusion import rrf_fuse
from kb.retrieval.sparse import SparseRawHit
from kb.retrieval.types import MatchVia


def _dense(child_id: str, rank: int, score: float = 0.9) -> DenseRawHit:
    return DenseRawHit(
        child_id=child_id,
        score=score,
        rank=rank,
        collection="public_v1",
        payload={"child_id": child_id, "parent_id": f"p_{child_id}"},
        matched=MatchVia(kind="content", score=score, rank=rank),
    )


def _sparse(child_id: str, rank: int, score: float = 5.0) -> SparseRawHit:
    return SparseRawHit(
        child_id=child_id,
        score=score,
        rank=rank,
        collection="public_v1",
        payload={"child_id": child_id, "parent_id": f"p_{child_id}"},
        matched=MatchVia(kind="sparse", score=score, rank=rank),
    )


class TestRRFFuse:
    def test_intersection_ranked_higher_than_singletons(self):
        fused = rrf_fuse(
            dense_hits=[_dense("a", 1), _dense("b", 2), _dense("c", 3)],
            sparse_hits=[_sparse("b", 1), _sparse("d", 2), _sparse("a", 3)],
        )
        # 'a': dense=1 + sparse=3
        # 'b': dense=2 + sparse=1
        # 'c': dense=3
        # 'd': sparse=2
        ids = [h.child_id for h in fused]
        assert ids[0] in {"a", "b"}
        assert ids[1] in {"a", "b"}
        assert {"a", "b"} == set(ids[:2])

    def test_dense_only_candidate_preserved(self):
        fused = rrf_fuse(
            dense_hits=[_dense("only", 1)], sparse_hits=[],
        )
        assert len(fused) == 1
        h = fused[0]
        assert h.child_id == "only"
        assert h.dense_rank == 1
        assert h.sparse_rank is None
        # 1 / (60 + 1)
        assert h.score == 1.0 / 61

    def test_weighting_biases_ranking(self):
        # Same ranks, opposite retrievers; with dense_weight=2 dense-only
        # outranks sparse-only.
        fused = rrf_fuse(
            dense_hits=[_dense("d", 1)],
            sparse_hits=[_sparse("s", 1)],
            dense_weight=2.0,
            sparse_weight=1.0,
        )
        ids = [h.child_id for h in fused]
        assert ids == ["d", "s"]

    def test_matched_via_carries_both_signals(self):
        fused = rrf_fuse(
            dense_hits=[_dense("x", 1)],
            sparse_hits=[_sparse("x", 1)],
        )
        assert len(fused) == 1
        kinds = {mv.kind for mv in fused[0].matched_via}
        assert kinds == {"content", "sparse"}

    def test_top_k_truncates(self):
        fused = rrf_fuse(
            dense_hits=[_dense(f"c{i}", i) for i in range(1, 6)],
            sparse_hits=[],
            top_k=2,
        )
        assert len(fused) == 2
        # Highest ranks (i=1,2) win.
        assert {h.child_id for h in fused} == {"c1", "c2"}

    def test_empty_inputs(self):
        assert rrf_fuse(dense_hits=[], sparse_hits=[]) == []
