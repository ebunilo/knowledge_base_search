"""
Tests for `kb.generation.confidence.compute_confidence`.

The math is heuristic — these tests pin the SHAPE of the function:
    * No retrieval / no hits → 0.0.
    * No faithfulness signal (None or fallback) → retrieval-only, capped.
    * Faithfulness signal blends in via geometric mean.
    * Both signals high → confidence high.
    * Either signal low → confidence pulled down.
"""

from __future__ import annotations

import math

import pytest

from kb.generation.confidence import compute_confidence
from kb.generation.types import FaithfulnessReport
from kb.retrieval.types import RetrievalHit, RetrievalResult


def _retrieval(top_score: float, rerank_applied: bool = True) -> RetrievalResult:
    h = RetrievalHit(
        child_id="c1", parent_id="p1", document_id="d1",
        source_id="s", source_uri="u", title="t",
        score=top_score,
    )
    return RetrievalResult(
        query="q", user_id="u", hits=[h],
        rerank_applied=rerank_applied,
    )


def _faithfulness(
    *,
    cited: int,
    supported: int,
    fallback: str = "",
) -> FaithfulnessReport:
    ratio = (supported / cited) if cited else 0.0
    return FaithfulnessReport(
        cited_sentences=cited,
        supported_sentences=supported,
        unsupported_sentences=cited - supported,
        supported_ratio=ratio,
        fallback_reason=fallback or None,
    )


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

class TestEdges:
    def test_no_retrieval_is_zero(self):
        assert compute_confidence(None, None) == 0.0

    def test_no_hits_is_zero(self):
        empty = RetrievalResult(query="q", user_id="u", hits=[])
        assert compute_confidence(empty, None) == 0.0

    def test_no_faithfulness_capped(self):
        # Strong rerank score without verification: capped at 0.85.
        r = _retrieval(top_score=10.0, rerank_applied=True)  # sigmoid ≈ 0.99996
        c = compute_confidence(r, None)
        assert c == pytest.approx(0.85, abs=1e-3)

    def test_faithfulness_fallback_treated_as_no_signal(self):
        r = _retrieval(top_score=10.0, rerank_applied=True)
        f = _faithfulness(cited=2, supported=2, fallback="lane skip")
        c = compute_confidence(r, f)
        # Same cap as no-faithfulness path — fallback means we don't trust it.
        assert c == pytest.approx(0.85, abs=1e-3)


# --------------------------------------------------------------------------- #
# Combination math
# --------------------------------------------------------------------------- #

class TestCombination:
    def test_both_strong_yields_high_confidence(self):
        r = _retrieval(top_score=5.0, rerank_applied=True)  # ≈ 0.993
        f = _faithfulness(cited=3, supported=3)              # ratio 1.0
        c = compute_confidence(r, f)
        # geometric mean of (~0.993, 1.0) ≈ 0.997
        assert 0.95 < c <= 1.0

    def test_weak_faithfulness_pulls_down_strong_retrieval(self):
        r = _retrieval(top_score=5.0, rerank_applied=True)   # ~0.99
        f = _faithfulness(cited=4, supported=1)               # ratio 0.25
        c = compute_confidence(r, f)
        # sqrt(0.99 * 0.25) ≈ 0.498
        assert 0.4 < c < 0.6

    def test_weak_retrieval_pulls_down_strong_faithfulness(self):
        # Negative rerank logit → low retrieval signal.
        r = _retrieval(top_score=-3.0, rerank_applied=True)   # sigmoid ≈ 0.047
        f = _faithfulness(cited=4, supported=4)               # ratio 1.0
        c = compute_confidence(r, f)
        assert 0.15 < c < 0.30

    def test_no_cited_sentences_is_neutral_faithfulness(self):
        r = _retrieval(top_score=5.0, rerank_applied=True)   # ~0.99
        f = _faithfulness(cited=0, supported=0)               # all unverified
        c = compute_confidence(r, f)
        # sqrt(0.99 * 0.5) ≈ 0.703
        assert 0.65 < c < 0.75


# --------------------------------------------------------------------------- #
# RRF (no rerank) path
# --------------------------------------------------------------------------- #

class TestRRFPath:
    def test_rrf_score_normalized(self):
        # Typical RRF rank-1 score ≈ 0.025 with two contributing retrievers.
        r = _retrieval(top_score=0.025, rerank_applied=False)
        f = _faithfulness(cited=2, supported=2)
        c = compute_confidence(r, f)
        # min(1, 0.025/0.025) * 1.0 → 1.0; geo mean → 1.0
        assert c == pytest.approx(1.0, abs=1e-3)

    def test_low_rrf_score_yields_low_confidence(self):
        r = _retrieval(top_score=0.005, rerank_applied=False)  # 1/5 of baseline
        f = _faithfulness(cited=2, supported=2)
        c = compute_confidence(r, f)
        # sqrt(0.2 * 1.0) ≈ 0.447
        assert 0.4 < c < 0.5
