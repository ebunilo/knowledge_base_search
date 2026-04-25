"""
Tests for `kb.generation.faithfulness.FaithfulnessChecker`.

NLIClient is mocked so these are offline. Coverage:
    * Sentences with no markers → UNVERIFIED, no NLI calls.
    * Sentences with cited markers → SUPPORTED iff max entailment ≥ thr.
    * Multiple cited sources → score = max across sources.
    * Aggregates (supported_ratio, mean_entailment) computed correctly.
    * NLI failure → empty report with fallback_reason.
    * Empty answer → empty report.
    * Lane-safe skip in prod profile when any hit is self_hosted_only.
    * Out-of-range markers silently dropped (citation extractor handles them).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kb.generation.faithfulness import FaithfulnessChecker
from kb.generation.nli_client import NLIClientError
from kb.generation.types import GenerationConfig
from kb.retrieval.types import RetrievalHit
from kb.settings import Profile, Settings


def _hit(parent_id: str, content: str, sensitivity: str = "hosted_ok") -> RetrievalHit:
    return RetrievalHit(
        child_id=parent_id + "_c0",
        parent_id=parent_id,
        document_id="doc1",
        source_id="src1",
        source_uri=f"file:///{parent_id}",
        title="Doc",
        section_path="/",
        content=content[:120],
        parent_content=content,
        score=0.9,
        sensitivity=sensitivity,
    )


def _checker(scores=None, raises=False, profile=Profile.DEMO) -> FaithfulnessChecker:
    """Build a checker backed by a MagicMock NLIClient."""
    nli = MagicMock()
    if raises:
        nli.entailment_score.side_effect = NLIClientError("network down")
    elif scores is not None:
        nli.entailment_score.side_effect = list(scores)
    # `demo_allow_hosted_for_selfhosted_lane` defaults to true in the
    # workspace's .env (demo profile). Force it off here so the prod
    # safety-rail validator doesn't trip when we test the prod path.
    return FaithfulnessChecker(
        Settings(
            hf_api_token="test",
            app_profile=profile,
            demo_allow_hosted_for_selfhosted_lane=False,
        ),
        nli=nli,
    )


# --------------------------------------------------------------------------- #
# Trivial inputs
# --------------------------------------------------------------------------- #

class TestTrivial:
    def test_empty_answer_returns_empty_report(self):
        chk = _checker()
        out = chk.check("", [_hit("p1", "anything")])
        assert out.per_sentence == []
        assert out.cited_sentences == 0
        assert chk._nli.entailment_score.call_count == 0

    def test_no_used_hits_marks_all_unverified(self):
        # A single sentence with [1] but no hits to map to → unverified.
        chk = _checker()
        out = chk.check("The system uses OAuth [1].", used_hits=[])
        # citation resolution returns nothing, sentence becomes unverified.
        assert len(out.per_sentence) == 1
        assert out.per_sentence[0].status == "unverified"
        assert chk._nli.entailment_score.call_count == 0


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #

class TestVerdicts:
    def test_supported_when_above_threshold(self):
        chk = _checker(scores=[0.85])
        hits = [_hit("p1", "OAuth secures the gateway.")]
        out = chk.check(
            "The gateway uses OAuth [1].", hits,
            config=GenerationConfig(faithfulness_threshold=0.5),
        )
        assert len(out.per_sentence) == 1
        s = out.per_sentence[0]
        assert s.status == "supported"
        assert s.entailment_score == pytest.approx(0.85)
        assert s.markers == [1]
        assert s.cited_parent_ids == ["p1"]
        assert out.supported_sentences == 1
        assert out.unsupported_sentences == 0
        assert out.cited_sentences == 1
        assert out.supported_ratio == 1.0
        assert out.nli_calls == 1

    def test_unsupported_when_below_threshold(self):
        chk = _checker(scores=[0.10])
        hits = [_hit("p1", "Documentation is updated quarterly.")]
        out = chk.check(
            "The gateway uses OAuth [1].", hits,
            config=GenerationConfig(faithfulness_threshold=0.5),
        )
        s = out.per_sentence[0]
        assert s.status == "unsupported"
        assert out.unsupported_sentences == 1
        assert out.supported_ratio == 0.0

    def test_uncited_sentence_is_unverified_no_nli_call(self):
        chk = _checker()
        hits = [_hit("p1", "anything")]
        # No [N] anywhere — purely framing text.
        out = chk.check("Here is how the system works overall.", hits)
        assert out.per_sentence[0].status == "unverified"
        assert out.unverified_sentences == 1
        assert out.cited_sentences == 0
        assert chk._nli.entailment_score.call_count == 0

    def test_max_across_multiple_sources(self):
        # [1] returns 0.2, [2] returns 0.8 — sentence is supported
        # (max), not unsupported (mean).
        chk = _checker(scores=[0.2, 0.8])
        hits = [
            _hit("p1", "First source."),
            _hit("p2", "Second source mentioning OAuth."),
        ]
        out = chk.check(
            "The gateway uses OAuth [1][2].", hits,
            config=GenerationConfig(faithfulness_threshold=0.5),
        )
        s = out.per_sentence[0]
        assert s.status == "supported"
        assert s.entailment_score == pytest.approx(0.8)
        assert s.per_source_scores == [pytest.approx(0.2), pytest.approx(0.8)]
        assert s.cited_parent_ids == ["p1", "p2"]
        assert out.nli_calls == 2

    def test_aggregates_mean_entailment(self):
        chk = _checker(scores=[0.9, 0.4])
        hits = [_hit("p1", "src1"), _hit("p2", "src2")]
        out = chk.check(
            "Claim one [1]. Claim two [2].", hits,
            config=GenerationConfig(faithfulness_threshold=0.5),
        )
        assert out.cited_sentences == 2
        assert out.supported_sentences == 1
        assert out.unsupported_sentences == 1
        assert out.supported_ratio == 0.5
        assert out.mean_entailment == pytest.approx(0.65)

    def test_out_of_range_markers_dropped(self):
        # Citation [9] points at no hit → no NLI call, sentence
        # becomes unverified (treated like a marker-less sentence).
        chk = _checker()
        hits = [_hit("p1", "src")]
        out = chk.check("Bogus reference [9].", hits)
        s = out.per_sentence[0]
        assert s.status == "unverified"
        assert chk._nli.entailment_score.call_count == 0

    def test_cache_avoids_duplicate_nli_calls(self):
        # Same sentence cited twice with same source within an answer.
        chk = _checker(scores=[0.7])
        hits = [_hit("p1", "OAuth secures things.")]
        # Two [1] references to the same source in one sentence:
        # extractor dedupes parent_ids → one NLI call.
        out = chk.check("Repeat [1][1].", hits)
        assert chk._nli.entailment_score.call_count == 1
        assert out.per_sentence[0].cited_parent_ids == ["p1"]


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #

class TestFailures:
    def test_nli_failure_returns_fallback_report(self):
        chk = _checker(raises=True)
        hits = [_hit("p1", "src")]
        out = chk.check("A claim [1].", hits)
        assert out.per_sentence == []
        assert out.fallback_reason and "NLIClientError" in out.fallback_reason
        assert out.cited_sentences == 0

    def test_prod_self_hosted_skips_check(self):
        # In prod profile, NLI must NOT be invoked when any hit is self_hosted_only.
        chk = _checker(scores=[0.99], profile=Profile.PROD)
        hits = [_hit("p1", "sensitive content", sensitivity="self_hosted_only")]
        out = chk.check("A claim [1].", hits)
        assert out.fallback_reason is not None
        assert "self_hosted_only" in out.fallback_reason
        assert chk._nli.entailment_score.call_count == 0

    def test_prod_hosted_only_runs_check(self):
        chk = _checker(scores=[0.7], profile=Profile.PROD)
        hits = [_hit("p1", "public content", sensitivity="hosted_ok")]
        out = chk.check("A claim [1].", hits,
                        config=GenerationConfig(faithfulness_threshold=0.5))
        assert out.fallback_reason is None
        assert out.per_sentence[0].status == "supported"
