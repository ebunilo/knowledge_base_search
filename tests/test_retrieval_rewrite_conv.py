"""
Tests for the conversation-aware extensions to QueryRewriter
(Phase 3 · Slice 2B): coreference resolution and step-back prompting.

The rewriter must:

    * make ONE LLM call covering every requested capability
    * fall back to the raw question if the LLM omits ``resolved_query``
    * leave Slice 2 behaviour unchanged when no history / no stepback
    * produce a sensible ``query_variants`` ordering: canonical first,
      then rewrites, then stepback (deduped, case-insensitive)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from kb.enrichment.llm_client import CompletionResult, LLMClientError
from kb.retrieval.rewrite import QueryRewriter
from kb.settings import Settings


def _rewriter(llm_responses=None, raise_on_call=False) -> QueryRewriter:
    llm = MagicMock()
    if raise_on_call:
        llm.complete.side_effect = LLMClientError("boom")
    else:
        llm.complete.side_effect = [
            CompletionResult(text=body, provider="openai", model="gpt-4o-mini", latency_ms=20)
            for body in (llm_responses or [])
        ]
    return QueryRewriter(settings=Settings(openai_api_key="test"), llm=llm)


# --------------------------------------------------------------------------- #
# Coref resolution
# --------------------------------------------------------------------------- #

class TestCoref:
    def test_resolves_pronouns_against_history(self):
        body = json.dumps({"resolved_query": "What are the rate limits of the gateway?"})
        rw = _rewriter([body])
        out = rw.rewrite(
            "What about its limits?",
            strategy="off",
            history=[("How does the API gateway work?", "It routes …")],
        )
        assert out.resolved == "What are the rate limits of the gateway?"
        assert out.canonical == "What are the rate limits of the gateway?"
        assert out.original == "What about its limits?"
        # canonical replaces original at the head of query_variants.
        assert out.query_variants[0] == "What are the rate limits of the gateway?"

    def test_no_history_skips_coref(self):
        rw = _rewriter()
        out = rw.rewrite("self-contained question", strategy="off")
        # Strategy=off + no history + no stepback → no LLM call.
        assert rw.llm.complete.call_count == 0
        assert out.resolved == ""
        assert out.canonical == "self-contained question"

    def test_llm_omits_resolved_falls_back_to_raw(self):
        # LLM returned an empty / missing resolved_query. We must NOT
        # lose the user's question — fall back to raw.
        body = json.dumps({"resolved_query": ""})
        rw = _rewriter([body])
        out = rw.rewrite(
            "follow up?",
            strategy="off",
            history=[("Q", "A")],
        )
        assert out.resolved == "follow up?"
        assert out.canonical == "follow up?"


# --------------------------------------------------------------------------- #
# Step-back
# --------------------------------------------------------------------------- #

class TestStepback:
    def test_emits_stepback_variant(self):
        body = json.dumps({"stepback_query": "How does the company define SLOs?"})
        rw = _rewriter([body])
        out = rw.rewrite(
            "What's the SLO for billing?",
            strategy="off",
            stepback=True,
        )
        assert out.stepback == "How does the company define SLOs?"
        # canonical first, then stepback in query_variants
        assert out.query_variants == [
            "What's the SLO for billing?",
            "How does the company define SLOs?",
        ]

    def test_stepback_only_makes_one_call(self):
        rw = _rewriter([json.dumps({"stepback_query": "general q"})])
        rw.rewrite("specific q", strategy="off", stepback=True)
        assert rw.llm.complete.call_count == 1


# --------------------------------------------------------------------------- #
# Combined: coref + multi_query + stepback in ONE call
# --------------------------------------------------------------------------- #

class TestCombined:
    def test_single_llm_call_with_all_signals(self):
        body = json.dumps({
            "resolved_query": "What are the latency requirements of the gateway?",
            "rewrites": ["latency targets for the API gateway"],
            "stepback_query": "What latency standards do we maintain?",
        })
        rw = _rewriter([body])
        out = rw.rewrite(
            "What about its latency?",
            strategy="multi_query",
            k=1,
            history=[("How does the API gateway work?", "Routes traffic …")],
            stepback=True,
        )
        # ONE round-trip — never two.
        assert rw.llm.complete.call_count == 1
        assert out.resolved == "What are the latency requirements of the gateway?"
        assert out.rewrites == ["latency targets for the API gateway"]
        assert out.stepback == "What latency standards do we maintain?"
        # canonical replaces original at the head; rewrites then stepback.
        assert out.query_variants == [
            "What are the latency requirements of the gateway?",
            "latency targets for the API gateway",
            "What latency standards do we maintain?",
        ]


# --------------------------------------------------------------------------- #
# Failure modes — must never raise
# --------------------------------------------------------------------------- #

class TestFallback:
    def test_llm_failure_with_history_yields_raw_canonical(self):
        rw = _rewriter(raise_on_call=True)
        out = rw.rewrite(
            "follow up?",
            strategy="off",
            history=[("Q", "A")],
        )
        assert out.resolved == ""
        assert out.canonical == "follow up?"
        assert "LLMClientError" in out.fallback_reason

    def test_unparseable_response_with_combined_strategy(self):
        rw = _rewriter(["not json"])
        out = rw.rewrite(
            "follow up?",
            strategy="multi_query", k=2,
            history=[("Q", "A")],
            stepback=True,
        )
        # Parse failure is silent — rewriter returns whatever defaults
        # apply. canonical falls back to raw because resolved was missing.
        assert out.canonical == "follow up?"
        assert out.rewrites == []
        assert out.stepback == ""
