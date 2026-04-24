"""
Tests for `kb.retrieval.rewrite`.

We mock the LLMClient so these are fast and offline. The rewriter must
NEVER raise — on LLM failure it has to return a RewriteResult with the
fallback_reason set, so retrieval can continue with the raw query.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from kb.enrichment.llm_client import CompletionResult, LLMClientError
from kb.retrieval.rewrite import QueryRewriter
from kb.settings import Settings


def _rewriter(llm_responses: list = None, raise_on_call: bool = False) -> QueryRewriter:
    """Build a QueryRewriter backed by a MagicMock LLMClient."""
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
# Strategy: off / empty
# --------------------------------------------------------------------------- #

class TestOff:
    def test_off_never_calls_llm(self):
        rw = _rewriter()
        out = rw.rewrite("how does x work?", strategy="off")
        assert out.strategy == "off"
        assert out.rewrites == []
        assert out.hyde_passage == ""
        assert out.fallback_reason == ""
        assert rw.llm.complete.call_count == 0

    def test_empty_query_short_circuits(self):
        rw = _rewriter()
        out = rw.rewrite("   ", strategy="multi_query")
        assert out.rewrites == []
        assert rw.llm.complete.call_count == 0


# --------------------------------------------------------------------------- #
# Multi-query
# --------------------------------------------------------------------------- #

class TestMultiQuery:
    def test_parses_rewrites(self):
        body = json.dumps({"rewrites": ["variant one", "variant two"]})
        rw = _rewriter([body])
        out = rw.rewrite("how does x work?", strategy="multi_query", k=2)
        assert out.rewrites == ["variant one", "variant two"]
        assert out.hyde_passage == ""
        assert out.fallback_reason == ""
        # query_variants includes the original first, then rewrites
        assert out.query_variants == ["how does x work?", "variant one", "variant two"]

    def test_truncates_to_k(self):
        body = json.dumps({"rewrites": ["a", "b", "c", "d"]})
        rw = _rewriter([body])
        out = rw.rewrite("q", strategy="multi_query", k=2)
        assert out.rewrites == ["a", "b"]

    def test_deduplicates_variants_case_insensitive(self):
        body = json.dumps({"rewrites": ["Hello", "hello", "HELLO world"]})
        rw = _rewriter([body])
        out = rw.rewrite("Hello", strategy="multi_query", k=3)
        # "Hello" == original; "hello" is a case-insensitive dup; "HELLO world" is new.
        assert out.query_variants == ["Hello", "HELLO world"]

    def test_tolerates_fenced_json(self):
        body = "```json\n" + json.dumps({"rewrites": ["a", "b"]}) + "\n```"
        rw = _rewriter([body])
        out = rw.rewrite("q", strategy="multi_query", k=2)
        assert out.rewrites == ["a", "b"]

    def test_tolerates_chatty_preamble(self):
        body = 'Sure thing! Here you go:\n{"rewrites": ["x", "y"]}\n\nHope that helps.'
        rw = _rewriter([body])
        out = rw.rewrite("q", strategy="multi_query", k=2)
        assert out.rewrites == ["x", "y"]


# --------------------------------------------------------------------------- #
# HyDE
# --------------------------------------------------------------------------- #

class TestHyde:
    def test_parses_passage(self):
        body = json.dumps({"passage": "This is a plausible answer paragraph."})
        rw = _rewriter([body])
        out = rw.rewrite("why the sky blue?", strategy="hyde")
        assert out.hyde_passage == "This is a plausible answer paragraph."
        assert out.rewrites == []
        assert out.fallback_reason == ""

    def test_empty_passage_becomes_empty_string(self):
        body = json.dumps({"passage": ""})
        rw = _rewriter([body])
        out = rw.rewrite("q", strategy="hyde")
        assert out.hyde_passage == ""


# --------------------------------------------------------------------------- #
# Combined strategy
# --------------------------------------------------------------------------- #

class TestBoth:
    def test_single_call_returns_both(self):
        body = json.dumps({
            "rewrites": ["paraphrase one"],
            "passage": "A factual answer paragraph.",
        })
        rw = _rewriter([body])
        out = rw.rewrite("q", strategy="both", k=1)
        assert out.rewrites == ["paraphrase one"]
        assert out.hyde_passage == "A factual answer paragraph."
        # one LLM call for the full payload
        assert rw.llm.complete.call_count == 1


# --------------------------------------------------------------------------- #
# Failure modes — rewriter must NEVER raise
# --------------------------------------------------------------------------- #

class TestFallback:
    def test_llm_exception_yields_fallback(self):
        rw = _rewriter(raise_on_call=True)
        out = rw.rewrite("q", strategy="multi_query", k=2)
        assert out.rewrites == []
        assert out.hyde_passage == ""
        assert "LLMClientError" in out.fallback_reason
        # query_variants still contains the original so callers can blindly use it.
        assert out.query_variants == ["q"]

    def test_unparseable_llm_output_yields_empty_but_no_raise(self):
        body = "not json at all — just prose"
        rw = _rewriter([body])
        out = rw.rewrite("q", strategy="multi_query", k=2)
        assert out.rewrites == []
        assert out.fallback_reason == ""   # parse failure is silent (best-effort)

    def test_missing_rewrites_key_defaults_empty(self):
        body = json.dumps({"other_key": "ignored"})
        rw = _rewriter([body])
        out = rw.rewrite("q", strategy="multi_query", k=2)
        assert out.rewrites == []

    def test_unknown_strategy_falls_through(self):
        rw = _rewriter()
        out = rw.rewrite("q", strategy="nope", k=2)  # type: ignore[arg-type]
        assert out.rewrites == []
        assert "unknown strategy" in out.fallback_reason
        assert rw.llm.complete.call_count == 0


# --------------------------------------------------------------------------- #
# Smoke
# --------------------------------------------------------------------------- #

def test_query_variants_strips_whitespace_and_empties():
    body = json.dumps({"rewrites": ["  padded  ", "", "real rewrite"]})
    rw = _rewriter([body])
    out = rw.rewrite("original", strategy="multi_query", k=3)
    assert out.query_variants == ["original", "padded", "real rewrite"]
