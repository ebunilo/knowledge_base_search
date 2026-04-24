"""
Tests for `kb.generation.generator.Generator`.

We mock out the Retriever and the LLMClient so these are offline. They
verify the full ask() and ask_stream() flow end-to-end:

    * Refusal when no hits.
    * Refusal when top hit below score floor.
    * Happy path: answer + citations + diagnostics.
    * Lane selection: any self_hosted_only hit forces the self-hosted lane.
    * Streaming yields start → token* → done.
    * LLM failure during ask() degrades to a refusal (not a raise).
    * LLM init failure during ask_stream() emits a refused event.
"""

from __future__ import annotations

from typing import Iterator
from unittest.mock import MagicMock

import pytest

from kb.enrichment.llm_client import (
    CompletionResult,
    LLMClientError,
    StreamingCompletion,
)
from kb.generation.generator import Generator
from kb.generation.types import GenerationConfig
from kb.retrieval.types import RetrievalConfig, RetrievalHit, RetrievalResult, UserContext
from kb.settings import Settings
from kb.types import SensitivityLane


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _hit(
    *,
    parent_id: str = "p1",
    title: str = "Doc",
    sensitivity: str = "hosted_ok",
    score: float = 0.9,
    content: str = "Some parent content describing the thing.",
) -> RetrievalHit:
    return RetrievalHit(
        child_id=parent_id + "_c0",
        parent_id=parent_id,
        document_id="doc1",
        source_id="src1",
        source_uri=f"file:///{parent_id}.md",
        title=title,
        section_path="intro",
        content=content[:120],
        parent_content=content,
        score=score,
        sensitivity=sensitivity,
        visibility="public",
    )


def _retrieval(hits: list[RetrievalHit]) -> RetrievalResult:
    return RetrievalResult(
        query="q",
        user_id="u",
        hits=hits,
        collections_searched=["public_v1"],
        final_hits=len(hits),
    )


def _streaming(chunks: list[str], provider="openai", model="gpt-4o-mini"):
    """Build a StreamingCompletion that yields the provided chunks."""
    def _gen():
        yield from chunks
    return StreamingCompletion(provider=provider, model=model, _chunks=_gen())


def _generator(
    *,
    hits: list[RetrievalHit] | None = None,
    completion_text: str | None = None,
    completion_raises: Exception | None = None,
    stream_chunks: list[str] | None = None,
    stream_raises: Exception | None = None,
) -> Generator:
    """Build a Generator with retriever + LLM mocked."""
    retriever = MagicMock()
    retriever.retrieve.return_value = _retrieval(hits or [])

    llm = MagicMock()
    if completion_raises is not None:
        llm.complete.side_effect = completion_raises
    elif completion_text is not None:
        llm.complete.return_value = CompletionResult(
            text=completion_text, provider="openai",
            model="gpt-4o-mini", latency_ms=42,
        )

    if stream_raises is not None:
        llm.stream.side_effect = stream_raises
    elif stream_chunks is not None:
        llm.stream.return_value = _streaming(stream_chunks)

    return Generator(
        settings=Settings(openai_api_key="test"),
        retriever=retriever,
        llm=llm,
    )


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #

class TestRefusals:
    def test_no_hits_returns_refusal_without_calling_llm(self):
        gen = _generator(hits=[], completion_text="should not be called")
        result = gen.ask("how do I deploy?")
        assert result.refused is True
        assert result.refusal_reason == "no_hits"
        assert result.answer  # non-empty refusal text
        assert gen.llm.complete.call_count == 0

    def test_low_confidence_refusal(self):
        # Top hit score 0.05; threshold 0.5 → refuse.
        gen = _generator(
            hits=[_hit(score=0.05)],
            completion_text="should not be called",
        )
        result = gen.ask(
            "x",
            generation_config=GenerationConfig(min_score_threshold=0.5),
        )
        assert result.refused is True
        assert result.refusal_reason == "low_confidence"
        assert gen.llm.complete.call_count == 0

    def test_refusal_passes_through_retrieval(self):
        gen = _generator(hits=[])
        result = gen.ask("x")
        assert result.retrieval is not None
        assert result.retrieval.hits == []


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #

class TestAsk:
    def test_returns_answer_with_citations(self):
        hits = [_hit(parent_id="p1", title="A"), _hit(parent_id="p2", title="B")]
        gen = _generator(hits=hits, completion_text="Answer body, see [1] and [2].")
        result = gen.ask("q")

        assert result.refused is False
        assert "[1]" in result.answer and "[2]" in result.answer
        assert [c.marker for c in result.citations] == [1, 2]
        assert result.citations[0].title == "A"
        assert result.citations[1].title == "B"
        assert result.provider == "openai"
        assert result.model == "gpt-4o-mini"
        assert result.lane == SensitivityLane.HOSTED_OK.value

    def test_invokes_llm_with_correct_lane(self):
        hits = [_hit(parent_id="p1")]
        gen = _generator(hits=hits, completion_text="answer [1].")
        gen.ask("q")

        kwargs = gen.llm.complete.call_args.kwargs
        assert kwargs["lane"] == SensitivityLane.HOSTED_OK
        assert "QUESTION:" in kwargs["prompt"]
        assert "[1] Title:" in kwargs["prompt"]

    def test_self_hosted_lane_when_any_hit_sensitive(self):
        hits = [
            _hit(parent_id="p1", sensitivity="hosted_ok"),
            _hit(parent_id="p2", sensitivity="self_hosted_only"),
        ]
        gen = _generator(hits=hits, completion_text="answer [1].")
        gen.ask("q")

        kwargs = gen.llm.complete.call_args.kwargs
        assert kwargs["lane"] == SensitivityLane.SELF_HOSTED_ONLY

    def test_uncited_hits_surface_in_result(self):
        hits = [_hit(parent_id="p1"), _hit(parent_id="p2"), _hit(parent_id="p3")]
        gen = _generator(hits=hits, completion_text="see [2] only.")
        result = gen.ask("q")
        assert result.uncited_hits == [1, 3]
        assert result.invalid_markers == []

    def test_invalid_marker_surfaces_in_result(self):
        hits = [_hit(parent_id="p1")]
        gen = _generator(hits=hits, completion_text="see [9].")
        result = gen.ask("q")
        assert result.citations == []
        assert result.invalid_markers == [9]

    def test_llm_failure_returns_refusal(self):
        gen = _generator(
            hits=[_hit(parent_id="p1")],
            completion_raises=LLMClientError("all providers down"),
        )
        result = gen.ask("q")
        assert result.refused is True
        assert result.refusal_reason == "llm_unavailable"


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #

class TestStream:
    def test_emits_start_token_done(self):
        hits = [_hit(parent_id="p1")]
        gen = _generator(hits=hits, stream_chunks=["Hel", "lo ", "[1]"])
        events = list(gen.ask_stream("q"))

        kinds = [e.kind for e in events]
        assert kinds[0] == "start"
        assert kinds[-1] == "done"
        # All token events between.
        assert all(k == "token" for k in kinds[1:-1])

        # Reconstruction.
        text = "".join(e.text for e in events if e.kind == "token")
        assert text == "Hello [1]"

        # Done event has full result with citations.
        done = events[-1]
        assert done.result is not None
        assert done.result.refused is False
        assert [c.marker for c in done.result.citations] == [1]

    def test_no_hits_emits_refused(self):
        gen = _generator(hits=[], stream_chunks=["should not be reached"])
        events = list(gen.ask_stream("q"))
        kinds = [e.kind for e in events]
        assert kinds == ["start", "refused"]
        assert events[1].result is not None
        assert events[1].result.refused is True
        # llm.stream must not be called when refusing.
        assert gen.llm.stream.call_count == 0

    def test_stream_init_failure_emits_refused(self):
        gen = _generator(
            hits=[_hit(parent_id="p1")],
            stream_raises=LLMClientError("init failed"),
        )
        events = list(gen.ask_stream("q"))
        kinds = [e.kind for e in events]
        # start (with partial), then refused.
        assert kinds == ["start", "refused"]
        assert events[1].result is not None
        assert events[1].result.refused is True
        assert events[1].result.refusal_reason == "llm_unavailable"

    def test_done_includes_diagnostics(self):
        hits = [_hit(parent_id="p1"), _hit(parent_id="p2")]
        gen = _generator(hits=hits, stream_chunks=["see [1]"])
        events = list(gen.ask_stream("q"))
        done = events[-1].result
        assert done is not None
        assert done.context_tokens > 0
        assert done.used_hit_count == 2
        assert done.uncited_hits == [2]
        assert done.provider == "openai"
        assert done.model == "gpt-4o-mini"
