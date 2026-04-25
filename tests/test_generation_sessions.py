"""
Tests for Generator session integration (Phase 3 · Slice 2B).

Verifies:
    * No session_id → identical behaviour to Slice 2A (no SessionManager call).
    * session_id supplied → manager.get_or_create + append_turn invoked.
    * conversation_history populated on RetrievalConfig before retrieve.
    * Refusals are persisted (so the rewriter can see them next turn).
    * SessionOwnershipError propagates out of ask().
    * Streaming carries session_id on `start` and `done`.
    * Caller-supplied conversation_history wins over session-derived.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import fakeredis
import pytest

from kb.enrichment.llm_client import CompletionResult, StreamingCompletion
from kb.generation.generator import Generator
from kb.generation.types import FaithfulnessReport
from kb.retrieval.types import (
    RetrievalConfig,
    RetrievalHit,
    RetrievalResult,
    UserContext,
)
from kb.sessions import (
    ConversationTurn,
    RedisSessionStore,
    SessionManager,
    SessionOwnershipError,
)
from kb.settings import Settings


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _settings() -> Settings:
    return Settings(openai_api_key="test", session_max_turns=10)


def _hit(parent_id: str = "p1") -> RetrievalHit:
    return RetrievalHit(
        child_id=parent_id + "_c0",
        parent_id=parent_id,
        document_id="d1",
        source_id="s1",
        source_uri="file:///x.md",
        title="Doc",
        content="parent content",
        parent_content="parent content",
        score=0.9,
        sensitivity="hosted_ok",
        visibility="public",
    )


def _retrieval(query: str = "q", hits=None, resolved=None) -> RetrievalResult:
    if hits is None:
        hits = [_hit()]
    return RetrievalResult(
        query=query,
        user_id="u1",
        hits=hits,
        collections_searched=["public_v1"],
        final_hits=len(hits),
        resolved_query=resolved,
    )


def _build(*, settings=None, sessions=None, completion_text="Answer [1].",
           stream_chunks=None, retrieval_result=None):
    settings = settings or _settings()
    retriever = MagicMock()
    retriever.retrieve.return_value = retrieval_result or _retrieval()

    llm = MagicMock()
    llm.complete.return_value = CompletionResult(
        text=completion_text, provider="openai", model="gpt-4o-mini", latency_ms=20,
    )
    if stream_chunks is not None:
        def _gen():
            yield from stream_chunks
        llm.stream.return_value = StreamingCompletion(
            provider="openai", model="gpt-4o-mini", _chunks=_gen(),
        )

    checker = MagicMock()
    checker.check.return_value = FaithfulnessReport()

    if sessions is None:
        store = RedisSessionStore(
            settings,
            client=fakeredis.FakeRedis(decode_responses=True),
        )
        sessions = SessionManager(settings, store=store)

    gen = Generator(
        settings=settings,
        retriever=retriever,
        llm=llm,
        faithfulness=checker,
        sessions=sessions,
    )
    return gen, retriever, sessions


# --------------------------------------------------------------------------- #
# No session_id → stateless
# --------------------------------------------------------------------------- #

def test_no_session_id_does_not_touch_session_manager():
    sessions = MagicMock(spec=SessionManager)
    gen, retriever, _ = _build(sessions=sessions)
    result = gen.ask("hello", user=UserContext(user_id="u1"))
    assert result.session_id is None
    sessions.get_or_create.assert_not_called()
    sessions.append_turn.assert_not_called()


# --------------------------------------------------------------------------- #
# session_id provided → load + append
# --------------------------------------------------------------------------- #

class TestSessionIntegration:
    def test_first_turn_creates_session_and_persists(self):
        gen, retriever, sessions = _build()
        result = gen.ask(
            "What is the gateway?",
            user=UserContext(user_id="u1"),
            session_id="abc",
        )
        assert result.session_id == "abc"
        # Session was materialised.
        s = sessions.get(session_id="abc", user_id="u1")
        assert len(s.turns) == 1
        assert s.turns[0].question == "What is the gateway?"

    def test_second_turn_passes_history_to_retriever(self):
        gen, retriever, sessions = _build()
        # Seed turn 1.
        sessions.get_or_create(session_id="s1", user_id="u1")
        sessions.append_turn(
            session_id="s1", user_id="u1",
            turn=ConversationTurn(question="Q1", answer="A1"),
        )
        # Turn 2.
        gen.ask("Q2", user=UserContext(user_id="u1"), session_id="s1")
        # Inspect the RetrievalConfig the retriever actually saw.
        call = retriever.retrieve.call_args
        cfg: RetrievalConfig = call.kwargs["config"]
        assert cfg.conversation_history == [("Q1", "A1")]

    def test_caller_supplied_history_overrides_session(self):
        gen, retriever, sessions = _build()
        sessions.get_or_create(session_id="s1", user_id="u1")
        sessions.append_turn(
            session_id="s1", user_id="u1",
            turn=ConversationTurn(question="seed", answer="seed-answer"),
        )
        rcfg = RetrievalConfig(
            conversation_history=[("override-q", "override-a")],
        )
        gen.ask("Q", user=UserContext(user_id="u1"),
                retrieval_config=rcfg, session_id="s1")
        call = retriever.retrieve.call_args
        cfg = call.kwargs["config"]
        assert cfg.conversation_history == [("override-q", "override-a")]

    def test_resolved_question_is_persisted_when_present(self):
        gen, retriever, sessions = _build(
            retrieval_result=_retrieval(
                query="What about its limits?",
                resolved="What are the rate limits of the gateway?",
            ),
        )
        gen.ask(
            "What about its limits?",
            user=UserContext(user_id="u1"),
            session_id="s1",
        )
        s = sessions.get(session_id="s1", user_id="u1")
        assert s.turns[0].question == "What about its limits?"
        assert s.turns[0].resolved_question == (
            "What are the rate limits of the gateway?"
        )


# --------------------------------------------------------------------------- #
# Refusal persistence
# --------------------------------------------------------------------------- #

class TestRefusalPersistence:
    def test_refusal_is_persisted(self):
        gen, retriever, sessions = _build(
            retrieval_result=_retrieval(hits=[]),
        )
        gen.ask("ghost", user=UserContext(user_id="u1"), session_id="s1")
        s = sessions.get(session_id="s1", user_id="u1")
        assert len(s.turns) == 1
        assert s.turns[0].refused is True
        # Refusal reason carried through.
        assert s.turns[0].refusal_reason == "no_hits"


# --------------------------------------------------------------------------- #
# Ownership
# --------------------------------------------------------------------------- #

def test_other_user_cannot_continue_session():
    gen, retriever, sessions = _build()
    sessions.get_or_create(session_id="s1", user_id="alice")
    with pytest.raises(SessionOwnershipError):
        gen.ask("Q", user=UserContext(user_id="bob"), session_id="s1")


# --------------------------------------------------------------------------- #
# Streaming carries session_id
# --------------------------------------------------------------------------- #

def test_stream_emits_session_id_on_start_and_done():
    gen, retriever, sessions = _build(stream_chunks=["A.", " B."])
    events = list(gen.ask_stream(
        "Q", user=UserContext(user_id="u1"), session_id="s1",
    ))
    starts = [e for e in events if e.kind == "start"]
    dones = [e for e in events if e.kind == "done"]
    assert len(starts) == 1 and len(dones) == 1
    assert starts[0].result.session_id == "s1"
    assert dones[0].result.session_id == "s1"
    # Append happened after the stream finished.
    s = sessions.get(session_id="s1", user_id="u1")
    assert len(s.turns) == 1


# --------------------------------------------------------------------------- #
# Bounded history
# --------------------------------------------------------------------------- #

def test_history_is_bounded_to_max_turns():
    settings = Settings(openai_api_key="test", session_max_turns=3)
    store = RedisSessionStore(
        settings, client=fakeredis.FakeRedis(decode_responses=True),
    )
    sessions = SessionManager(settings, store=store)
    gen, retriever, _ = _build(settings=settings, sessions=sessions)

    for i in range(5):
        gen.ask(
            f"Q{i}", user=UserContext(user_id="u1"), session_id="s1",
        )
    s = sessions.get(session_id="s1", user_id="u1")
    assert len(s.turns) == 3
    assert [t.question for t in s.turns] == ["Q2", "Q3", "Q4"]
