"""
Tests for `kb.sessions.manager.SessionManager`.

Covers create-on-first-use, ownership checks, bounded turn history,
and graceful degrade on store failures.
"""

from __future__ import annotations

import fakeredis
import pytest

from kb.sessions.manager import (
    SessionManager,
    SessionNotFoundError,
    SessionOwnershipError,
)
from kb.sessions.store import RedisSessionStore, SessionStoreError
from kb.sessions.types import ConversationTurn
from kb.settings import Settings


def _manager(*, max_turns: int = 10, ttl: int = 60) -> SessionManager:
    settings = Settings(
        openai_api_key="test",
        session_ttl_seconds=ttl,
        session_max_turns=max_turns,
    )
    store = RedisSessionStore(settings, client=fakeredis.FakeRedis(decode_responses=True))
    return SessionManager(settings, store=store)


# --------------------------------------------------------------------------- #
# get_or_create
# --------------------------------------------------------------------------- #

class TestGetOrCreate:
    def test_no_id_creates_fresh(self):
        mgr = _manager()
        s = mgr.get_or_create(session_id=None, user_id="u1")
        assert s.user_id == "u1"
        assert s.turns == []
        assert len(s.session_id) == 32   # uuid4 hex

    def test_known_id_returns_existing(self):
        mgr = _manager()
        s1 = mgr.get_or_create(session_id="abc", user_id="u1")
        s2 = mgr.get_or_create(session_id="abc", user_id="u1")
        assert s1.session_id == s2.session_id
        assert s1.created_at == s2.created_at

    def test_unknown_id_creates_at_that_id(self):
        """Client supplies a stable ID; server materialises on first use."""
        mgr = _manager()
        s = mgr.get_or_create(session_id="my-stable-id", user_id="u1")
        assert s.session_id == "my-stable-id"

    def test_wrong_owner_raises(self):
        mgr = _manager()
        mgr.get_or_create(session_id="s1", user_id="alice")
        with pytest.raises(SessionOwnershipError):
            mgr.get_or_create(session_id="s1", user_id="bob")


# --------------------------------------------------------------------------- #
# get (strict)
# --------------------------------------------------------------------------- #

class TestStrictGet:
    def test_missing_raises(self):
        mgr = _manager()
        with pytest.raises(SessionNotFoundError):
            mgr.get(session_id="ghost", user_id="u1")

    def test_wrong_owner_raises(self):
        mgr = _manager()
        mgr.get_or_create(session_id="s1", user_id="alice")
        with pytest.raises(SessionOwnershipError):
            mgr.get(session_id="s1", user_id="bob")

    def test_happy_path(self):
        mgr = _manager()
        mgr.get_or_create(session_id="s1", user_id="alice")
        s = mgr.get(session_id="s1", user_id="alice")
        assert s.user_id == "alice"


# --------------------------------------------------------------------------- #
# append_turn — bounded history
# --------------------------------------------------------------------------- #

class TestAppendTurn:
    def test_appends_turn_and_persists(self):
        mgr = _manager()
        mgr.get_or_create(session_id="s1", user_id="u1")
        mgr.append_turn(
            session_id="s1",
            user_id="u1",
            turn=ConversationTurn(question="q1", answer="a1"),
        )
        s = mgr.get(session_id="s1", user_id="u1")
        assert len(s.turns) == 1
        assert s.turns[0].question == "q1"

    def test_drops_oldest_when_over_cap(self):
        mgr = _manager(max_turns=3)
        mgr.get_or_create(session_id="s1", user_id="u1")
        for i in range(5):
            mgr.append_turn(
                session_id="s1", user_id="u1",
                turn=ConversationTurn(question=f"q{i}", answer=f"a{i}"),
            )
        s = mgr.get(session_id="s1", user_id="u1")
        assert len(s.turns) == 3
        assert [t.question for t in s.turns] == ["q2", "q3", "q4"]

    def test_other_user_cannot_append(self):
        mgr = _manager()
        mgr.get_or_create(session_id="s1", user_id="alice")
        with pytest.raises(SessionOwnershipError):
            mgr.append_turn(
                session_id="s1", user_id="bob",
                turn=ConversationTurn(question="q", answer="a"),
            )

    def test_recreates_when_session_expired(self):
        """If TTL eats the session mid-conversation, the user's answer
        still gets recorded — we recreate the session under the same ID."""
        mgr = _manager()
        mgr.get_or_create(session_id="s1", user_id="u1")
        # Forcibly evict to simulate TTL race.
        mgr.store._client.delete(mgr.store._key("s1"))
        s = mgr.append_turn(
            session_id="s1", user_id="u1",
            turn=ConversationTurn(question="late", answer="answer"),
        )
        assert len(s.turns) == 1
        assert s.turns[0].question == "late"


# --------------------------------------------------------------------------- #
# delete
# --------------------------------------------------------------------------- #

class TestDelete:
    def test_delete_happy_path(self):
        mgr = _manager()
        mgr.get_or_create(session_id="s1", user_id="u1")
        assert mgr.delete(session_id="s1", user_id="u1") is True
        with pytest.raises(SessionNotFoundError):
            mgr.get(session_id="s1", user_id="u1")

    def test_delete_missing_returns_false(self):
        mgr = _manager()
        assert mgr.delete(session_id="ghost", user_id="u1") is False

    def test_delete_wrong_owner_raises(self):
        mgr = _manager()
        mgr.get_or_create(session_id="s1", user_id="alice")
        with pytest.raises(SessionOwnershipError):
            mgr.delete(session_id="s1", user_id="bob")


# --------------------------------------------------------------------------- #
# Graceful degrade on store failure
# --------------------------------------------------------------------------- #

class _BrokenStore:
    """Drop-in store that raises on every call."""
    def get(self, _):
        raise SessionStoreError("redis down")
    def save(self, _):
        raise SessionStoreError("redis down")
    def delete(self, _):
        raise SessionStoreError("redis down")
    def append_turn_atomic(self, *_a, **_kw):
        raise SessionStoreError("redis down")


def test_get_or_create_runs_stateless_on_redis_failure(caplog):
    settings = Settings(openai_api_key="test")
    mgr = SessionManager(settings, store=_BrokenStore())  # type: ignore[arg-type]
    s = mgr.get_or_create(session_id="x", user_id="u1")
    # The manager returns a synthetic in-memory session so the answer
    # path can keep going. Nothing was persisted; that's the trade.
    assert s.session_id == "x"
    assert s.user_id == "u1"
    assert any("read failed" in r.message for r in caplog.records)
