"""
Tests for `kb.sessions.store.RedisSessionStore`.

We use fakeredis so the suite is fast and offline. The store wraps a
`redis.Redis` instance; fakeredis is API-compatible.
"""

from __future__ import annotations

import fakeredis
import pytest

from kb.sessions.store import RedisSessionStore, SessionStoreError
from kb.sessions.types import ConversationTurn, Session
from kb.settings import Settings


def _store(ttl: int = 60) -> RedisSessionStore:
    settings = Settings(
        openai_api_key="test",
        session_ttl_seconds=ttl,
        session_key_prefix="kb:sess:",
    )
    return RedisSessionStore(settings, client=fakeredis.FakeRedis(decode_responses=True))


def _session(sid: str = "s1", uid: str = "u1") -> Session:
    return Session(session_id=sid, user_id=uid)


# --------------------------------------------------------------------------- #
# CRUD basics
# --------------------------------------------------------------------------- #

class TestCrud:
    def test_save_then_get_round_trips(self):
        store = _store()
        store.save(_session())
        out = store.get("s1")
        assert out is not None
        assert out.session_id == "s1"
        assert out.user_id == "u1"

    def test_get_unknown_returns_none(self):
        store = _store()
        assert store.get("nope") is None

    def test_get_refreshes_ttl(self):
        store = _store(ttl=120)
        store.save(_session())
        # Bake the key with a short TTL, then GET — TTL should bump.
        store._client.expire(store._key("s1"), 5)
        assert store._client.ttl(store._key("s1")) == 5
        store.get("s1")
        assert store._client.ttl(store._key("s1")) == 120

    def test_save_sets_ttl(self):
        store = _store(ttl=42)
        store.save(_session())
        assert store._client.ttl(store._key("s1")) == 42

    def test_delete_returns_true_when_present(self):
        store = _store()
        store.save(_session())
        assert store.delete("s1") is True
        assert store.get("s1") is None

    def test_delete_returns_false_when_absent(self):
        store = _store()
        assert store.delete("ghost") is False


# --------------------------------------------------------------------------- #
# Atomic append (WATCH/MULTI)
# --------------------------------------------------------------------------- #

class TestAppendAtomic:
    def test_append_persists_turn(self):
        store = _store()
        store.save(_session())
        store.append_turn_atomic(
            "s1",
            lambda s: s.append(
                ConversationTurn(question="hi", answer="hello"),
                max_turns=10,
            ),
        )
        loaded = store.get("s1")
        assert len(loaded.turns) == 1
        assert loaded.turns[0].question == "hi"

    def test_append_to_missing_session_raises(self):
        store = _store()
        with pytest.raises(SessionStoreError, match="not found"):
            store.append_turn_atomic("ghost", lambda s: None)


# --------------------------------------------------------------------------- #
# SCAN — list_keys
# --------------------------------------------------------------------------- #

class TestList:
    def test_list_only_returns_session_keys(self):
        store = _store()
        store.save(_session("a"))
        store.save(_session("b"))
        # Pollute the namespace with an unrelated key — must NOT show up.
        store._client.set("not-a-session", "x")
        ids = store.list_keys()
        assert set(ids) == {"a", "b"}

    def test_list_respects_limit(self):
        store = _store()
        for i in range(5):
            store.save(_session(f"s{i}"))
        ids = store.list_keys(limit=2)
        assert len(ids) == 2


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

def test_ping_returns_true_when_healthy():
    store = _store()
    assert store.ping() is True
