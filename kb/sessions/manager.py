"""
SessionManager — opinionated facade over RedisSessionStore.

What this layer adds beyond raw store CRUD:

    * `get_or_create(session_id, user_id)` — create-on-first-use, the
      common conversational pattern.
    * Ownership check — refuses to return a session that doesn't
      belong to the requesting user. Defence in depth: session IDs are
      random UUIDs (unguessable), but binding to a `user_id` means a
      leaked ID still can't be used cross-user.
    * Bounded turn history — drops the oldest turn when count exceeds
      `session_max_turns`. Bounds Redis memory and rewriter prompt
      cost.
    * Graceful degrade — Redis outages turn into "no session" + a
      warning log, rather than failing the whole `kb ask` call. The
      generator continues stateless.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from kb.sessions.store import RedisSessionStore, SessionStoreError
from kb.sessions.types import ConversationTurn, Session
from kb.settings import Settings, get_settings


logger = logging.getLogger(__name__)


class SessionNotFoundError(Exception):
    """Raised when a session ID is supplied but doesn't exist (or expired)."""


class SessionOwnershipError(Exception):
    """Raised when a user tries to read another user's session."""


class SessionManager:
    """High-level CRUD with ownership + bounded history."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: Optional[RedisSessionStore] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or RedisSessionStore(self.settings)
        self.max_turns = self.settings.session_max_turns

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def new_session_id(self) -> str:
        """Generate a fresh, unguessable session ID."""
        return uuid.uuid4().hex

    def get_or_create(
        self,
        *,
        session_id: Optional[str],
        user_id: str,
    ) -> Session:
        """
        Resolve a session for `user_id`.

        * `session_id is None` → create a new session.
        * `session_id` exists → return it (after ownership check).
        * `session_id` is supplied but missing/expired → create a new
          session WITH THAT ID. This matches client expectations: the
          client supplies a stable ID, and the server materialises it
          on first use.
        """
        if session_id is None:
            return self._create(self.new_session_id(), user_id=user_id)

        try:
            existing = self.store.get(session_id)
        except SessionStoreError as exc:
            logger.warning("session store read failed: %s — running stateless", exc)
            # Synthetic in-memory session so the caller can still
            # answer; nothing will be persisted.
            return Session(session_id=session_id, user_id=user_id)

        if existing is None:
            return self._create(session_id, user_id=user_id)

        if existing.user_id != user_id:
            raise SessionOwnershipError(
                f"session {session_id!r} belongs to a different user"
            )
        return existing

    def get(self, *, session_id: str, user_id: str) -> Session:
        """Strict load — raises if missing or wrong owner."""
        try:
            existing = self.store.get(session_id)
        except SessionStoreError as exc:
            raise SessionNotFoundError(
                f"failed to load session {session_id!r}: {exc}"
            ) from exc
        if existing is None:
            raise SessionNotFoundError(f"session {session_id!r} not found")
        if existing.user_id != user_id:
            raise SessionOwnershipError(
                f"session {session_id!r} belongs to a different user"
            )
        return existing

    def append_turn(
        self,
        *,
        session_id: str,
        user_id: str,
        turn: ConversationTurn,
    ) -> Session:
        """
        Atomically append a turn, refresh TTL.

        If the session disappeared between the question and the answer
        (rare — TTL race), we re-create it with this turn so the user
        doesn't lose their answer; we log a warning.
        """
        max_turns = self.max_turns

        def _apply(session: Session) -> None:
            if session.user_id != user_id:
                raise SessionOwnershipError(
                    f"session {session_id!r} belongs to a different user"
                )
            session.append(turn, max_turns=max_turns)

        try:
            return self.store.append_turn_atomic(session_id, _apply)
        except SessionStoreError as exc:
            # Likely "session not found" due to TTL expiry between
            # ask() start and end. Recreate.
            logger.warning(
                "atomic append failed for session %s (%s) — recreating",
                session_id, exc,
            )
            session = self._create(session_id, user_id=user_id)
            session.append(turn, max_turns=max_turns)
            try:
                self.store.save(session)
            except SessionStoreError as save_exc:
                logger.warning(
                    "failed to persist recreated session %s: %s",
                    session_id, save_exc,
                )
            return session

    def delete(self, *, session_id: str, user_id: str) -> bool:
        """Delete with ownership check (no-op if already gone)."""
        try:
            existing = self.store.get(session_id)
        except SessionStoreError as exc:
            raise SessionNotFoundError(
                f"failed to load session {session_id!r}: {exc}"
            ) from exc
        if existing is None:
            return False
        if existing.user_id != user_id:
            raise SessionOwnershipError(
                f"session {session_id!r} belongs to a different user"
            )
        return self.store.delete(session_id)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _create(self, session_id: str, *, user_id: str) -> Session:
        session = Session(session_id=session_id, user_id=user_id)
        try:
            self.store.save(session)
        except SessionStoreError as exc:
            logger.warning(
                "failed to persist new session %s: %s — running stateless",
                session_id, exc,
            )
        return session
