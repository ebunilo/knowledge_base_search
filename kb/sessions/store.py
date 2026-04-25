"""
Redis-backed session store.

Storage layout:

    KEY  := f"{prefix}{session_id}"
    VAL  := UTF-8 JSON of `Session` (model_dump_json)

We use a single key per session — Pydantic JSON dumps + atomic SET / GET
beats per-turn list semantics for our access pattern (read whole
session, modify, write back). Sessions are bounded
(`session_max_turns × ~2KB`) so the blob stays small.

TTL semantics:

    Every read AND every write applies a fresh `EXPIRE`. So an active
    session is kept alive as long as someone touches it within
    `ttl_seconds`. An idle session evicts itself.

Concurrency:

    The manager is the only writer per session within one process; for
    cross-process safety we use Redis WATCH / MULTI on append. (One
    user typing in two browser tabs is the realistic concurrency
    case here.)

Errors:

    All Redis exceptions are wrapped in `SessionStoreError` so callers
    don't need to know about `redis.exceptions.*`. The manager turns
    these into graceful degrades.
"""

from __future__ import annotations

import logging
from typing import Optional

import redis
from redis.exceptions import RedisError, WatchError

from kb.sessions.types import Session
from kb.settings import Settings, get_settings


logger = logging.getLogger(__name__)


class SessionStoreError(Exception):
    """Wraps Redis-level failures so callers can degrade gracefully."""


class RedisSessionStore:
    """Thin wrapper over `redis.Redis` for full-session JSON blobs."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Optional[redis.Redis] = None,
    ) -> None:
        self.settings = settings or get_settings()
        # Tests inject a `fakeredis.FakeRedis` (or a real one) here.
        self._client = client or self._build_client()
        self.prefix = self.settings.session_key_prefix
        self.ttl_seconds = self.settings.session_ttl_seconds

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get(self, session_id: str) -> Optional[Session]:
        """Load a session, refreshing its TTL. None if missing."""
        key = self._key(session_id)
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            session = Session.model_validate_json(_to_str(raw))
            # Rolling TTL — touch on read.
            self._client.expire(key, self.ttl_seconds)
            return session
        except (RedisError, ValueError) as exc:
            raise SessionStoreError(
                f"failed to load session {session_id!r}: {exc}"
            ) from exc

    def save(self, session: Session) -> None:
        """Write the session and (re)set its TTL atomically."""
        key = self._key(session.session_id)
        payload = session.model_dump_json()
        try:
            # SET with EX is atomic — value + ttl in one round-trip.
            self._client.set(key, payload, ex=self.ttl_seconds)
        except RedisError as exc:
            raise SessionStoreError(
                f"failed to save session {session.session_id!r}: {exc}"
            ) from exc

    def append_turn_atomic(
        self,
        session_id: str,
        mutate: "callable[[Session], None]",
        *,
        max_retries: int = 3,
    ) -> Session:
        """
        Read-modify-write a session atomically across processes.

        Used for `SessionManager.append_turn()`. Two browser tabs that
        ask in parallel under the same session shouldn't lose a turn —
        WATCH/MULTI ensures the second writer sees the first writer's
        version.

        `mutate(session)` is called with the latest stored Session and
        is expected to mutate it in place.
        """
        key = self._key(session_id)
        try:
            with self._client.pipeline() as pipe:
                for attempt in range(max_retries):
                    try:
                        pipe.watch(key)
                        raw = pipe.get(key)
                        if raw is None:
                            pipe.unwatch()
                            raise SessionStoreError(
                                f"session {session_id!r} not found "
                                "(may have expired)"
                            )
                        session = Session.model_validate_json(_to_str(raw))
                        mutate(session)
                        pipe.multi()
                        pipe.set(
                            key, session.model_dump_json(),
                            ex=self.ttl_seconds,
                        )
                        pipe.execute()
                        return session
                    except WatchError:
                        # Someone else wrote between WATCH and EXEC; retry.
                        logger.debug(
                            "session %s: WATCH conflict on attempt %d, retrying",
                            session_id, attempt + 1,
                        )
                        continue
                raise SessionStoreError(
                    f"session {session_id!r}: too many concurrent writes "
                    f"({max_retries} retries exhausted)"
                )
        except RedisError as exc:
            raise SessionStoreError(
                f"failed to append turn to {session_id!r}: {exc}"
            ) from exc

    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        try:
            return bool(self._client.delete(self._key(session_id)))
        except RedisError as exc:
            raise SessionStoreError(
                f"failed to delete session {session_id!r}: {exc}"
            ) from exc

    def list_keys(self, *, limit: int = 100) -> list[str]:
        """
        SCAN session keys (without their values) for the CLI.

        Returns RAW session IDs (prefix stripped). `limit` is a cursor
        cap, not a strict ceiling — Redis SCAN is best-effort. Used
        only by `kb sessions list` which is admin-only.
        """
        try:
            out: list[str] = []
            for raw in self._client.scan_iter(
                match=f"{self.prefix}*", count=min(limit * 2, 500),
            ):
                out.append(_to_str(raw)[len(self.prefix):])
                if len(out) >= limit:
                    break
            return out
        except RedisError as exc:
            raise SessionStoreError(f"SCAN failed: {exc}") from exc

    def ping(self) -> bool:
        """Health check — used by `kb health`."""
        try:
            return bool(self._client.ping())
        except RedisError:
            return False

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    def _build_client(self) -> redis.Redis:
        s = self.settings
        if s.redis_url:
            return redis.Redis.from_url(s.redis_url, decode_responses=True)
        return redis.Redis(
            host=s.redis_host, port=s.redis_port,
            password=s.redis_password or None,
            decode_responses=True,
        )


# --------------------------------------------------------------------------- #
# Helper — fakeredis returns bytes by default
# --------------------------------------------------------------------------- #

def _to_str(raw) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)
