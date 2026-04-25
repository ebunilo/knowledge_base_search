"""
Session data contracts.

Two shapes:

    * `ConversationTurn` — one (question → answer) round, with the
      shrunken metadata the rewriter and audit trail need. Deliberately
      a slim subset of `GenerationResult` — embedding the full result
      would balloon Redis memory and force every consumer to parse
      details they don't care about.

    * `Session` — owner + TTL-managed list of turns.

Both are Pydantic BaseModel, so JSON round-trips are safe through
`model_dump_json()` / `model_validate_json()`. The Redis store relies on
that.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def _utcnow_iso() -> str:
    """ISO-8601, second-precision, UTC. Stable across hosts."""
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


class ConversationTurn(BaseModel):
    """One conversational round.

    The rewriter only needs `question` + `answer` for coref-resolution.
    Everything else is for audit / replay (CLI `kb sessions show`).
    """

    # The user's raw question for this turn. We persist BOTH this and
    # the resolved form so an audit can show how coref was resolved.
    question: str
    # Coref-resolved, self-contained question that retrieval actually
    # ran on. Equals `question` when no rewrite happened.
    resolved_question: str = ""
    answer: str = ""
    # Per-turn citation IDs (parent_ids of cited hits) — keeps the turn
    # under ~1 KB even when the original answer cited many sources.
    cited_parent_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    refusal_reason: Optional[str] = None
    created_at: str = Field(default_factory=_utcnow_iso)


class Session(BaseModel):
    """Server-side conversation state.

    Identified by an unguessable random UUID. Always bound to a
    `user_id` — `SessionManager.get()` refuses to return a session that
    a different user is trying to read.
    """

    session_id: str
    user_id: str
    created_at: str = Field(default_factory=_utcnow_iso)
    last_used_at: str = Field(default_factory=_utcnow_iso)
    turns: list[ConversationTurn] = Field(default_factory=list)

    def touch(self) -> None:
        """Mark the session as just-used. Caller is responsible for
        persisting (the manager does)."""
        self.last_used_at = _utcnow_iso()

    def append(self, turn: ConversationTurn, *, max_turns: int) -> None:
        """Append a turn and drop the oldest if we exceed the cap."""
        self.turns.append(turn)
        if max_turns > 0 and len(self.turns) > max_turns:
            self.turns = self.turns[-max_turns:]
        self.touch()
