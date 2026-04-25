"""Map golden-set `prior_turns` to rewriter history tuples."""

from __future__ import annotations

from kb.eval.types import GoldenExample


def turns_to_conversation_history(ex: GoldenExample) -> list[tuple[str, str]]:
    """(user, assistant) pairs in order."""
    out: list[tuple[str, str]] = []
    ts = ex.prior_turns
    i = 0
    while i + 1 < len(ts):
        if ts[i].role == "user" and ts[i + 1].role == "assistant":
            out.append((ts[i].content, ts[i + 1].content))
            i += 2
        else:
            i += 1
    return out
