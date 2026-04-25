"""
Query rewriting — multi-query, HyDE, conversation coreference, step-back.

Original (Slice 2):
    1. Multi-query expansion — paraphrase the question k ways.
    2. HyDE — generate a hypothetical answer to embed alongside.

Phase 3 · Slice 2B additions:
    3. Coreference resolution — given prior turns of a conversation,
       rewrite a follow-up question into a self-contained one.
       Pronouns and implicit references that BM25 / dense embeddings
       can't ground (e.g. "What about *its* limits?") become explicit.
    4. Step-back — generate a single broader version of the question
       (Zheng et al. 2023, https://arxiv.org/abs/2310.06117) to
       improve retrieval coverage when the original is too specific.

All four are combined into ONE LLM call. The prompt is built
dynamically from whichever capabilities the caller asked for, and the
JSON response carries only the requested fields.

Privacy note:
    Conversation history is sent to the rewriter's hosted lane along
    with the current question. For deployments that treat user queries
    themselves as sensitive, flip `rewrite_lane` in settings to
    self_hosted_only. The private collection's contents never touch
    the rewriter — only the user's prior questions and answers do.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

from kb.enrichment.llm_client import LLMClient, LLMClientError
from kb.settings import Settings, get_settings
from kb.types import SensitivityLane


logger = logging.getLogger(__name__)


RewriteStrategy = Literal["off", "multi_query", "hyde", "both"]


@dataclass
class RewriteResult:
    """What the rewriter produced. Safe to consume even on LLM failure."""
    strategy: RewriteStrategy
    original: str
    # Coref-resolved canonical query. Empty iff no history was supplied
    # OR the LLM returned no resolution.
    resolved: str = ""
    rewrites: list[str] = field(default_factory=list)   # paraphrases (excludes canonical)
    # Single broader / step-back variant. Empty when not requested or
    # the LLM omitted it.
    stepback: str = ""
    hyde_passage: str = ""
    llm_ms: int = 0
    llm_provider: str = ""
    llm_model: str = ""
    fallback_reason: str = ""                            # empty iff LLM succeeded

    @property
    def canonical(self) -> str:
        """The query retrieval should anchor on.

        When coref resolution ran, the resolved form replaces the raw
        query — that's the whole point. Otherwise the raw query stands.
        """
        return (self.resolved or self.original).strip()

    @property
    def query_variants(self) -> list[str]:
        """Canonical + rewrites + step-back, deduplicated, in order."""
        out: list[str] = []
        seen: set[str] = set()
        for q in [self.canonical, *self.rewrites, self.stepback]:
            qs = (q or "").strip()
            if not qs:
                continue
            key = qs.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(qs)
        return out


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #

_SYSTEM_BASE = (
    "You are a retrieval assistant. Rewrite user questions to improve "
    "search recall against an enterprise knowledge base. Be faithful to "
    "the original intent. Output STRICT JSON only — no prose, no "
    "markdown fences."
)


def _format_history(history: Iterable[tuple[str, str]]) -> str:
    """Render `(question, answer)` pairs in chronological order."""
    lines: list[str] = ["Conversation so far (oldest first):"]
    for i, (q, a) in enumerate(history, start=1):
        # Trim each turn so the prompt budget stays predictable. The
        # exact figures here are heuristic — Slice 2C will calibrate.
        q_short = (q or "").strip()
        a_short = (a or "").strip()
        if len(a_short) > 280:
            a_short = a_short[:280].rstrip() + "…"
        lines.append(f"  {i}. Q: {q_short}")
        if a_short:
            lines.append(f"     A: {a_short}")
    return "\n".join(lines)


def _build_prompt(
    *,
    query: str,
    history: list[tuple[str, str]],
    multi_query: bool,
    multi_query_k: int,
    hyde: bool,
    stepback: bool,
) -> str:
    """
    Build the user prompt for ONE LLM call covering every requested
    capability. Only sections relevant to enabled flags appear.
    """
    sections: list[str] = []
    if history:
        sections.append(_format_history(history))
        sections.append(
            "The user's NEW question may use pronouns or refer to "
            "prior turns implicitly. Rewrite it to be a fully "
            "self-contained question that someone seeing only the "
            "rewrite would understand. Preserve intent. If the "
            "question is already self-contained, return it unchanged."
        )
    instructions: list[str] = []
    if multi_query:
        instructions.append(
            f"Generate {max(1, multi_query_k)} alternative phrasings "
            "of the question. Each should preserve intent but use "
            "different wording (synonyms, expansion of acronyms, "
            "more/less specific framing)."
        )
    if stepback:
        instructions.append(
            "Generate ONE step-back question — a more general version "
            "of the user's question that would surface broader "
            "background documents. Keep it answerable from a generic "
            "knowledge base."
        )
    if hyde:
        instructions.append(
            "Write a short, factual passage (1 to 3 sentences) that "
            "would plausibly answer the question, written in the "
            "neutral declarative style of an enterprise knowledge "
            "base article."
        )
    if instructions:
        sections.append("\n\n".join(instructions))

    json_keys: list[str] = []
    if history:
        json_keys.append('"resolved_query": "<self-contained question>"')
    if multi_query:
        json_keys.append('"rewrites": ["<q1>", "<q2>", ...]')
    if stepback:
        json_keys.append('"stepback_query": "<broader question>"')
    if hyde:
        json_keys.append('"passage": "<text>"')
    sections.append("Output JSON:\n{" + ", ".join(json_keys) + "}")
    sections.append(f"Question: {query}")
    return "\n\n".join(sections)


# --------------------------------------------------------------------------- #
# Rewriter
# --------------------------------------------------------------------------- #

class QueryRewriter:
    """LLM-backed query rewriter. Always safe to call — never raises."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)

    def rewrite(
        self,
        query: str,
        *,
        strategy: RewriteStrategy = "off",
        k: int = 2,
        history: list[tuple[str, str]] | None = None,
        stepback: bool = False,
    ) -> RewriteResult:
        """
        One LLM call covering every requested capability. Returns a
        populated RewriteResult on success; on failure returns a result
        with `fallback_reason` set and all enrichment fields empty.

        Backward compatibility: callers that passed only
        `strategy=...` and `k=...` get the same behaviour as Slice 2.
        """
        original = (query or "").strip()
        history = history or []

        # No work to do — short-circuit.
        wants_multi = strategy in {"multi_query", "both"}
        wants_hyde = strategy in {"hyde", "both"}
        wants_coref = bool(history)
        wants_anything = wants_multi or wants_hyde or wants_coref or stepback

        if not original or strategy == "off" and not wants_coref and not stepback:
            return RewriteResult(strategy=strategy, original=original)
        if strategy not in {"off", "multi_query", "hyde", "both"}:
            logger.warning("unknown rewrite strategy %r — skipping", strategy)
            return RewriteResult(
                strategy=strategy, original=original,
                fallback_reason=f"unknown strategy {strategy!r}",
            )
        if not wants_anything:
            return RewriteResult(strategy=strategy, original=original)

        prompt = _build_prompt(
            query=original,
            history=history,
            multi_query=wants_multi,
            multi_query_k=k,
            hyde=wants_hyde,
            stepback=stepback,
        )

        try:
            text, meta = self._call(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("query rewrite failed: %s — falling back", exc)
            return RewriteResult(
                strategy=strategy, original=original,
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )

        parsed = _parse_json_loose(text) or {}
        resolved = (
            str(parsed.get("resolved_query", "") or "").strip()
            if wants_coref else ""
        )
        rewrites = (
            _as_str_list(parsed.get("rewrites", []))[:k]
            if wants_multi else []
        )
        sb = (
            str(parsed.get("stepback_query", "") or "").strip()
            if stepback else ""
        )
        passage = (
            str(parsed.get("passage", "") or "").strip()
            if wants_hyde else ""
        )

        # If the LLM ignored the coref instruction, fall back to the
        # raw query — never lose the question.
        if wants_coref and not resolved:
            resolved = original

        return RewriteResult(
            strategy=strategy,
            original=original,
            resolved=resolved,
            rewrites=rewrites,
            stepback=sb,
            hyde_passage=passage,
            **meta,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _call(self, prompt: str) -> tuple[str, dict]:
        try:
            result = self.llm.complete(
                prompt=prompt,
                system=_SYSTEM_BASE,
                lane=SensitivityLane.HOSTED_OK,
                max_tokens=500,
                temperature=0.2,
                json_mode=True,
            )
            return result.text, {
                "llm_ms": result.latency_ms,
                "llm_provider": result.provider,
                "llm_model": result.model,
            }
        except LLMClientError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMClientError(f"rewrite LLM call failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# JSON parsing helpers — tolerate chatty models
# --------------------------------------------------------------------------- #

def _parse_json_loose(text: str) -> dict | None:
    """Best-effort: parse either a bare JSON object or the first {...} in text."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _as_str_list(v: object) -> list[str]:
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for item in v:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out
