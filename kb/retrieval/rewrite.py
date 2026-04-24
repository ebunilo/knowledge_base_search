"""
Query rewriting — multi-query expansion and HyDE.

Problem: a single user query rarely exercises the full semantic space of
the documents that should answer it. Two well-known mitigations:

    1. Multi-query expansion (Langchain's MultiQueryRetriever pattern)
       — use an LLM to paraphrase the user's question in k different
       ways; retrieve for each and union the results. Recall up,
       latency up (by a factor of k).

    2. HyDE — Hypothetical Document Embeddings
       (Gao et al. 2022, https://arxiv.org/abs/2212.10496)
       — use an LLM to *answer* the question plausibly, then embed the
       answer and use it as an additional query vector. This bridges
       the query↔document distribution gap (users ask short questions;
       docs are long declarative prose).

The "both" strategy packs both prompts into ONE LLM call to halve cost
and latency — the model emits a JSON object with `rewrites` and
`passage` at once.

Privacy note:
    Rewrites / HyDE are LLM-generated from the raw user query. In the
    demo profile we route them through the hosted lane (cheap, fast).
    For a deployment that treats user queries themselves as sensitive,
    flip `rewrite_lane` in settings to self_hosted_only. The private
    collection's contents never touch the rewriter.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

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
    rewrites: list[str] = field(default_factory=list)   # paraphrases only (excludes original)
    hyde_passage: str = ""
    llm_ms: int = 0
    llm_provider: str = ""
    llm_model: str = ""
    fallback_reason: str = ""                            # empty iff LLM succeeded

    @property
    def query_variants(self) -> list[str]:
        """Original query + rewrites, deduplicated, preserving order."""
        seen: set[str] = set()
        out: list[str] = []
        for q in [self.original, *self.rewrites]:
            q = q.strip()
            if not q or q.lower() in seen:
                continue
            seen.add(q.lower())
            out.append(q)
        return out


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_SYSTEM_BASE = (
    "You are a retrieval assistant. Rewrite user questions to improve "
    "search recall against an enterprise knowledge base. Be faithful to "
    "the original intent. Output STRICT JSON only — no prose, no "
    "markdown fences."
)

_PROMPT_MULTI_QUERY = """\
Generate {k} alternative phrasings of the question below. Each should
preserve the original intent but use different wording (synonyms,
expansion of acronyms, more/less specific framing).

Output JSON:
{{"rewrites": ["<q1>", "<q2>", ...]}}

Question: {query}
"""

_PROMPT_HYDE = """\
Write a short, factual passage (1 to 3 sentences) that would plausibly
answer the question below. Use a neutral, declarative style — match how
the answer would appear in an enterprise knowledge base article. If you
don't know the exact answer, write a plausible one based on domain
conventions.

Output JSON:
{{"passage": "<text>"}}

Question: {query}
"""

_PROMPT_BOTH = """\
Given the question below, produce two things:

  1. {k} alternative phrasings that preserve intent but vary wording.
  2. A short, factual passage (1 to 3 sentences) that would plausibly
     answer the question in the style of an enterprise knowledge base.

Output JSON:
{{"rewrites": ["<q1>", ...], "passage": "<text>"}}

Question: {query}
"""


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
    ) -> RewriteResult:
        """
        Run the requested rewriting strategy. Returns a RewriteResult even
        on failure (rewrites=[], passage="", fallback_reason filled in) so
        callers can unconditionally use `.query_variants` / `.hyde_passage`.
        """
        original = (query or "").strip()
        if strategy == "off" or not original:
            return RewriteResult(strategy=strategy, original=original)

        try:
            if strategy == "multi_query":
                return self._multi_query(original, k=k)
            if strategy == "hyde":
                return self._hyde(original)
            if strategy == "both":
                return self._both(original, k=k)
            logger.warning("unknown rewrite strategy %r — skipping", strategy)
            return RewriteResult(
                strategy=strategy, original=original,
                fallback_reason=f"unknown strategy {strategy!r}",
            )
        except Exception as exc:  # noqa: BLE001
            # Never let rewriting break retrieval.
            logger.warning("query rewrite failed (%s): %s — falling back", strategy, exc)
            return RewriteResult(
                strategy=strategy, original=original,
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------ #
    # Strategy implementations
    # ------------------------------------------------------------------ #

    def _multi_query(self, original: str, *, k: int) -> RewriteResult:
        prompt = _PROMPT_MULTI_QUERY.format(k=max(1, k), query=original)
        text, meta = self._call(prompt)
        parsed = _parse_json_loose(text) or {}
        rewrites = _as_str_list(parsed.get("rewrites", []))[:k]
        return RewriteResult(
            strategy="multi_query", original=original, rewrites=rewrites,
            **meta,
        )

    def _hyde(self, original: str) -> RewriteResult:
        prompt = _PROMPT_HYDE.format(query=original)
        text, meta = self._call(prompt)
        parsed = _parse_json_loose(text) or {}
        passage = str(parsed.get("passage", "") or "").strip()
        return RewriteResult(
            strategy="hyde", original=original, hyde_passage=passage,
            **meta,
        )

    def _both(self, original: str, *, k: int) -> RewriteResult:
        prompt = _PROMPT_BOTH.format(k=max(1, k), query=original)
        text, meta = self._call(prompt)
        parsed = _parse_json_loose(text) or {}
        rewrites = _as_str_list(parsed.get("rewrites", []))[:k]
        passage = str(parsed.get("passage", "") or "").strip()
        return RewriteResult(
            strategy="both", original=original,
            rewrites=rewrites, hyde_passage=passage,
            **meta,
        )

    def _call(self, prompt: str) -> tuple[str, dict]:
        try:
            result = self.llm.complete(
                prompt=prompt,
                system=_SYSTEM_BASE,
                lane=SensitivityLane.HOSTED_OK,
                max_tokens=400,
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
    # Strip ```json fences some models emit despite json_mode
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
