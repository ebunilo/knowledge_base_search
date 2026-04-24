"""
Context assembly — pack retrieval hits into a numbered, token-budgeted
CONTEXT block.

The block is the single most important input to the LLM. It directly
controls:
    * Which sources the answer can use.
    * Which `[N]` markers the LLM can produce. (We give it numbered
      blocks; it cites them by number; we map back.)
    * How much of each source the LLM sees — full parent (the
      "small-to-big" expansion from retrieval) is preferred, but if a
      single parent is itself larger than the per-hit cap, we truncate
      it from the end and keep the start.

Format per block:

    [N] Title: <doc title>
        Source: <source_uri>
        Section: <section_path>
        Summary: <chunk summary>          # only when include_summaries_in_context=True

        <parent content (truncated if needed)>

Hits are emitted in retrieval order (best first, post-rerank). When the
budget is exhausted partway through a hit, we either truncate that hit
(if it fits at all when shortened) or drop it; remaining hits are
skipped and counted in `dropped_hits` for diagnostics.
"""

from __future__ import annotations

import logging
from typing import Iterable

from kb.chunking.tokens import count_tokens
from kb.generation.types import AssembledContext, GenerationConfig
from kb.retrieval.types import RetrievalHit


logger = logging.getLogger(__name__)


# Approx tokens for the per-block header (Title/Source/Section labels).
# Used as a budget reservation before we even add content.
_HEADER_OVERHEAD_TOKENS = 60


class ContextAssembler:
    """Stateless. One method: `assemble(hits, config)`."""

    def assemble(
        self,
        hits: list[RetrievalHit],
        config: GenerationConfig,
    ) -> AssembledContext:
        if not hits:
            return AssembledContext(text="", total_tokens=0)

        budget = max(0, config.context_budget_tokens)
        per_hit_cap = max(0, config.per_hit_max_tokens)
        include_summary = config.include_summaries_in_context

        blocks: list[str] = []
        used_hit_ids: list[str] = []
        running_tokens = 0
        dropped = 0

        for idx, hit in enumerate(hits, start=0):
            if running_tokens >= budget:
                dropped = len(hits) - idx
                break

            content = _select_content(hit)
            if not content.strip():
                # Nothing to cite — skip silently.
                continue

            remaining = budget - running_tokens
            # Reserve some budget for the header and any block separator.
            available_for_content = max(
                0, min(per_hit_cap, remaining - _HEADER_OVERHEAD_TOKENS),
            )
            if available_for_content <= 0:
                dropped = len(hits) - idx
                break

            content_truncated = _truncate_to_tokens(content, available_for_content)
            block = _format_block(
                marker=len(used_hit_ids) + 1,
                hit=hit,
                content=content_truncated,
                include_summary=include_summary,
            )
            block_tokens = count_tokens(block)

            if running_tokens + block_tokens > budget:
                # The block plus the soft overhead overshot. Try a tighter
                # truncation; if even that doesn't fit, drop the hit.
                slack = max(0, budget - running_tokens - _HEADER_OVERHEAD_TOKENS)
                if slack <= 50:
                    dropped = len(hits) - idx
                    break
                content_truncated = _truncate_to_tokens(content, slack)
                block = _format_block(
                    marker=len(used_hit_ids) + 1,
                    hit=hit,
                    content=content_truncated,
                    include_summary=include_summary,
                )
                block_tokens = count_tokens(block)
                if running_tokens + block_tokens > budget:
                    dropped = len(hits) - idx
                    break

            blocks.append(block)
            used_hit_ids.append(hit.parent_id or hit.child_id)
            running_tokens += block_tokens

        text = "\n\n".join(blocks)
        return AssembledContext(
            text=text,
            used_hit_ids=used_hit_ids,
            total_tokens=running_tokens,
            dropped_hits=dropped,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _select_content(hit: RetrievalHit) -> str:
    """
    Prefer the parent content (small-to-big), fall back to child content.

    We strip leading/trailing whitespace and collapse fully empty multi-
    blank-line runs but otherwise preserve the document's structure —
    citation accuracy depends on the LLM seeing the same prose a human
    would.
    """
    src = hit.parent_content or hit.content or ""
    return src.strip()


def _format_block(
    *,
    marker: int,
    hit: RetrievalHit,
    content: str,
    include_summary: bool,
) -> str:
    """Render one numbered context block."""
    lines = [f"[{marker}] Title: {hit.title or '(untitled)'}"]
    if hit.source_uri:
        lines.append(f"    Source: {hit.source_uri}")
    if hit.section_path:
        lines.append(f"    Section: {hit.section_path}")
    if include_summary and hit.summary:
        # Single-line summary to keep the header compact.
        summary = hit.summary.strip().replace("\n", " ")
        if len(summary) > 280:
            summary = summary[:277] + "..."
        lines.append(f"    Summary: {summary}")
    lines.append("")  # blank line before content
    lines.append(content)
    return "\n".join(lines)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate `text` so that count_tokens(result) ≈ max_tokens.

    `tiktoken` is the source of truth for our token counts; we slice on
    decoded characters because that's what callers actually pass into
    prompts. The slice converges with O(log n) doublings rather than
    re-tokenizing on every character.
    """
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text

    # Binary search on character length.
    lo, hi = 0, len(text)
    best = ""
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid]
        if count_tokens(candidate) <= max_tokens:
            best = candidate
            lo = mid
        else:
            hi = mid - 1

    # Append a marker so the LLM knows it's been truncated.
    return (best.rstrip() + "\n[…content truncated…]") if best else ""
