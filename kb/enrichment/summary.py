"""
Per-chunk summary generation.

We ask the LLM for a single sentence that captures what this chunk is about.
The summary is:
    * Shown inline in the UI next to citations.
    * Used by the context-compression step at query time to drop chunks
      that are topically off-relevance even if they scored high on BM25.
    * Cheap enough (~20 output tokens) that doing it alongside question
      generation barely moves the ingestion bill.
"""

from __future__ import annotations

import logging

from kb.enrichment.llm_client import LLMClient, LLMClientError
from kb.types import SensitivityLane


logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You write one-sentence summaries of passages from internal "
    "documentation. The summary must be factual, self-contained, and "
    "under 30 words."
)

_USER_PROMPT_TEMPLATE = """\
Passage:
\"\"\"
{chunk}
\"\"\"

Write exactly ONE sentence summarising this passage. Do not add prefixes
like "This passage..." — just the sentence."""


def generate_summary(
    *,
    chunk_text: str,
    lane: SensitivityLane,
    llm: LLMClient,
) -> str | None:
    if not chunk_text.strip():
        return None

    try:
        result = llm.complete(
            prompt=_USER_PROMPT_TEMPLATE.format(chunk=chunk_text.strip()),
            lane=lane,
            system=_SYSTEM_PROMPT,
            max_tokens=80,
            temperature=0.0,
        )
    except LLMClientError as exc:
        logger.warning("summary LLM call failed: %s", exc)
        return None

    text = result.text.strip().strip('"').strip("'")
    if not text:
        return None

    # Some models return multi-sentence output despite the instruction.
    # Keep only the first sentence.
    for sep in (". ", "\n"):
        if sep in text:
            text = text.split(sep, 1)[0].rstrip(".").strip() + "."
            break

    return text or None
