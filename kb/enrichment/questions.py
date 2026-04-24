"""
Hypothetical question generation.

For each selected child chunk we ask the LLM to produce 3-5 questions that
the chunk answers. At query time, the user's real question is embedded and
matched against these hypothetical question vectors as well as the content
vector — a form of HyDE-in-reverse that consistently beats content-only
retrieval on factoid and conversational queries.

Design choices:
    * Temperature 0.0 for determinism. Ingestion is a batch job; variance
      across runs would make debugging hard.
    * Strict JSON output. We parse with `json.loads`; on failure, we retry
      once with a stricter prompt before giving up.
    * Bounded by `max_questions` (default 5) so the embedding cost stays
      predictable.
"""

from __future__ import annotations

import json
import logging
import re

from kb.enrichment.llm_client import LLMClient, LLMClientError
from kb.types import SensitivityLane


logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You generate short, natural-language questions that a user might ask "
    "to look up the given passage. The questions must be answerable using "
    "ONLY the passage — do not invent facts. Output strict JSON."
)

_USER_PROMPT_TEMPLATE = """\
Passage:
\"\"\"
{chunk}
\"\"\"

Generate between {min_q} and {max_q} distinct, self-contained questions
that this passage answers. Keep each question under 20 words. Vary the
phrasing (what / how / when / who / why where natural). Do NOT include
questions that require information outside this passage.

Return JSON of the form:
{{"questions": ["q1", "q2", ...]}}\
"""


def generate_questions(
    *,
    chunk_text: str,
    lane: SensitivityLane,
    llm: LLMClient,
    min_questions: int = 3,
    max_questions: int = 5,
) -> list[str]:
    """
    Returns between `min_questions` and `max_questions` hypothetical questions.

    Returns an empty list on total failure (after one retry) so the caller
    can continue ingestion without this optional signal.
    """
    if not chunk_text.strip():
        return []

    prompt = _USER_PROMPT_TEMPLATE.format(
        chunk=chunk_text.strip(),
        min_q=min_questions,
        max_q=max_questions,
    )

    for attempt in (1, 2):
        try:
            result = llm.complete(
                prompt=prompt,
                lane=lane,
                system=_SYSTEM_PROMPT,
                max_tokens=400,
                temperature=0.0,
                json_mode=True,
            )
        except LLMClientError as exc:
            logger.warning("question-gen LLM call failed (attempt %d): %s", attempt, exc)
            if attempt == 2:
                return []
            continue

        parsed = _parse_questions(result.text)
        if parsed:
            return parsed[:max_questions]

        if attempt == 1:
            logger.debug("question-gen parse failed, retrying with stricter prompt")
            prompt = prompt + (
                "\n\nIMPORTANT: reply with valid JSON only, no prose, "
                "no code fences."
            )

    return []


# --------------------------------------------------------------------------- #
# Parsing — tolerant to model quirks (code fences, trailing text, etc.)
# --------------------------------------------------------------------------- #

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _parse_questions(raw: str) -> list[str]:
    if not raw:
        return []

    text = raw.strip()
    # Strip code fences if the model added them despite json_mode
    m = _CODE_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Last-ditch: find the first JSON object in the string
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if isinstance(data, dict):
        qs = data.get("questions") or []
    elif isinstance(data, list):
        qs = data
    else:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for q in qs:
        if not isinstance(q, str):
            continue
        q = q.strip()
        if not q or q.lower() in seen:
            continue
        seen.add(q.lower())
        # Normalise: ensure it ends with a question mark.
        if not q.endswith("?"):
            q = q.rstrip(".!") + "?"
        out.append(q)
    return out
