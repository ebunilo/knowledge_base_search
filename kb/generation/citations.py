"""
Citation extraction.

After the LLM finishes (or while streaming) we scan the answer text for
`[N]` markers and resolve each to the underlying retrieval hit it points
at. We also surface two health signals:

    * invalid_markers — `[N]` references that don't map to a hit
      (e.g. `[9]` when only 5 blocks were given). Indicates the model
      either invented a number or misread the indexing.
    * uncited_hits — context hits whose `[N]` never appeared in the
      answer. Useful for evals: did the model use everything we gave it?

The marker regex is intentionally tight: `[<digits>]`, no spaces, no
ranges (`[1-3]`), no comma lists (`[1,2]`). The system prompt instructs
the model to use multiple separate markers for multi-source claims, so
these forms shouldn't appear; if they do, we ignore them rather than
guessing.
"""

from __future__ import annotations

import re

from kb.generation.types import Citation, CitationExtraction
from kb.retrieval.types import RetrievalHit


_MARKER_RE = re.compile(r"\[(\d+)\]")


def extract_citations(
    answer: str,
    used_hits: list[RetrievalHit],
) -> CitationExtraction:
    """
    Parse all `[N]` markers in `answer` and link them to `used_hits`.

    `used_hits[i]` is the hit numbered `[i+1]` in the prompt — i.e. the
    same ordering the assembler used. Marker numbers are 1-based.

    Returns:
        CitationExtraction with citations (deduplicated by marker, in
        order of first appearance), invalid_markers, and uncited_hits.
    """
    seen_markers: set[int] = set()
    cited_in_order: list[int] = []
    if answer:
        for m in _MARKER_RE.finditer(answer):
            n = int(m.group(1))
            if n in seen_markers:
                continue
            seen_markers.add(n)
            cited_in_order.append(n)

    citations: list[Citation] = []
    invalid: list[int] = []
    valid_range = range(1, len(used_hits) + 1)
    for n in cited_in_order:
        if n not in valid_range:
            invalid.append(n)
            continue
        hit = used_hits[n - 1]
        citations.append(_to_citation(n, hit))

    cited_set = {c.marker for c in citations}
    uncited = [n for n in valid_range if n not in cited_set]

    return CitationExtraction(
        citations=citations,
        invalid_markers=invalid,
        uncited_hits=uncited,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _to_citation(marker: int, hit: RetrievalHit) -> Citation:
    return Citation(
        marker=marker,
        document_id=hit.document_id,
        parent_id=hit.parent_id,
        source_id=hit.source_id,
        source_uri=hit.source_uri,
        title=hit.title,
        section_path=hit.section_path or "",
        score=hit.score,
        visibility=hit.visibility,
        sensitivity=hit.sensitivity,
    )
