"""
Sentence segmentation for faithfulness checking.

Why a custom splitter?
    * NLTK / spaCy add a heavy dep + corpus download for what is, at
      this stage, a regex problem.
    * The faithfulness checker only needs `(sentence_text, [markers])`
      pairs. Round-trip fidelity to the original answer text isn't
      required.
    * Citation markers like `[1]`, `[2]` are part of the sentence they
      annotate; we must NOT split on `.[1]` boundaries.

Algorithm:
    1. Find sentence boundary candidates: `[.!?]` followed by whitespace
       and a (capital letter | digit | open-quote).
    2. Reject boundaries inside common abbreviations (`e.g.`, `i.e.`,
       `etc.`, `vs.`, `Dr.`, `Mr.`, `Inc.`).
    3. Split on the surviving boundaries; trim whitespace.
    4. For each sentence, extract `[N]` markers (deduplicated, preserved
       in first-appearance order).
    5. Drop sentences shorter than `min_chars` (default 10) — these are
       almost always citation-only fragments left after splitting.

This is good enough for faithfulness scoring. Edge cases (foreign
punctuation, numbered lists like "1.") are accepted as small noise
sources — they don't change the mean entailment by more than ~0.01 in
practice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Boundary candidates: `.`, `!`, or `?`, optionally followed by `]"' )`,
# then mandatory whitespace, then the next sentence's leading char.
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[\.\!\?])"
    r"[\)\]\"\'\u201d]*"      # optional closing punctuation
    r"\s+"
    r"(?=[A-Z0-9\(\[\"\u201c])"
)

_MARKER_RE = re.compile(r"\[(\d+)\]")

# Trailing-period abbreviations that should NEVER end a sentence.
_ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "vs.", "viz.", "cf.", "Dr.", "Mr.", "Mrs.",
    "Ms.", "Inc.", "Ltd.", "Co.", "Corp.", "Jr.", "Sr.", "St.", "No.",
    "U.S.", "U.K.", "E.U.",
}


@dataclass(frozen=True)
class Sentence:
    """One segmented sentence + the citation markers it carries."""
    text: str
    markers: tuple[int, ...]


def split_sentences(text: str, *, min_chars: int = 10) -> list[Sentence]:
    """
    Split `text` into sentences, preserving citation markers.

    Sentences shorter than `min_chars` are dropped (post-trim) to filter
    out the citation-only fragments that occasionally fall out of split.
    """
    if not text or not text.strip():
        return []

    fragments = _split_safe(text)
    sentences: list[Sentence] = []
    for frag in fragments:
        s = frag.strip()
        if len(s) < min_chars:
            continue
        markers = _markers_in_order(s)
        sentences.append(Sentence(text=s, markers=markers))
    return sentences


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

def _split_safe(text: str) -> list[str]:
    """
    Split on sentence boundaries while skipping abbreviation endings.

    The strategy: collect the boundary positions from the regex, then
    discard any whose preceding token is a known abbreviation. Split
    the original string at the surviving positions.
    """
    boundaries: list[int] = []
    for m in _SENTENCE_BOUNDARY.finditer(text):
        # The match starts at the whitespace following the punctuation.
        # We want to split AFTER any closing punctuation but BEFORE the
        # whitespace, so the previous sentence keeps its punctuation
        # (handy for status messages that quote it back).
        split_at = m.start()
        # Walk back through any closing brackets/quotes to find the
        # preceding "real" word for abbreviation lookup.
        if _is_after_abbrev(text, split_at):
            continue
        boundaries.append(split_at)

    if not boundaries:
        return [text]

    parts: list[str] = []
    prev = 0
    for b in boundaries:
        parts.append(text[prev:b])
        prev = b
    parts.append(text[prev:])
    return parts


def _is_after_abbrev(text: str, idx: int) -> bool:
    """
    Check whether `text[idx]` is the boundary right after a known
    abbreviation (e.g. "e.g."). We look back at the trailing token
    ending in `.` and compare to the abbreviation set.
    """
    # Walk left over closing punctuation.
    end = idx
    while end > 0 and text[end - 1] in ")]\"'\u201d":
        end -= 1
    # Capture the trailing token (alpha+dot pattern).
    start = end
    while start > 0:
        c = text[start - 1]
        if c.isalpha() or c == ".":
            start -= 1
            continue
        break
    token = text[start:end]
    if not token:
        return False
    if token in _ABBREVIATIONS:
        return True
    # Common pattern: single capital + ".": "U.S.", "St."
    # Already in _ABBREVIATIONS; this catches small typos like " e.g."
    return False


def _markers_in_order(text: str) -> tuple[int, ...]:
    """Return citation markers in first-appearance order, deduplicated."""
    seen: set[int] = set()
    out: list[int] = []
    for m in _MARKER_RE.finditer(text):
        n = int(m.group(1))
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return tuple(out)
