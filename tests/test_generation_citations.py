"""
Tests for `kb.generation.citations.extract_citations`.
"""

from __future__ import annotations

from kb.generation.citations import extract_citations
from kb.retrieval.types import RetrievalHit


def _hit(parent_id: str, marker_seed: int) -> RetrievalHit:
    return RetrievalHit(
        child_id=f"{parent_id}_c0",
        parent_id=parent_id,
        document_id=f"doc{marker_seed}",
        source_id=f"src{marker_seed}",
        source_uri=f"file:///doc{marker_seed}",
        title=f"Doc {marker_seed}",
        section_path=f"section/{marker_seed}",
        score=0.9 - marker_seed * 0.1,
        visibility="public",
        sensitivity="hosted_ok",
    )


class TestExtraction:
    def test_empty_answer(self):
        out = extract_citations("", [_hit("p1", 1)])
        assert out.citations == []
        assert out.invalid_markers == []
        # Hit was provided but never referenced.
        assert out.uncited_hits == [1]

    def test_basic_marker(self):
        hits = [_hit("p1", 1), _hit("p2", 2)]
        ans = "The thing works as documented in [1]."
        out = extract_citations(ans, hits)
        assert [c.marker for c in out.citations] == [1]
        assert out.citations[0].source_id == "src1"
        assert out.invalid_markers == []
        assert out.uncited_hits == [2]

    def test_multiple_markers_in_order(self):
        hits = [_hit("p1", 1), _hit("p2", 2), _hit("p3", 3)]
        ans = "Step one [2]. Step two [1]. Step three [3]."
        out = extract_citations(ans, hits)
        # Order = first-appearance order.
        assert [c.marker for c in out.citations] == [2, 1, 3]
        assert out.uncited_hits == []

    def test_dedupes_repeated_markers(self):
        hits = [_hit("p1", 1)]
        ans = "First [1]. Restated [1]. Again [1]."
        out = extract_citations(ans, hits)
        assert len(out.citations) == 1
        assert out.citations[0].marker == 1

    def test_invalid_marker_flagged(self):
        hits = [_hit("p1", 1), _hit("p2", 2)]
        ans = "True per [1] and false per [9]."
        out = extract_citations(ans, hits)
        assert [c.marker for c in out.citations] == [1]
        assert out.invalid_markers == [9]

    def test_zero_marker_invalid(self):
        # [0] is out of valid range (1..n).
        hits = [_hit("p1", 1)]
        ans = "Nope [0]."
        out = extract_citations(ans, hits)
        assert out.citations == []
        assert out.invalid_markers == [0]

    def test_ignores_complex_forms(self):
        # The system prompt asks for [N] only; ranges / lists are not parsed.
        hits = [_hit("p1", 1), _hit("p2", 2), _hit("p3", 3)]
        ans = "See [1, 2] or [1-3] or [a]."
        out = extract_citations(ans, hits)
        # [1, 2] doesn't match `\[\d+\]`, so nothing is captured.
        assert out.citations == []
        assert out.invalid_markers == []

    def test_uncited_hits(self):
        hits = [_hit("p1", 1), _hit("p2", 2), _hit("p3", 3)]
        ans = "Only the second [2] was useful."
        out = extract_citations(ans, hits)
        assert [c.marker for c in out.citations] == [2]
        assert out.uncited_hits == [1, 3]

    def test_citation_carries_metadata(self):
        hits = [_hit("p1", 1)]
        ans = "See [1]."
        out = extract_citations(ans, hits)
        c = out.citations[0]
        assert c.title == "Doc 1"
        assert c.source_uri == "file:///doc1"
        assert c.parent_id == "p1"
        assert c.score == 0.8
