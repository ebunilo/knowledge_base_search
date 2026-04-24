"""
Tests for `kb.generation.context.ContextAssembler`.

The assembler must:
    * Number blocks [1], [2], … in retrieval order.
    * Stay under the token budget.
    * Truncate (not drop) the last block when it would otherwise just
      barely overshoot.
    * Drop hits with no usable content silently.
    * Include the summary line only when configured to.
    * Report `dropped_hits` when budget is exhausted before all hits fit.
"""

from __future__ import annotations

import pytest

from kb.chunking.tokens import count_tokens
from kb.generation.context import ContextAssembler
from kb.generation.types import GenerationConfig
from kb.retrieval.types import RetrievalHit


def _hit(
    *,
    parent_id: str,
    title: str = "Doc",
    content: str = "Some parent content.",
    summary: str | None = None,
    section_path: str = "intro",
    source_uri: str = "file:///doc",
    source_id: str = "src",
) -> RetrievalHit:
    return RetrievalHit(
        child_id=parent_id + "_c0",
        parent_id=parent_id,
        document_id="doc1",
        source_id=source_id,
        source_uri=source_uri,
        title=title,
        section_path=section_path,
        content=content[:200],
        parent_content=content,
        summary=summary,
        score=1.0,
    )


class TestAssemble:
    def test_empty_hits_returns_empty(self):
        ctx = ContextAssembler().assemble([], GenerationConfig())
        assert ctx.text == ""
        assert ctx.used_hit_ids == []
        assert ctx.total_tokens == 0
        assert ctx.dropped_hits == 0

    def test_numbers_blocks_in_order(self):
        hits = [
            _hit(parent_id="p1", title="First", content="Alpha content."),
            _hit(parent_id="p2", title="Second", content="Beta content."),
            _hit(parent_id="p3", title="Third", content="Gamma content."),
        ]
        ctx = ContextAssembler().assemble(hits, GenerationConfig())
        assert "[1] Title: First" in ctx.text
        assert "[2] Title: Second" in ctx.text
        assert "[3] Title: Third" in ctx.text
        assert ctx.used_hit_ids == ["p1", "p2", "p3"]
        # [1] must come before [2] which must come before [3].
        assert ctx.text.index("[1]") < ctx.text.index("[2]") < ctx.text.index("[3]")

    def test_includes_source_and_section(self):
        hit = _hit(
            parent_id="p1", title="Doc",
            section_path="api/auth",
            source_uri="file:///docs/api.md",
            source_id="src_api",
        )
        ctx = ContextAssembler().assemble([hit], GenerationConfig())
        assert "Source: file:///docs/api.md" in ctx.text
        assert "Section: api/auth" in ctx.text

    def test_summary_included_when_configured(self):
        hit = _hit(
            parent_id="p1", summary="A short helpful summary.",
        )
        on = ContextAssembler().assemble(
            [hit], GenerationConfig(include_summaries_in_context=True),
        )
        off = ContextAssembler().assemble(
            [hit], GenerationConfig(include_summaries_in_context=False),
        )
        assert "Summary: A short helpful summary." in on.text
        assert "Summary:" not in off.text

    def test_summary_truncated_at_280_chars(self):
        hit = _hit(parent_id="p1", summary="x" * 1000)
        ctx = ContextAssembler().assemble(
            [hit], GenerationConfig(include_summaries_in_context=True),
        )
        line = next(ln for ln in ctx.text.splitlines() if ln.lstrip().startswith("Summary:"))
        # 280 char cap → "..." replaces the tail at 277.
        assert "..." in line
        assert len(line) < 320

    def test_skips_hits_with_empty_content(self):
        hits = [
            _hit(parent_id="p1", title="A", content="Real content."),
            RetrievalHit(
                child_id="c2", parent_id="p2", document_id="d2",
                source_id="s", source_uri="x", title="Empty",
                content="", parent_content="   ",
            ),
            _hit(parent_id="p3", title="B", content="Second real."),
        ]
        ctx = ContextAssembler().assemble(hits, GenerationConfig())
        assert ctx.used_hit_ids == ["p1", "p3"]
        assert "[1] Title: A" in ctx.text
        assert "[2] Title: B" in ctx.text


class TestBudget:
    def test_stays_under_budget(self):
        long_content = "word " * 5000  # ~5000 tokens
        hits = [
            _hit(parent_id=f"p{i}", title=f"Doc{i}", content=long_content)
            for i in range(5)
        ]
        ctx = ContextAssembler().assemble(
            hits,
            GenerationConfig(context_budget_tokens=2000, per_hit_max_tokens=1800),
        )
        assert ctx.total_tokens <= 2000
        assert count_tokens(ctx.text) <= 2000

    def test_truncates_oversized_block(self):
        long_content = "word " * 5000
        hit = _hit(parent_id="p1", title="Big", content=long_content)
        ctx = ContextAssembler().assemble(
            [hit],
            GenerationConfig(context_budget_tokens=400, per_hit_max_tokens=1800),
        )
        assert ctx.total_tokens > 0
        assert ctx.total_tokens <= 400
        assert "[…content truncated…]" in ctx.text

    def test_records_dropped_hits(self):
        long_content = "word " * 1500
        hits = [
            _hit(parent_id=f"p{i}", title=f"Doc{i}", content=long_content)
            for i in range(5)
        ]
        ctx = ContextAssembler().assemble(
            hits,
            GenerationConfig(context_budget_tokens=600, per_hit_max_tokens=400),
        )
        assert ctx.dropped_hits >= 1
        assert len(ctx.used_hit_ids) + ctx.dropped_hits <= 5

    def test_per_hit_cap_enforced(self):
        long_content = "word " * 5000
        hits = [
            _hit(parent_id="p1", title="A", content=long_content),
            _hit(parent_id="p2", title="B", content=long_content),
        ]
        # Generous budget but tight per-hit cap.
        ctx = ContextAssembler().assemble(
            hits,
            GenerationConfig(context_budget_tokens=8000, per_hit_max_tokens=200),
        )
        # Both hits must fit (per-hit cap, not total).
        assert len(ctx.used_hit_ids) == 2
        # Each block, including header, is ~per-hit + 60 overhead.
        assert ctx.total_tokens < 600
