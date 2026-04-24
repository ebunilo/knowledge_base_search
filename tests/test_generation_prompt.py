"""
Tests for `kb.generation.prompt.PromptBuilder`.

Prompts are stable text; we mainly verify:
    * The system prompt contains the citation contract and refusal rule.
    * The user template puts CONTEXT before QUESTION before ANSWER:
    * Refusal text is non-empty and stable across calls.
"""

from __future__ import annotations

from kb.generation.prompt import PromptBuilder
from kb.generation.types import AssembledContext


class TestSystem:
    def test_mentions_citations(self):
        sys_text = PromptBuilder.system()
        assert "[N]" in sys_text or "[1]" in sys_text
        assert "context" in sys_text.lower()

    def test_mentions_refusal(self):
        sys_text = PromptBuilder.system()
        # Must instruct the model to refuse rather than invent.
        assert "do not" in sys_text.lower() or "do not invent" in sys_text.lower()
        assert "not contain" in sys_text.lower()

    def test_stable_across_calls(self):
        assert PromptBuilder.system() == PromptBuilder.system()


class TestUser:
    def test_orders_context_question_answer(self):
        ctx = AssembledContext(
            text="[1] Title: X\n    Source: u\n\nbody",
            used_hit_ids=["p1"],
            total_tokens=10,
        )
        text = PromptBuilder.user("how does x work?", ctx)
        i_ctx = text.index("CONTEXT:")
        i_q = text.index("QUESTION:")
        i_a = text.index("ANSWER:")
        assert i_ctx < i_q < i_a
        assert "how does x work?" in text
        assert "[1] Title: X" in text

    def test_strips_question_whitespace(self):
        ctx = AssembledContext(text="anything", used_hit_ids=["p1"], total_tokens=1)
        text = PromptBuilder.user("   how do I deploy?  \n", ctx)
        assert "QUESTION: how do I deploy?\n" in text or "QUESTION: how do I deploy?" in text

    def test_empty_context_uses_placeholder(self):
        ctx = AssembledContext(text="", used_hit_ids=[], total_tokens=0)
        text = PromptBuilder.user("anything", ctx)
        assert "(no relevant context found)" in text


class TestRefusals:
    def test_no_hits_message_non_empty(self):
        msg = PromptBuilder.refusal_no_hits()
        assert len(msg) > 20
        assert "knowledge base" in msg.lower()

    def test_low_confidence_message_non_empty(self):
        msg = PromptBuilder.refusal_low_confidence()
        assert len(msg) > 20
