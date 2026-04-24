"""
Prompt construction.

Two outputs:
    * `system_prompt()` — defines the assistant's role, citation contract,
      and refusal policy. Stable across calls.
    * `user_prompt(question, context)` — wraps the assembled CONTEXT
      block and the user's question into one message.

Why so opinionated?
    * The system prompt forces the model into "extractive QA" mode: never
      invent facts, always cite, refuse when the answer isn't in context.
    * The user prompt's structure (CONTEXT block, then QUESTION, then
      ANSWER:) is what most modern instruction-tuned models recognise as
      the "answer the question using the supplied context" pattern.
    * Citation rules are explicit and minimal — `[N]` only, matching the
      block numbering we control. Anything more elaborate (e.g.
      `[author, year]`) bleeds into hallucination.

The refusal text used when retrieval is empty is also defined here so it
sits next to the rest of the user-visible language.
"""

from __future__ import annotations

from kb.generation.types import AssembledContext


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are an enterprise knowledge-base assistant. You answer the user's
question using ONLY the information in the CONTEXT block below. The
CONTEXT contains numbered passages [1], [2], [3], …

Rules — follow ALL of them:

1. Cite every factual claim with [N] markers that match the numbered
   CONTEXT items. You may cite multiple sources for one claim, e.g.
   "...as described in [2][5]." Do not invent citation numbers; only
   numbers present in the CONTEXT are valid.

2. Never use information that is not in the CONTEXT. If a relevant fact
   is not in the CONTEXT, do not include it in your answer.

3. If the CONTEXT does not contain enough information to answer the
   question, say so explicitly. Begin such answers with: "The provided
   sources do not contain enough information to answer that." Then
   describe what would be needed.

4. Be concise and direct. Prefer short paragraphs and bulleted lists for
   procedural questions. Do not add disclaimers, apologies, or
   pleasantries.

5. Do not output the CONTEXT verbatim. Synthesise the answer in your own
   words, then cite.
"""


def system_prompt() -> str:
    """Return the canonical system prompt — stable across requests."""
    return _SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# User prompt
# --------------------------------------------------------------------------- #

_USER_TEMPLATE = """\
CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def user_prompt(question: str, context: AssembledContext) -> str:
    """Wrap the context block and the question in the standard template."""
    question = (question or "").strip()
    return _USER_TEMPLATE.format(
        context=context.text or "(no relevant context found)",
        question=question,
    )


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #

_NO_HITS_REFUSAL = (
    "I couldn't find any information in the knowledge base that addresses "
    "your question. Try rephrasing the question, or contact the team that "
    "owns the topic directly."
)

_LOW_CONFIDENCE_REFUSAL = (
    "The knowledge base does not contain a strong match for that question. "
    "I'd rather say so than guess. Try rephrasing or asking a more specific "
    "question."
)


def refusal_no_hits() -> str:
    return _NO_HITS_REFUSAL


def refusal_low_confidence() -> str:
    return _LOW_CONFIDENCE_REFUSAL


# --------------------------------------------------------------------------- #
# Convenience class wrapping the above (matches the rest of the package's
# class-style ergonomics).
# --------------------------------------------------------------------------- #

class PromptBuilder:
    @staticmethod
    def system() -> str:
        return system_prompt()

    @staticmethod
    def user(question: str, context: AssembledContext) -> str:
        return user_prompt(question, context)

    @staticmethod
    def refusal_no_hits() -> str:
        return refusal_no_hits()

    @staticmethod
    def refusal_low_confidence() -> str:
        return refusal_low_confidence()
