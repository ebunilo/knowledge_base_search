"""
Generation-side data contracts.

These mirror the shape of `kb.retrieval.types` — pydantic models that flow
between stages of the answer pipeline:

    retrieve()  →  assemble_context()  →  build_prompt()
                →  llm.stream() / llm.complete()  →  extract_citations()
                →  GenerationResult

Streaming consumers receive a stream of `StreamEvent`s; non-streaming
consumers get a single `GenerationResult`. The two paths share the same
metadata schema so the eval harness can drive either uniformly.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from kb.retrieval.types import RetrievalResult


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

class GenerationConfig(BaseModel):
    """Tunables for a single ask() call. Defaults mirror Settings."""

    # Context budget for the CONTEXT block in the user prompt. The
    # assembler packs hits in score order until this is exhausted.
    context_budget_tokens: int = 6000

    # Per-hit maximum — even a single huge parent gets truncated to this.
    per_hit_max_tokens: int = 1800

    # Cap on the LLM's response.
    max_answer_tokens: int = 1024

    temperature: float = 0.1

    # When True, stream the response. When False, return the final text in
    # one shot. The CLI and any UI that wants progressive rendering should
    # use stream=True.
    stream: bool = False

    # If the top retrieved hit's aggregate score is below this floor, we
    # refuse without calling the LLM. 0.0 disables the gate.
    min_score_threshold: float = 0.0

    # Refuse with a deterministic message (no LLM call) when retrieval
    # returns nothing. Saves cost and prevents hallucination on
    # out-of-corpus questions.
    refuse_when_no_hits: bool = True

    # Whether to inline each chunk's summary alongside its parent content.
    include_summaries_in_context: bool = True

    # ---- Faithfulness (NLI) ---- #
    # Run a post-generation NLI check against the cited sources?
    check_faithfulness: bool = True
    # Entailment probability ≥ this → sentence is SUPPORTED. < threshold
    # but cited → UNSUPPORTED. Sentences with no citations are UNVERIFIED.
    faithfulness_threshold: float = 0.5


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #

class AssembledContext(BaseModel):
    """
    The CONTEXT block fed to the LLM, plus the metadata needed to map
    the LLM's [N] citations back to the underlying retrieval hits.
    """
    text: str
    # Index in `used_hits` is `marker - 1`. So a [3] in the answer points at
    # used_hits[2]. used_hits is the post-truncation, in-order list of hits
    # that actually made it into the context.
    used_hit_ids: list[str] = Field(default_factory=list)   # parent_id of each used hit
    total_tokens: int = 0
    # Hits that retrieve produced but didn't fit / were dropped.
    dropped_hits: int = 0


# --------------------------------------------------------------------------- #
# Citations
# --------------------------------------------------------------------------- #

class Citation(BaseModel):
    """One [N] reference in the answer, resolved back to a retrieval hit."""

    marker: int                               # the integer in [N]
    document_id: str
    parent_id: str
    source_id: str
    source_uri: str
    title: Optional[str] = None
    section_path: str = ""
    score: float = 0.0
    visibility: str = "public"
    sensitivity: str = "hosted_ok"


class CitationExtraction(BaseModel):
    """Output of citation parsing — citations + diagnostics."""

    citations: list[Citation] = Field(default_factory=list)
    # Markers that appear in the answer but don't map to a context hit.
    invalid_markers: list[int] = Field(default_factory=list)
    # Hits that were in the context but not cited by the LLM. Useful for
    # eval / "Did the model use everything we gave it?".
    uncited_hits: list[int] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Faithfulness (post-stream NLI)
# --------------------------------------------------------------------------- #

# A sentence's verification verdict. The 4-state model is intentional:
#   * SUPPORTED   — at least one cited source entails the sentence (NLI ≥ thr)
#   * UNSUPPORTED — sentence had citations but none of them entailed it
#   * UNVERIFIED  — sentence had no citation markers at all (e.g. framing /
#                   connective text); we don't try to score these against
#                   the whole context — too noisy for one signal.
#   * SKIPPED     — segmentation kept the sentence (e.g. very short, or only
#                   citation tokens) but it isn't worth NLI-checking.
SentenceStatus = Literal["supported", "unsupported", "unverified", "skipped"]


class SentenceCheck(BaseModel):
    """One sentence's verification result."""
    text: str
    markers: list[int] = Field(default_factory=list)
    cited_parent_ids: list[str] = Field(default_factory=list)
    # Max entailment probability across cited sources (0..1).
    entailment_score: float = 0.0
    # Per-source entailment probabilities, in marker order. For audit /
    # debugging; the UI only needs `entailment_score` + `status`.
    per_source_scores: list[float] = Field(default_factory=list)
    status: SentenceStatus = "unverified"


class FaithfulnessReport(BaseModel):
    """Aggregate faithfulness over an answer."""
    per_sentence: list[SentenceCheck] = Field(default_factory=list)
    # Aggregates over CITED sentences (status in {supported, unsupported}).
    cited_sentences: int = 0
    supported_sentences: int = 0
    unsupported_sentences: int = 0
    unverified_sentences: int = 0
    # supported / cited (0..1). 0 when no cited sentences.
    supported_ratio: float = 0.0
    # Mean entailment over cited sentences (0..1).
    mean_entailment: float = 0.0
    # Number of NLI HTTP calls actually issued (for cost tracing).
    nli_calls: int = 0
    # Set when the checker silently degraded (NLI service down, lane
    # skip, etc). The aggregate fields are 0 in that case.
    fallback_reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #

class GenerationResult(BaseModel):
    """Top-level result for one ask() call."""

    query: str
    user_id: str
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)

    refused: bool = False
    refusal_reason: Optional[str] = None

    # Pass-through retrieval diagnostics — full visibility into why this
    # answer was produced. Always populated, even on refusal.
    retrieval: Optional[RetrievalResult] = None

    # Generation diagnostics
    provider: Optional[str] = None
    model: Optional[str] = None
    lane: Optional[str] = None                   # "hosted_ok" | "self_hosted_only"
    context_tokens: int = 0
    answer_chars: int = 0
    used_hit_count: int = 0
    invalid_markers: list[int] = Field(default_factory=list)
    uncited_hits: list[int] = Field(default_factory=list)

    # Post-stream NLI verification — populated when check_faithfulness is
    # on and the answer was non-empty. None when skipped (e.g. refusal,
    # lane-safe skip in prod).
    faithfulness: Optional[FaithfulnessReport] = None
    # Aggregate confidence in the answer, 0..1. Combines retrieval signal
    # with `faithfulness.supported_ratio` (geometric mean). Heuristic;
    # Slice 2C calibrates against the golden set.
    confidence: float = 0.0

    generation_ms: int = 0
    faithfulness_ms: int = 0
    total_ms: int = 0


# --------------------------------------------------------------------------- #
# Streaming protocol
# --------------------------------------------------------------------------- #

class StreamEvent(BaseModel):
    """
    One event in the streaming protocol.

    Event kinds:
        * `start`     — emitted once before any tokens; carries provider /
                        model / lane / context_tokens / used_hit_count.
        * `token`     — incremental text chunk; aggregate them client-side.
        * `refused`   — emitted instead of any tokens when we refuse.
        * `done`      — emitted last; carries the final GenerationResult
                        with citations populated.
    """
    kind: Literal["start", "token", "refused", "done"]
    text: str = ""
    # Populated on `start` and `done`.
    result: Optional[GenerationResult] = None
