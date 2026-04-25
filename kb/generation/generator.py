"""
Generator orchestrator — the top of the answer pipeline.

Stages:
    1. retrieve (delegates to kb.retrieval.Retriever)
    2. lane decision (most-restrictive of the retrieved hits)
    3. refusal gates (no hits, low confidence)
    4. context assembly (numbered, token-budgeted CONTEXT block)
    5. prompt build (system + user)
    6. LLM call — `stream()` for incremental UX, `complete()` for batch
    7. citation extraction (parse [N] markers, link to hits)
    8. result assembly (or stream the events)

Two public methods mirror the two ergonomic shapes:

    Generator().ask(query, user)               -> GenerationResult
    Generator().ask_stream(query, user)        -> Iterator[StreamEvent]

Both share the same retrieval, assembly, prompt, lane, and citation
logic; only the LLM-call shape differs.

Lane policy:
    * If ANY retrieved hit is `self_hosted_only`, the whole answer goes
      through the self-hosted lane. We do not allow hosted+self-hosted
      mixing for a single answer — the answer text could leak content
      from the self-hosted side via paraphrase.
    * If retrieval is empty (refusal path), we never call the LLM, so
      lane is irrelevant.

Refusal policy:
    * `refuse_when_no_hits` (default True) → deterministic message,
      no LLM call, no tokens spent.
    * `min_score_threshold` → if the top hit's aggregate score is below
      the floor, refuse with a "low confidence" message. Disabled by
      default (0.0) until Slice 2 calibrates it against the eval set.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Optional

from kb.enrichment.llm_client import LLMClient, LLMClientError
from kb.generation.citations import extract_citations
from kb.generation.confidence import compute_confidence
from kb.generation.context import ContextAssembler
from kb.generation.faithfulness import FaithfulnessChecker
from kb.generation.prompt import PromptBuilder
from kb.generation.types import (
    AssembledContext,
    FaithfulnessReport,
    GenerationConfig,
    GenerationResult,
    StreamEvent,
)
from kb.retrieval import Retriever
from kb.retrieval.types import (
    RetrievalConfig,
    RetrievalHit,
    RetrievalResult,
    UserContext,
)
from kb.settings import Settings, get_settings
from kb.types import SensitivityLane


logger = logging.getLogger(__name__)


class Generator:
    """Synchronous orchestrator. Holds references to retriever + LLM."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        retriever: Retriever | None = None,
        llm: LLMClient | None = None,
        assembler: ContextAssembler | None = None,
        faithfulness: FaithfulnessChecker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or Retriever(settings=self.settings)
        self.llm = llm or LLMClient(self.settings)
        self.assembler = assembler or ContextAssembler()
        # Lazy — verifying answers is opt-in via GenerationConfig, and
        # the checker only touches HF when actually used. Tests inject
        # a mock here; production goes through the default path.
        self._faithfulness_override = faithfulness
        self._faithfulness: FaithfulnessChecker | None = faithfulness

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ask(
        self,
        query: str,
        user: UserContext | None = None,
        retrieval_config: RetrievalConfig | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Synchronous ask. Returns the full result in one shot."""
        ctx = self._prepare(
            query=query,
            user=user,
            retrieval_config=retrieval_config,
            generation_config=generation_config,
        )

        if ctx.refusal is not None:
            return ctx.refusal

        t_gen = time.monotonic()
        try:
            completion = self.llm.complete(
                prompt=ctx.user_prompt,
                lane=ctx.lane,
                system=ctx.system_prompt,
                max_tokens=ctx.gen_config.max_answer_tokens,
                temperature=ctx.gen_config.temperature,
            )
        except LLMClientError as exc:
            logger.warning("generation failed for q=%r: %s", query, exc)
            return self._build_refusal(
                ctx=ctx,
                reason="llm_unavailable",
                answer=(
                    "The answer service is currently unavailable. "
                    "Please try again in a moment."
                ),
            )

        gen_ms = int((time.monotonic() - t_gen) * 1000)
        report, faithfulness_ms = self._run_faithfulness(
            answer=completion.text, ctx=ctx,
        )
        return self._finalize(
            ctx=ctx,
            answer=completion.text,
            provider=completion.provider,
            model=completion.model,
            generation_ms=gen_ms,
            faithfulness=report,
            faithfulness_ms=faithfulness_ms,
        )

    def ask_stream(
        self,
        query: str,
        user: UserContext | None = None,
        retrieval_config: RetrievalConfig | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> Iterator[StreamEvent]:
        """
        Streaming ask. Yields a sequence of StreamEvents:

            start → token* → done            (happy path)
            start → refused                    (no hits / low confidence)
            start → refused                    (LLM init failed)

        The final `done` event carries the full GenerationResult,
        including parsed citations and timing.
        """
        ctx = self._prepare(
            query=query,
            user=user,
            retrieval_config=retrieval_config,
            generation_config=generation_config,
        )

        if ctx.refusal is not None:
            yield StreamEvent(kind="start", result=ctx.refusal)
            yield StreamEvent(kind="refused", text=ctx.refusal.answer, result=ctx.refusal)
            return

        # Emit a start event before we even initiate the LLM call. The
        # caller can immediately render "thinking..." UX with the
        # context_tokens / used_hit_count metadata.
        partial = GenerationResult(
            query=query,
            user_id=ctx.user.user_id,
            retrieval=ctx.retrieval,
            lane=ctx.lane.value,
            context_tokens=ctx.context.total_tokens,
            used_hit_count=len(ctx.context.used_hit_ids),
        )
        yield StreamEvent(kind="start", result=partial)

        t_gen = time.monotonic()
        try:
            stream = self.llm.stream(
                prompt=ctx.user_prompt,
                lane=ctx.lane,
                system=ctx.system_prompt,
                max_tokens=ctx.gen_config.max_answer_tokens,
                temperature=ctx.gen_config.temperature,
            )
        except LLMClientError as exc:
            logger.warning("stream init failed for q=%r: %s", query, exc)
            refusal = self._build_refusal(
                ctx=ctx,
                reason="llm_unavailable",
                answer=(
                    "The answer service is currently unavailable. "
                    "Please try again in a moment."
                ),
            )
            yield StreamEvent(kind="refused", text=refusal.answer, result=refusal)
            return

        for chunk in stream:
            yield StreamEvent(kind="token", text=chunk)

        gen_ms = int((time.monotonic() - t_gen) * 1000)
        # Faithfulness verification runs AFTER the stream completes —
        # the user has already seen the answer. The report rides on the
        # `done` event so a UI can annotate the rendered answer.
        report, faithfulness_ms = self._run_faithfulness(
            answer=stream.text, ctx=ctx,
        )
        result = self._finalize(
            ctx=ctx,
            answer=stream.text,
            provider=stream.provider,
            model=stream.model,
            generation_ms=gen_ms,
            faithfulness=report,
            faithfulness_ms=faithfulness_ms,
        )
        yield StreamEvent(kind="done", result=result)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _prepare(
        self,
        *,
        query: str,
        user: UserContext | None,
        retrieval_config: RetrievalConfig | None,
        generation_config: GenerationConfig | None,
    ) -> "_AskContext":
        """Run everything up to (but not including) the LLM call."""
        query = (query or "").strip()
        user = user or UserContext()
        retrieval_config = retrieval_config or RetrievalConfig()
        gen_config = generation_config or self._default_gen_config()

        t_total = time.monotonic()

        retrieval = self.retriever.retrieve(
            query=query,
            user=user,
            config=retrieval_config,
        )

        ctx = _AskContext(
            query=query,
            user=user,
            retrieval=retrieval,
            gen_config=gen_config,
            t_total=t_total,
            lane=SensitivityLane.HOSTED_OK,
            context=AssembledContext(text="", total_tokens=0),
            system_prompt="",
            user_prompt="",
            refusal=None,
        )

        # Refusal: no hits.
        if not retrieval.hits and gen_config.refuse_when_no_hits:
            ctx.refusal = self._build_refusal(
                ctx=ctx,
                reason="no_hits",
                answer=PromptBuilder.refusal_no_hits(),
            )
            return ctx

        # Refusal: top hit below confidence floor.
        if retrieval.hits and retrieval.hits[0].score < gen_config.min_score_threshold:
            ctx.refusal = self._build_refusal(
                ctx=ctx,
                reason="low_confidence",
                answer=PromptBuilder.refusal_low_confidence(),
            )
            return ctx

        # Lane = most restrictive among the retrieved hits.
        ctx.lane = self._select_lane(retrieval.hits)

        # Assemble context + prompt.
        ctx.context = self.assembler.assemble(retrieval.hits, gen_config)
        if not ctx.context.used_hit_ids and gen_config.refuse_when_no_hits:
            # Edge case: hits existed but every block was empty / dropped
            # (e.g. all parent_content was blank). Still a refusal.
            ctx.refusal = self._build_refusal(
                ctx=ctx,
                reason="no_usable_context",
                answer=PromptBuilder.refusal_no_hits(),
            )
            return ctx

        ctx.system_prompt = PromptBuilder.system()
        ctx.user_prompt = PromptBuilder.user(ctx.query, ctx.context)
        return ctx

    def _finalize(
        self,
        *,
        ctx: "_AskContext",
        answer: str,
        provider: str,
        model: str,
        generation_ms: int,
        faithfulness: FaithfulnessReport | None = None,
        faithfulness_ms: int = 0,
    ) -> GenerationResult:
        used_hits = self._used_hits_in_order(ctx)
        cite = extract_citations(answer, used_hits)

        confidence = compute_confidence(ctx.retrieval, faithfulness)
        total_ms = int((time.monotonic() - ctx.t_total) * 1000)
        return GenerationResult(
            query=ctx.query,
            user_id=ctx.user.user_id,
            answer=answer.strip(),
            citations=cite.citations,
            invalid_markers=cite.invalid_markers,
            uncited_hits=cite.uncited_hits,
            refused=False,
            retrieval=ctx.retrieval,
            provider=provider,
            model=model,
            lane=ctx.lane.value,
            context_tokens=ctx.context.total_tokens,
            answer_chars=len(answer),
            used_hit_count=len(used_hits),
            faithfulness=faithfulness,
            confidence=confidence,
            generation_ms=generation_ms,
            faithfulness_ms=faithfulness_ms,
            total_ms=total_ms,
        )

    def _build_refusal(
        self,
        *,
        ctx: "_AskContext",
        reason: str,
        answer: str,
    ) -> GenerationResult:
        total_ms = int((time.monotonic() - ctx.t_total) * 1000)
        return GenerationResult(
            query=ctx.query,
            user_id=ctx.user.user_id,
            answer=answer,
            refused=True,
            refusal_reason=reason,
            retrieval=ctx.retrieval,
            lane=ctx.lane.value,
            context_tokens=ctx.context.total_tokens,
            answer_chars=len(answer),
            used_hit_count=len(ctx.context.used_hit_ids),
            total_ms=total_ms,
        )

    def _default_gen_config(self) -> GenerationConfig:
        s = self.settings
        return GenerationConfig(
            context_budget_tokens=s.generation_context_budget_tokens,
            max_answer_tokens=s.generation_max_answer_tokens,
            temperature=s.generation_temperature,
            min_score_threshold=s.generation_min_score_threshold,
            include_summaries_in_context=s.generation_include_summaries_in_context,
            check_faithfulness=s.generation_check_faithfulness,
            faithfulness_threshold=s.generation_faithfulness_threshold,
        )

    def _run_faithfulness(
        self,
        *,
        answer: str,
        ctx: "_AskContext",
    ) -> tuple[FaithfulnessReport | None, int]:
        """
        Run the post-generation NLI check. Returns
        ``(report_or_None, elapsed_ms)``. Only None when the check is
        DISABLED in config; failures yield a populated report with
        ``fallback_reason`` so the caller can audit the degrade path.
        """
        if not ctx.gen_config.check_faithfulness:
            return None, 0
        if not answer or not answer.strip():
            return None, 0

        used_hits = self._used_hits_in_order(ctx)
        if not used_hits:
            # Nothing to verify against. Empty report (signals neutral
            # for confidence; not None — verification was *attempted*).
            return FaithfulnessReport(), 0

        t0 = time.monotonic()
        try:
            report = self._checker().check(
                answer=answer, used_hits=used_hits, config=ctx.gen_config,
            )
        except Exception as exc:  # noqa: BLE001 — checker handles its own; this is belt-and-braces
            logger.warning("faithfulness check raised: %s", exc)
            report = FaithfulnessReport(
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )
        elapsed = int((time.monotonic() - t0) * 1000)
        return report, elapsed

    def _checker(self) -> FaithfulnessChecker:
        if self._faithfulness is None:
            self._faithfulness = (
                self._faithfulness_override
                or FaithfulnessChecker(self.settings)
            )
        return self._faithfulness

    @staticmethod
    def _select_lane(hits: list[RetrievalHit]) -> SensitivityLane:
        if any(getattr(h, "sensitivity", "hosted_ok") == "self_hosted_only" for h in hits):
            return SensitivityLane.SELF_HOSTED_ONLY
        return SensitivityLane.HOSTED_OK

    @staticmethod
    def _used_hits_in_order(ctx: "_AskContext") -> list[RetrievalHit]:
        """
        Map context's used_hit_ids back to RetrievalHit objects, preserving
        order. Used by citation extraction.
        """
        if not ctx.retrieval or not ctx.context.used_hit_ids:
            return []
        by_parent: dict[str, RetrievalHit] = {}
        for h in ctx.retrieval.hits:
            key = h.parent_id or h.child_id
            by_parent.setdefault(key, h)
        out: list[RetrievalHit] = []
        for pid in ctx.context.used_hit_ids:
            hit = by_parent.get(pid)
            if hit is not None:
                out.append(hit)
        return out


# --------------------------------------------------------------------------- #
# Internal value type — easier to pass around than 6 kwargs.
# --------------------------------------------------------------------------- #

class _AskContext:
    __slots__ = (
        "query", "user", "retrieval", "gen_config",
        "t_total", "lane", "context", "system_prompt",
        "user_prompt", "refusal",
    )

    def __init__(
        self,
        *,
        query: str,
        user: UserContext,
        retrieval: RetrievalResult,
        gen_config: GenerationConfig,
        t_total: float,
        lane: SensitivityLane,
        context: AssembledContext,
        system_prompt: str,
        user_prompt: str,
        refusal: Optional[GenerationResult],
    ) -> None:
        self.query = query
        self.user = user
        self.retrieval = retrieval
        self.gen_config = gen_config
        self.t_total = t_total
        self.lane = lane
        self.context = context
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.refusal = refusal
