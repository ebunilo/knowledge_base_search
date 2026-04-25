"""
FaithfulnessChecker — post-stream NLI verification.

For each sentence in the answer:
    * If the sentence carries `[N]` markers, run NLI against EACH cited
      source (`premise = parent_content`, `hypothesis = sentence`). The
      sentence's score is the MAX entailment across cited sources.
        - score ≥ threshold → SUPPORTED
        - score <  threshold → UNSUPPORTED
    * If the sentence carries no markers → UNVERIFIED. We deliberately
      do NOT run NLI against the whole context here: signal-to-noise is
      too low, and the right place for "implicit support" detection is
      a separate Slice-2C grader against the eval set.

Lane / privacy:
    The premise is the cited chunk content; the hypothesis is the
    answer sentence (which paraphrases that content). No new data
    egresses beyond what already left the network during embedding /
    rerank in Phase 1/2. However, in `prod` profile the safety rail
    refuses ANY hosted egress for `self_hosted_only` sources — the
    checker honours that and degrades gracefully (`fallback_reason`).

Failure / cost:
    * One HTTP call per (cited sentence × distinct cited source).
      Typical answer with 4 cited sentences and 1–2 sources each → 4–8
      calls, ~200–500 ms total on a warm endpoint.
    * Any NLI failure → empty `FaithfulnessReport` with
      `fallback_reason` populated. Never raises.
"""

from __future__ import annotations

import logging
from typing import Optional

from kb.generation.nli_client import NLIClient, NLIClientError
from kb.generation.segmentation import Sentence, split_sentences
from kb.generation.types import (
    FaithfulnessReport,
    GenerationConfig,
    SentenceCheck,
)
from kb.retrieval.types import RetrievalHit
from kb.settings import Profile, Settings, get_settings


logger = logging.getLogger(__name__)


class FaithfulnessChecker:
    """Single-method orchestrator. Holds an NLI client + settings."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        nli: NLIClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        # Lazy — instantiating NLIClient touches HF env. The checker
        # itself is cheap; a request that ends up SKIPPED (lane / no
        # citations) never builds the client.
        self._nli_override = nli
        self._nli: NLIClient | None = nli

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def check(
        self,
        answer: str,
        used_hits: list[RetrievalHit],
        *,
        config: Optional[GenerationConfig] = None,
    ) -> FaithfulnessReport:
        """
        Verify `answer` against the cited subset of `used_hits`.

        `used_hits[i]` is the hit numbered `[i+1]` in the prompt. We
        pull premises by 1-based marker. Hits with empty
        `parent_content` fall back to `content`.
        """
        cfg = config or GenerationConfig()
        if not answer or not answer.strip():
            return FaithfulnessReport()

        # Lane safety rail — defence in depth (settings already enforces
        # this combination at startup).
        skip_reason = self._lane_skip_reason(used_hits)
        if skip_reason is not None:
            return FaithfulnessReport(fallback_reason=skip_reason)

        sentences = split_sentences(answer)
        if not sentences:
            return FaithfulnessReport()

        try:
            checks, nli_calls = self._verify_sentences(
                sentences=sentences,
                used_hits=used_hits,
                threshold=cfg.faithfulness_threshold,
            )
        except NLIClientError as exc:
            logger.warning("NLI service down — faithfulness skipped: %s", exc)
            return FaithfulnessReport(
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )

        return _aggregate(checks, nli_calls=nli_calls)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _verify_sentences(
        self,
        *,
        sentences: list[Sentence],
        used_hits: list[RetrievalHit],
        threshold: float,
    ) -> tuple[list[SentenceCheck], int]:
        nli_calls = 0
        results: list[SentenceCheck] = []
        # Cache (sentence_text, parent_id) → score so a repeat citation
        # within an answer doesn't pay twice. Rare in practice but
        # cheap to add.
        cache: dict[tuple[str, str], float] = {}

        for sent in sentences:
            if not sent.markers:
                results.append(SentenceCheck(text=sent.text, status="unverified"))
                continue

            cited_pairs = self._resolve_citations(sent.markers, used_hits)
            if not cited_pairs:
                # Markers all invalid (caught upstream; defensive here).
                results.append(SentenceCheck(
                    text=sent.text,
                    markers=list(sent.markers),
                    status="unverified",
                ))
                continue

            per_source: list[float] = []
            for parent_id, premise in cited_pairs:
                key = (sent.text, parent_id)
                score = cache.get(key)
                if score is None:
                    score = self._nli_client().entailment_score(
                        premise=premise, hypothesis=sent.text,
                    )
                    cache[key] = score
                    nli_calls += 1
                per_source.append(score)

            top = max(per_source)
            status = "supported" if top >= threshold else "unsupported"
            results.append(SentenceCheck(
                text=sent.text,
                markers=list(sent.markers),
                cited_parent_ids=[pid for pid, _ in cited_pairs],
                entailment_score=top,
                per_source_scores=per_source,
                status=status,
            ))

        return results, nli_calls

    def _resolve_citations(
        self,
        markers: tuple[int, ...],
        used_hits: list[RetrievalHit],
    ) -> list[tuple[str, str]]:
        """
        Map `[N]` markers → list of (parent_id, premise_text) for the
        cited hits. Out-of-range markers are silently dropped here —
        citation extraction already surfaced those as `invalid_markers`.
        """
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for n in markers:
            if n < 1 or n > len(used_hits):
                continue
            hit = used_hits[n - 1]
            premise = (hit.parent_content or hit.content or "").strip()
            if not premise:
                continue
            key = hit.parent_id or hit.child_id
            if key in seen:
                continue
            seen.add(key)
            out.append((key, premise))
        return out

    def _lane_skip_reason(self, used_hits: list[RetrievalHit]) -> Optional[str]:
        """Return a human-readable skip reason or None."""
        if self.settings.app_profile != Profile.PROD:
            return None
        for h in used_hits:
            if getattr(h, "sensitivity", "hosted_ok") == "self_hosted_only":
                return (
                    "skipped: prod profile + self_hosted_only sources "
                    "(NLI runs on HF Serverless)"
                )
        return None

    def _nli_client(self) -> NLIClient:
        if self._nli is None:
            self._nli = self._nli_override or NLIClient(self.settings)
        return self._nli


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _aggregate(
    per_sentence: list[SentenceCheck],
    *,
    nli_calls: int,
) -> FaithfulnessReport:
    cited_count = sum(1 for s in per_sentence if s.status in {"supported", "unsupported"})
    supported = sum(1 for s in per_sentence if s.status == "supported")
    unsupported = sum(1 for s in per_sentence if s.status == "unsupported")
    unverified = sum(1 for s in per_sentence if s.status == "unverified")

    supported_ratio = (supported / cited_count) if cited_count else 0.0
    cited_scores = [
        s.entailment_score for s in per_sentence
        if s.status in {"supported", "unsupported"}
    ]
    mean_entailment = (sum(cited_scores) / len(cited_scores)) if cited_scores else 0.0

    return FaithfulnessReport(
        per_sentence=per_sentence,
        cited_sentences=cited_count,
        supported_sentences=supported,
        unsupported_sentences=unsupported,
        unverified_sentences=unverified,
        supported_ratio=supported_ratio,
        mean_entailment=mean_entailment,
        nli_calls=nli_calls,
    )
