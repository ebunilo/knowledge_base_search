"""
Enrichment orchestrator.

Walks a ChunkedDocument and produces an EnrichedDocument by generating
hypothetical questions and a 1-sentence summary for each selected child.

Selection is controlled by `INGESTION_ENRICH_STRATEGY`:
    * "off"         — no enrichment (questions=[] , summary=None)
    * "all"         — enrich every child
    * "sample_first" — enrich the first N children per source, where N is
                       `INGESTION_ENRICH_LIMIT`. Good default for a cost-
                       capped demo: you still get the retrieval benefit on
                       the most-read sections without paying for the tail.

The orchestrator is stateless; callers (or tests) can pass a custom
strategy by providing their own `select_children` predicate.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from kb.enrichment.llm_client import LLMClient
from kb.enrichment.questions import generate_questions
from kb.enrichment.summary import generate_summary
from kb.settings import get_settings
from kb.types import (
    ChildChunk,
    ChunkedDocument,
    EnrichedChildChunk,
    EnrichedDocument,
)


logger = logging.getLogger(__name__)


@dataclass
class EnrichmentStats:
    children_seen: int = 0
    children_enriched: int = 0
    questions_written: int = 0
    summaries_written: int = 0
    failures: int = 0


ChildPredicate = Callable[[ChildChunk, int], bool]


class Enricher:
    """Reusable enricher — carries one LLMClient and a global budget counter."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        strategy: str | None = None,
        global_limit: int | None = None,
    ) -> None:
        settings = get_settings()
        self.llm = llm or LLMClient(settings)
        self.strategy = strategy or settings.ingestion_enrich_strategy
        self.global_limit = (
            global_limit if global_limit is not None else settings.ingestion_enrich_limit
        )
        self.enriched_so_far = 0
        self.stats = EnrichmentStats()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def enrich(self, chunked: ChunkedDocument) -> EnrichedDocument:
        predicate = self._build_predicate()

        new_children: list[EnrichedChildChunk] = []
        for idx, child in enumerate(chunked.children):
            self.stats.children_seen += 1

            if not predicate(child, idx):
                new_children.append(self._passthrough(child))
                continue

            questions, summary = self._enrich_child(child, chunked)
            new_children.append(
                EnrichedChildChunk(
                    **child.model_dump(),
                    hypothetical_questions=questions,
                    summary=summary,
                )
            )
            self.stats.children_enriched += 1
            self.enriched_so_far += 1
            self.stats.questions_written += len(questions)
            if summary:
                self.stats.summaries_written += 1

        base = chunked.model_dump(exclude={"children"})
        return EnrichedDocument(**base, children=new_children)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _enrich_child(
        self,
        child: ChildChunk,
        parent_doc: ChunkedDocument,
    ) -> tuple[list[str], str | None]:
        try:
            questions = generate_questions(
                chunk_text=child.content,
                lane=parent_doc.sensitivity,
                llm=self.llm,
            )
        except Exception:  # noqa: BLE001
            logger.exception("question-gen crashed for %s", child.child_id)
            questions = []
            self.stats.failures += 1

        try:
            summary = generate_summary(
                chunk_text=child.content,
                lane=parent_doc.sensitivity,
                llm=self.llm,
            )
        except Exception:  # noqa: BLE001
            logger.exception("summary-gen crashed for %s", child.child_id)
            summary = None
            self.stats.failures += 1

        return questions, summary

    def _build_predicate(self) -> ChildPredicate:
        strategy = self.strategy
        if strategy == "off":
            return lambda _child, _idx: False
        if strategy == "all":
            return lambda _child, _idx: self.enriched_so_far < self.global_limit
        if strategy == "sample_first":
            return lambda _child, _idx: self.enriched_so_far < self.global_limit
        logger.warning("Unknown enrichment strategy %r; defaulting to off", strategy)
        return lambda _child, _idx: False

    @staticmethod
    def _passthrough(child: ChildChunk) -> EnrichedChildChunk:
        return EnrichedChildChunk(**child.model_dump())


# --------------------------------------------------------------------------- #
# Convenience function for one-shot use
# --------------------------------------------------------------------------- #

def enrich_document(chunked: ChunkedDocument) -> EnrichedDocument:
    return Enricher().enrich(chunked)
