"""
Aggregate confidence scoring.

Combines the two strongest available signals:

    1. Retrieval confidence — does the corpus actually contain something
       relevant to the question?
    2. Faithfulness — given that the LLM answered, are its claims
       entailed by the cited sources?

Score = sqrt( retrieval_signal * faithfulness_signal )

Geometric mean is deliberate: a low score on EITHER signal pulls the
aggregate down. (An arithmetic mean would let a flawless answer "carry"
weak retrieval, or vice versa.)

This is a heuristic. Phase 3 · Slice 2C will replace the hand-tuned
constants below with values calibrated against `data/golden_set.json`
using a simple offline grid search on (retrieval_norm_factor,
faithfulness_floor, threshold).

Edge cases
    * No retrieval hits → 0.0 (the answer is by definition unsupported).
    * Faithfulness skipped (e.g. lane safety, NLI down) → fall back to
      retrieval signal alone, capped at 0.85 to honestly reflect the
      missing verification signal.
    * No cited sentences (answer was free-form, e.g. refusal that
      slipped through) → treat as 0.5 faithfulness — neutral.
"""

from __future__ import annotations

import math
from typing import Optional

from kb.generation.types import FaithfulnessReport
from kb.retrieval.types import RetrievalResult


# Retrieval normalisation:
#   * RRF scores are tiny (max ~ 1/k for k=60 → 0.017 per retriever).
#     We rescale by a plausible "good first hit" baseline.
#   * Reranker scores (cross-encoder logits) are unbounded; sigmoid them.
_RRF_NORM_FACTOR = 0.025      # ≈ rank-1 score with both retrievers contributing
_RERANK_NO_VERIFICATION_CAP = 0.85


def compute_confidence(
    retrieval: Optional[RetrievalResult],
    faithfulness: Optional[FaithfulnessReport],
) -> float:
    """
    Return aggregate confidence in [0, 1].

    See module docstring for semantics.
    """
    r = _retrieval_signal(retrieval)
    if r <= 0.0:
        return 0.0

    f = _faithfulness_signal(faithfulness)
    if f is None:
        # Verification skipped — be honest about uncertainty.
        return min(_RERANK_NO_VERIFICATION_CAP, r)

    return math.sqrt(r * f)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

def _retrieval_signal(retrieval: Optional[RetrievalResult]) -> float:
    if retrieval is None or not retrieval.hits:
        return 0.0
    top = retrieval.hits[0]
    score = float(top.score or 0.0)
    if retrieval.rerank_applied:
        # Cross-encoder logit → entailment-style probability.
        return 1.0 / (1.0 + math.exp(-score))
    # RRF score: scale by a typical good-first-rank baseline.
    return max(0.0, min(1.0, score / _RRF_NORM_FACTOR))


def _faithfulness_signal(
    faithfulness: Optional[FaithfulnessReport],
) -> Optional[float]:
    """
    Returns None when the verification signal is unavailable (skipped /
    failed). Returns 0.5 (neutral) when faithfulness ran but produced
    no cited-sentence verdicts. Otherwise returns supported_ratio.
    """
    if faithfulness is None:
        return None
    if faithfulness.fallback_reason:
        return None
    if faithfulness.cited_sentences == 0:
        # Answer ran, but nothing was citation-verifiable — neutral.
        return 0.5
    return faithfulness.supported_ratio
