"""
Heuristic threshold suggestions from an eval run.

We report percentiles of `top_hit_score` and `confidence` over rows that
passed vs failed rule checks. The operator then sets
`GENERATION_MIN_SCORE_THRESHOLD` and `GENERATION_MIN_CONFIDENCE` in `.env`
— there is no single mathematically perfect split without a labelled
precision/recall target; this module documents the *empirical* split from
*this* run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel, Field


def _percentile(sorted_vals: list[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * p
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


class CalibrationReport(BaseModel):
    n_total: int = 0
    n_rule_pass: int = 0
    n_rule_fail: int = 0
    # top hit aggregate score (rerank or RRF path) — from retrieval.hits[0].score
    top_score_pass: dict[str, Optional[float]] = Field(
        default_factory=dict,
    )
    top_score_fail: dict[str, Optional[float]] = Field(
        default_factory=dict,
    )
    confidence_pass: dict[str, Optional[float]] = Field(
        default_factory=dict,
    )
    confidence_fail: dict[str, Optional[float]] = Field(
        default_factory=dict,
    )
    suggested_min_score: Optional[float] = None
    suggested_min_confidence: Optional[float] = None
    notes: list[str] = Field(default_factory=list)

    def model_dump_jsonable(self) -> dict[str, Any]:
        return self.model_dump()


def _stats(vals: list[float], label: str) -> dict[str, Optional[float]]:
    s = sorted(vals)
    if not s:
        return {f"{label}_n": 0, f"{label}_p10": None, f"{label}_p50": None, f"{label}_p90": None}
    return {
        f"{label}_n": len(s),
        f"{label}_p10": _percentile(s, 0.10),
        f"{label}_p50": _percentile(s, 0.50),
        f"{label}_p90": _percentile(s, 0.90),
    }


@dataclass
class _Row:
    rule_pass: bool
    top_score: float
    confidence: float


def build_report(rows: list[dict[str, Any]]) -> CalibrationReport:
    """Build calibration report from a list of serialised `EvalResultRow`."""
    parsed: list[_Row] = []
    for r in rows:
        if r.get("error"):
            continue
        parsed.append(
            _Row(
                rule_pass=bool(r.get("rule_pass")),
                top_score=float(r.get("top_hit_score", 0.0) or 0.0),
                confidence=float(r.get("confidence", 0.0) or 0.0),
            ),
        )
    if not parsed:
        return CalibrationReport(
            n_total=0, notes=["No successful rows; nothing to calibrate on."],
        )

    p_scores = [x.top_score for x in parsed if x.rule_pass]
    f_scores = [x.top_score for x in parsed if not x.rule_pass]
    p_conf = [x.confidence for x in parsed if x.rule_pass]
    f_conf = [x.confidence for x in parsed if not x.rule_pass]

    rep = CalibrationReport(
        n_total=len(parsed),
        n_rule_pass=sum(1 for x in parsed if x.rule_pass),
        n_rule_fail=sum(1 for x in parsed if not x.rule_pass),
        top_score_pass=_stats(p_scores, "top_score") if p_scores else {},
        top_score_fail=_stats(f_scores, "top_score") if f_scores else {},
        confidence_pass=_stats(p_conf, "confidence") if p_conf else {},
        confidence_fail=_stats(f_conf, "confidence") if f_conf else {},
    )

    # Suggestion: set floor slightly below the passing cohort's 10th
    # percentile, so 90% of (rule-pass) rows would still pass the floor.
    # Fails that score *above* the floor are then visible as warnings.
    if p_scores:
        p10s = _percentile(sorted(p_scores), 0.10)
        rep.suggested_min_score = (
            max(0.0, p10s * 0.85) if p10s is not None else None
        )
    if p_conf:
        p10c = _percentile(sorted(p_conf), 0.10)
        rep.suggested_min_confidence = (
            max(0.0, p10c * 0.90) if p10c is not None else None
        )

    if rep.suggested_min_score is not None:
        rep.notes.append(
            f"suggested_min_score: ~{rep.suggested_min_score:.3f} "
            "(tuned from 10th pctile of top_hit_score on rule_pass rows, "
            "shrink factor 0.85 — hand-edit before production)"
        )
    if rep.suggested_min_confidence is not None:
        rep.notes.append(
            f"suggested_min_confidence: ~{rep.suggested_min_confidence:.3f} "
            "(tuned from 10th pctile of confidence on rule_pass rows, "
            "shrink 0.90 — UX floor only, not an automatic gate)"
        )
    return rep
