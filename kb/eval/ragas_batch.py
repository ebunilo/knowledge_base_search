"""
Optional Ragas batch metrics (Ragas 0.4+). Install with `pip install ragas datasets`.

Uses the standard `evaluate()` contract: columns `question`, `answer`,
`contexts` (list[str]), `ground_truth`. If anything fails, returns {} and
logs — the main eval run still succeeds.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def run_ragas_on_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    try:
        from datasets import Dataset
        from ragas import evaluate
    except ImportError as exc:  # pragma: no cover
        logger.warning("Ragas not available (%s) — skip --ragas", exc)
        return {}

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []
    for r in rows:
        if r.get("error"):
            continue
        ctxs = r.get("contexts")
        if not ctxs:
            continue
        questions.append(r.get("query", ""))
        answers.append((r.get("answer") or "")[:8000])
        if isinstance(ctxs, list) and all(isinstance(c, str) for c in ctxs):
            contexts.append(ctxs)
        else:
            contexts.append([str(ctxs)])
        ground_truths.append((r.get("ground_truth") or " ")[:2000])

    if not questions:
        logger.warning("Ragas: no rows with `contexts` + `ground_truth` — skip")
        return {}

    try:
        ds = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            },
        )
        # Default metric bundle from Ragas
        out = evaluate(ds, metrics=None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ragas evaluate() failed: %s", exc)
        return {}

    # `out` is EvaluationResult with aggregate scores
    ag: dict[str, float] = {}
    try:
        if hasattr(out, "scores") and out.scores:
            keys: set[str] = set()
            for s in out.scores:
                for k, v in s.items():
                    if isinstance(v, (int, float)) and v == v:
                        keys.add(k)
            for k in keys:
                vals = [row[k] for row in out.scores
                        if k in row and isinstance(row[k], (int, float))]
                if vals:
                    ag[k] = float(sum(vals) / len(vals))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ragas result aggregation failed: %s", exc)
    return ag
