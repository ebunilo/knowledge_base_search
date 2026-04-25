"""
Run `data/golden_set.json` end-to-end through the Generator and collect
per-row diagnostics for calibration + (optional) Ragas.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from langsmith import traceable

from kb.eval.checks import rule_passes
from kb.eval.history import turns_to_conversation_history
from kb.eval.types import GoldenSetFile, load_golden_set
from kb.eval.users import user_for_qid
from kb.generation import Generator, GenerationConfig
from kb.generation.types import GenerationResult
from kb.retrieval.types import RetrievalConfig
from kb.settings import Settings, get_settings


logger = logging.getLogger(__name__)


@dataclass
class EvalResultRow:
    qid: str
    query: str
    user_id: str
    error: str = ""
    # --- outcome
    answer: str = ""
    refused: bool = False
    refusal_reason: str = ""
    # --- pipeline
    top_hit_score: float = 0.0
    n_hits: int = 0
    resolved_query: str = ""
    stepback_query: str = ""
    confidence: float = 0.0
    faithfulness_supported_ratio: float = 0.0
    nli_calls: int = 0
    total_ms: int = 0
    # --- for Ragas / relevancy
    contexts: list[str] = field(default_factory=list)
    ground_truth: str = ""
    # --- eval
    rule_pass: bool = False
    checks: dict[str, bool] = field(default_factory=dict)
    # --- ragas (optional, filled in batch)
    ragas: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "qid": self.qid,
            "query": self.query,
            "user_id": self.user_id,
            "error": self.error,
            "answer": self.answer,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "top_hit_score": self.top_hit_score,
            "n_hits": self.n_hits,
            "resolved_query": self.resolved_query,
            "stepback_query": self.stepback_query,
            "confidence": self.confidence,
            "faithfulness_supported_ratio": self.faithfulness_supported_ratio,
            "nli_calls": self.nli_calls,
            "total_ms": self.total_ms,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
            "rule_pass": self.rule_pass,
            "checks": self.checks,
            "ragas": self.ragas,
        }
        return d


@traceable(name="golden_eval_row", run_type="chain")
def _contexts_from_result(result: GenerationResult, *, max_blocks: int = 10) -> list[str]:
    r = result.retrieval
    if not r or not r.hits:
        return []
    out: list[str] = []
    for h in r.hits[:max_blocks]:
        text = (h.parent_content or h.content or "").strip()
        if text:
            out.append(text[:6000])
    return out


def _run_one_inner(
    ex,
    gen: Generator,
    *,
    gcfg: GenerationConfig,
    rcfg_base: RetrievalConfig,
) -> tuple[GenerationResult, EvalResultRow]:
    user = user_for_qid(ex)
    h = turns_to_conversation_history(ex)
    if h:
        rc = rcfg_base.model_copy(update={"conversation_history": h})
    else:
        rc = rcfg_base

    t0 = time.monotonic()
    result = gen.ask(
        ex.query, user=user,
        retrieval_config=rc, generation_config=gcfg,
    )
    elapsed = int((time.monotonic() - t0) * 1000)

    row = EvalResultRow(
        qid=ex.qid, query=ex.query, user_id=user.user_id,
    )
    row.error = ""
    row.answer = result.answer
    row.refused = result.refused
    row.refusal_reason = result.refusal_reason or ""
    r = result.retrieval
    if r and r.hits:
        row.top_hit_score = float(r.hits[0].score)
        row.n_hits = len(r.hits)
    row.resolved_query = (r.resolved_query or "") if r else ""
    row.stepback_query = (r.stepback_query or "") if r else ""
    row.confidence = float(result.confidence)
    if result.faithfulness:
        row.faithfulness_supported_ratio = float(
            result.faithfulness.supported_ratio,
        )
        row.nli_calls = int(result.faithfulness.nli_calls)
    row.total_ms = elapsed
    row.contexts = _contexts_from_result(result)
    row.ground_truth = (ex.expected_answer_summary or "").strip()
    res_checks = rule_passes(result, ex)
    row.checks = res_checks["checks"]
    row.rule_pass = bool(res_checks["pass"])
    return result, row


def _run_one(
    ex,
    gen: Generator,
    gcfg: GenerationConfig,
    rcfg_base: RetrievalConfig,
) -> EvalResultRow:
    u = user_for_qid(ex)
    try:
        _r, row = _run_one_inner(
            ex, gen, gcfg=gcfg, rcfg_base=rcfg_base,
        )
        row.user_id = u.user_id
        return row
    except Exception as exc:  # noqa: BLE001 — eval must never kill the run
        logger.exception("eval row %s failed: %s", ex.qid, exc)
        return EvalResultRow(
            qid=ex.qid, query=ex.query, user_id=u.user_id,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def run_golden_eval(
    golden_path: str | Path,
    *,
    settings: Optional[Settings] = None,
    limit: int | None = None,
    qid_filter: list[str] | None = None,
    skip_faithfulness: bool = True,
    retrieval: RetrievalConfig | None = None,
    generator_factory: Optional[Callable[[], Generator]] = None,
) -> list[dict[str, Any]]:
    """
    Run every (filtered) example and return serialised rows.
    """
    s = settings or get_settings()
    gfile = load_golden_set(golden_path)
    examples = list(gfile.examples)
    if qid_filter:
        fset = set(qid_filter)
        examples = [e for e in examples if e.qid in fset]
    if limit is not None:
        examples = examples[:limit]

    gcfg = GenerationConfig(
        check_faithfulness=not skip_faithfulness,
        stream=False,
    )
    rc = retrieval or RetrievalConfig(
        rewrite_strategy="off",
    )

    if generator_factory is None:
        gen: Generator = Generator(settings=s)
    else:
        gen = generator_factory()

    rows: list[EvalResultRow] = []
    for ex in examples:
        rows.append(_run_one(ex, gen, gcfg, rc))

    return [r.to_dict() for r in rows]


def save_json(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)
