"""
Rule-based pass/fail against golden set expectations.

Ragas (Slice 2C) adds continuous scores; these checks are the binary
contract: source coverage, citation, ACL denial, refusal patterns.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Optional

from kb.eval.types import GoldenExample
from kb.generation.types import GenerationResult
from kb.retrieval.types import RetrievalResult


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _sources_in_hits(
    r: Optional[RetrievalResult],
    expected: list[str],
) -> set[str]:
    if not r or not r.hits or not expected:
        return set()
    return {e for e in expected if any(h.source_id == e for h in r.hits)}


def _cited_sources(res: GenerationResult) -> set[str]:
    return {c.source_id for c in (res.citations or []) if c.source_id}


def match_expected_sources(
    res: GenerationResult,
    ex: GoldenExample,
) -> bool:
    if not ex.expected_source_ids:
        return True
    in_hits = _sources_in_hits(res.retrieval, ex.expected_source_ids)
    if in_hits:
        return True
    in_cite = set(ex.expected_source_ids) & _cited_sources(res)
    return bool(in_cite)


def check_rewrite(
    r: Optional[RetrievalResult],
    ex: GoldenExample,
) -> bool:
    if not ex.expected_rewritten_query:
        return True
    if r is None:
        return False
    can = (r.resolved_query or r.query or "").strip()
    want = ex.expected_rewritten_query.strip()
    if not can:
        return _ratio(r.query, want) >= 0.55
    return _ratio(can, want) >= 0.55


def check_citation(
    res: GenerationResult,
    ex: GoldenExample,
) -> bool:
    if not ex.must_cite or res.refused:
        return True
    if not (res.citations or []):
        return False
    if not ex.expected_source_ids:
        return True
    have = set(ex.expected_source_ids) & _cited_sources(res)
    return bool(have)


def check_f005_employee_denial(
    res: GenerationResult,
    ex: GoldenExample,
) -> bool:
    if ex.qid != "f005" or ex.expects_acl_denial_for_role != "employee":
        return True
    if _sources_in_hits(res.retrieval, ["src_legal_contracts"]):
        return not any(
            c.source_id == "src_legal_contracts" for c in (res.citations or [])
        )
    if res.refused:
        return True
    return not any(
        c.source_id == "src_legal_contracts" for c in (res.citations or [])
    )


def check_neg001(res: GenerationResult, ex: GoldenExample) -> bool:
    if ex.qid != "neg001" or ex.expected_behavior != "refuse_low_confidence":
        return True
    t = (res.answer or "").lower()
    if res.refused:
        return True
    return (
        "do not" in t
        or "don't have" in t
        or "doesn't have" in t
        or ("not" in t and "information" in t)
    )


def check_neg002(res: GenerationResult, ex: GoldenExample) -> bool:
    if ex.qid != "neg002" or ex.expected_behavior != "acl_deny_for_non_legal":
        return True
    t = (res.answer or "").lower()
    in_hits = _sources_in_hits(res.retrieval, ["src_legal_contracts"])
    cited_legal = any(
        c.source_id == "src_legal_contracts" for c in (res.citations or [])
    )
    if in_hits and cited_legal:
        return False
    if not in_hits or res.refused:
        return True
    return "cannot" in t or "don't" in t or "do not" in t


def rule_passes(
    res: GenerationResult,
    ex: GoldenExample,
) -> dict[str, Any]:
    r = res.retrieval
    checks: dict[str, bool] = {}

    checks["sources_or_citation"] = match_expected_sources(res, ex)
    if ex.prior_turns and ex.expected_rewritten_query:
        checks["rewrite"] = check_rewrite(r, ex)
    else:
        checks["rewrite"] = True
    checks["citation"] = check_citation(res, ex) if ex.must_cite else True
    checks["f005_acl"] = check_f005_employee_denial(res, ex)
    checks["neg001"] = check_neg001(res, ex)
    checks["neg002"] = check_neg002(res, ex)

    ok = all(checks.values())
    return {"checks": checks, "pass": ok}
