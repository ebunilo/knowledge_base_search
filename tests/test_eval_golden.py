"""Tests for golden-set loading, rule checks, calibration, and eval runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from kb.eval.calibration import build_report
from kb.eval.checks import match_expected_sources, rule_passes
from kb.eval.types import load_golden_set
from kb.eval.users import user_for_qid
from kb.generation.types import Citation, GenerationResult, FaithfulnessReport
from kb.retrieval.types import RetrievalHit, RetrievalResult


ROOT = Path(__file__).resolve().parents[1]


def test_load_golden_set():
    gs = load_golden_set(ROOT / "data" / "golden_set.json")
    assert gs.schema_version == "1.0"
    assert len(gs.examples) >= 20
    f005 = next(e for e in gs.examples if e.qid == "f005")
    assert f005.expects_acl_denial_for_role == "employee"


def test_user_f005_is_employee_not_legal():
    gs = load_golden_set(ROOT / "data" / "golden_set.json")
    f005 = next(e for e in gs.examples if e.qid == "f005")
    u = user_for_qid(
        f005, staff_directory=ROOT / "data" / "staff_directory.json",
    )
    assert u.user_id == "u_002"
    assert u.department == "engineering"


def test_user_neg002_non_legal():
    gs = load_golden_set(ROOT / "data" / "golden_set.json")
    n2 = next(e for e in gs.examples if e.qid == "neg002")
    u = user_for_qid(
        n2, staff_directory=ROOT / "data" / "staff_directory.json",
    )
    assert u.user_id == "u_002"


def _hit(source_id: str) -> RetrievalHit:
    return RetrievalHit(
        child_id="c1", parent_id="p1", document_id="d1",
        source_id=source_id, source_uri="u",
        content="x", parent_content="x", score=0.5,
    )


def test_match_expected_sources():
    from kb.eval.types import GoldenExample
    ex = GoldenExample(
        qid="t", intent="factoid", query="q",
        expected_source_ids=["src_public_kb"],
    )
    res = GenerationResult(
        query="q", user_id="u",
        retrieval=RetrievalResult(
            query="q", user_id="u", hits=[_hit("src_public_kb")],
        ),
    )
    assert match_expected_sources(res, ex) is True


def test_rule_pass_simple():
    from kb.eval.types import GoldenExample
    ex = GoldenExample(
        qid="t1", intent="factoid", query="q",
        expected_source_ids=["src_public_kb"],
    )
    res = GenerationResult(
        query="q", user_id="u",
        citations=[Citation(
            marker=1, document_id="d1", parent_id="p1", source_id="src_public_kb",
            source_uri="s",
        )],
        retrieval=RetrievalResult(
            query="q", user_id="u", hits=[_hit("src_public_kb")],
        ),
    )
    out = rule_passes(res, ex)
    assert out["pass"] is True
    assert out["checks"].get("sources_or_citation") is True


def test_calibration_suggestion():
    rows = [
        {"qid": "a", "rule_pass": True, "top_hit_score": 0.5, "confidence": 0.7},
        {"qid": "b", "rule_pass": True, "top_hit_score": 0.6, "confidence": 0.8},
    ]
    rep = build_report(rows)
    assert rep.suggested_min_score is not None
    assert rep.suggested_min_confidence is not None


def test_run_golden_eval_mocked():
    from kb.eval.runner import run_golden_eval
    from kb.generation import Generator

    gen = MagicMock(spec=Generator)
    gen.ask.return_value = GenerationResult(
        query="q", user_id="u", answer="A [1].", refused=False,
        confidence=0.5, faithfulness=FaithfulnessReport(),
        retrieval=RetrievalResult(
            query="q", user_id="u", hits=[_hit("src_public_kb")],
        ),
    )

    def _fac():
        return gen

    rows = run_golden_eval(
        ROOT / "data" / "golden_set.json",
        limit=1,
        qid_filter=["f002"],
        skip_faithfulness=True,
        generator_factory=_fac,
    )
    assert len(rows) == 1
    assert rows[0]["qid"] == "f002"
    assert gen.ask.call_count == 1
