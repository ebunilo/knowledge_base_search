"""
Golden-set evaluation (Phase 3 · Slice 2C).

* `load_golden_set` / `GoldenSetFile` — schema for `data/golden_set.json`
* `run_golden_eval` — drive `Generator.ask` over each example
* `build_report` — heuristics for suggested score / confidence floors
* Optional Ragas: `kb.eval.ragas_batch.run_ragas_on_rows` after a run
"""

from kb.eval.calibration import build_report, CalibrationReport
from kb.eval.runner import run_golden_eval, save_json, EvalResultRow
from kb.eval.types import GoldenExample, GoldenSetFile, load_golden_set

__all__ = [
    "CalibrationReport",
    "EvalResultRow",
    "GoldenExample",
    "GoldenSetFile",
    "build_report",
    "load_golden_set",
    "run_golden_eval",
    "save_json",
]
