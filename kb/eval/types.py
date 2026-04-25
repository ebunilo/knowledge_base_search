"""
Load `data/golden_set.json` — the schema is documented at the file level.
Each example is optional-enriched: fields like `expected_behavior` only
appear on some rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Turn(BaseModel):
    role: str
    content: str


class GoldenExample(BaseModel):
    model_config = ConfigDict(extra="ignore")

    qid: str
    intent: str
    query: str
    # Optional: conversational context for Slice 2B rewrites
    prior_turns: list[Turn] = Field(default_factory=list)
    expected_rewritten_query: str = ""
    expected_answer_summary: str = ""
    expected_source_ids: list[str] = Field(default_factory=list)
    expected_acl: dict[str, Any] = Field(default_factory=dict)
    must_cite: bool = True
    sensitivity_lane: str = "hosted_ok"
    language: str = "en"
    # how_to
    expected_step_count_min: Optional[int] = None
    # Negative / special
    expected_behavior: str = ""
    # f005: employee in non-legal dept should not get legal-only DPA
    expects_acl_denial_for_role: str = ""
    # Per-example user override (optional) — u_001, anonymous, etc.
    eval_user: str = ""


class GoldenSetFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str
    description: str = ""
    examples: list[GoldenExample] = Field(default_factory=list)
    intent_distribution: dict[str, int] = Field(default_factory=dict)
    metrics_to_track: list[str] = Field(default_factory=list)

    @field_validator("examples", mode="before")
    @classmethod
    def _coerce_empty(cls, v: Any) -> Any:
        return v or []


def load_golden_set(path: str | Path) -> GoldenSetFile:
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        return GoldenSetFile.model_validate(json.load(f))
