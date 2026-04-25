"""HTTP request/response models for the web API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class UserPayload(BaseModel):
    """Resolves to `UserContext` — either `user_id` from staff, or free-form."""

    user_id: str = "anonymous"
    role: Optional[str] = None
    department: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    user: UserPayload = Field(default_factory=UserPayload)
    top_k: int = 8
    top_k_dense: int = 30
    top_k_sparse: int = 30
    rerank: bool = True
    rerank_top_n: int = 30
    rewrite: str = "off"  # off, multi_query, hyde, both
    multi_query_k: int = 2
    include_parent_content: bool = True
    rrf_dense_weight: float = 1.0
    rrf_sparse_weight: float = 1.0


class AskRequest(BaseModel):
    query: str
    user: UserPayload = Field(default_factory=UserPayload)
    top_k: int = 8
    rerank: bool = True
    rewrite: str = "off"
    multi_query_k: int = 2
    stepback: bool = False
    session_id: Optional[str] = None
    # Generation
    check_faithfulness: Optional[bool] = None
    max_answer_tokens: Optional[int] = None
    temperature: Optional[float] = None


class HealthResponse(BaseModel):
    ok: bool
    services: dict[str, str]  # name -> "ok" | "down" | "skip"


class ConfigResponse(BaseModel):
    app_profile: str
    users: list[dict[str, str]]  # {id, label, department, role}
