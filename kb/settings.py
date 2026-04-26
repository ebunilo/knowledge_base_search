"""
Application settings — profile-aware, loaded from .env.

Exposes a single `get_settings()` helper that returns a cached, validated
Settings object. Every module imports settings through this helper so that
env-var parsing (types, defaults, validation) is centralized.

Safety rails implemented here:
  * refuse to start when APP_PROFILE=prod AND the self-hosted-lane override
    flag is true (see infra_provisioning.md §8.5).
  * warn at startup when the override flag is active, naming which providers
    are configured on each lane.
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)


class Profile(str, Enum):
    DEMO = "demo"
    DEMO_ISOLATED = "demo-isolated"
    PROD = "prod"


class Settings(BaseSettings):
    """
    All configuration comes from environment variables / .env. The defaults
    here mirror the demo profile so a bare checkout can run end-to-end.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------- App -------------------- #
    app_profile: Profile = Profile.DEMO
    app_env: str = "dev"
    app_region: str = "eu-west-1"
    app_tenant_internal_id: str = "tenant_acme_staff"
    app_public_collection: str = "public_v1"
    app_private_collection: str = "tenant_acme_staff_private_v1"

    # -------------------- Lane control -------------------- #
    demo_allow_hosted_for_selfhosted_lane: bool = False

    hosted_lane_priority: str = "openai:gpt-4o-mini,qwen:qwen-max"
    selfhosted_lane_priority: str = "openai:gpt-4o-mini,qwen:qwen-max"

    # -------------------- Hugging Face -------------------- #
    hf_api_token: str = ""
    hf_embed_mode: Literal["serverless", "endpoint"] = "serverless"
    hf_embed_model_id: str = "BAAI/bge-m3"
    hf_embed_endpoint_url: str = ""

    hf_rerank_mode: Literal["serverless", "endpoint"] = "serverless"
    hf_rerank_model_id: str = "BAAI/bge-reranker-v2-m3"
    hf_rerank_endpoint_url: str = ""

    hf_nli_model_id: str = "facebook/bart-large-mnli"

    hf_llm_selfhosted_endpoint_url: str = ""
    hf_llm_selfhosted_model: str = "Qwen/Qwen2.5-14B-Instruct"
    hf_llm_selfhosted_keep_warm: bool = False
    hf_llm_selfhosted_cold_start_timeout_s: int = 120

    # -------------------- Qwen (OpenAI-compatible) -------------------- #
    qwen_api_key: str = ""
    qwen_api_base: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    qwen_model_small: str = "qwen-turbo"
    qwen_model_mid: str = "qwen-plus"
    qwen_model_large: str = "qwen-max"

    # -------------------- OpenAI -------------------- #
    openai_api_key: str = ""
    openai_model_generator: str = "gpt-4o-mini"

    # -------------------- Query guardrails (RAG / API) -------------------- #
    # Block overly long, obviously injected, or code-dump input before
    # rewriter/retrieval/LLM. Disable only for local experiments.
    guardrails_enabled: bool = True
    guardrails_max_query_chars: int = 4000
    guardrails_max_query_lines: int = 50

    # -------------------- Ingestion controls -------------------- #
    bm25_backend: Literal["rank_bm25", "opensearch"] = "rank_bm25"
    ingestion_enrich_limit: int = 500
    ingestion_enrich_strategy: Literal["sample_first", "all", "off"] = "sample_first"

    # -------------------- Generation (Phase 3) -------------------- #
    # Tokens reserved for the CONTEXT block in the user prompt. Leaves
    # room for system + question + answer in a 128k-class window.
    generation_context_budget_tokens: int = 6000
    generation_max_answer_tokens: int = 1024
    generation_temperature: float = 0.1
    # Refuse to invoke the LLM when the top retrieved hit's aggregate
    # score is below this floor. 0.0 disables the gate; Phase 3 · Slice 2
    # tunes this against the eval set.
    generation_min_score_threshold: float = 0.0
    # Whether to include each chunk's summary alongside its parent
    # content in the context block. Costs ~1 line per hit; pays for
    # itself by improving the cross-encoder framing for short answers.
    generation_include_summaries_in_context: bool = True

    # ---- Faithfulness (NLI) — Phase 3 · Slice 2A ---- #
    # When True, every answer is post-verified by running each cited
    # sentence through an NLI model against its cited source(s). Adds
    # ~1 HF call per cited sentence; ~200–500 ms total on a warm endpoint.
    generation_check_faithfulness: bool = True
    # Entailment probability (0..1) at or above which a sentence is
    # judged SUPPORTED. Below it the sentence is UNSUPPORTED. Slice 2C
    # calibrates against the golden set.
    generation_faithfulness_threshold: float = 0.5
    # Confidence (0..1) below which the CLI surfaces a warning. Not a
    # hard refusal — just a UX signal until calibration data exists.
    generation_min_confidence: float = 0.0

    # ---- Sessions (Redis) — Phase 3 · Slice 2B ---- #
    # Rolling TTL on each session key. Every read AND write resets it,
    # so an active conversation stays alive for `session_ttl_seconds`
    # past the most recent activity. 1h matches typical chat UX.
    session_ttl_seconds: int = 3600
    # Hard cap on retained turns per session. Older turns are dropped
    # before the rewriter sees them. Bounds Redis memory and prompt
    # token cost; 10 covers a 5-minute conversation comfortably.
    session_max_turns: int = 10
    # Key prefix in Redis. Lets multiple apps share one Redis instance.
    session_key_prefix: str = "kb:sess:"

    # Chunking
    parent_chunk_target_tokens: int = 1800
    child_chunk_target_tokens: int = 320
    child_chunk_overlap_tokens: int = 40

    # -------------------- Warm-up -------------------- #
    warmup_on_app_start: bool = False
    warmup_cron_window: str = ""

    # -------------------- Observability -------------------- #
    langsmith_api_key: str = ""
    langsmith_project: str = "acme-kb-search"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # -------------------- Infra (local Docker in demo) -------------------- #
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    postgres_url: str = "postgresql://kb:kb@localhost:5432/kb"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "kb"
    postgres_password: str = ""
    postgres_db: str = "kb"

    redis_url: str = "redis://localhost:6379/0"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""

    tei_url: str = "http://localhost:8080"

    # -------------------- Validators / rails -------------------- #
    @field_validator("hosted_lane_priority", "selfhosted_lane_priority")
    @classmethod
    def _non_empty_priority(cls, v: str) -> str:
        if not v or ":" not in v:
            raise ValueError(f"Invalid lane priority string: {v!r}")
        return v

    @model_validator(mode="after")
    def _refuse_prod_with_override(self) -> Settings:
        if self.app_profile == Profile.PROD and self.demo_allow_hosted_for_selfhosted_lane:
            raise ValueError(
                "Refusing to start: APP_PROFILE=prod is incompatible with "
                "DEMO_ALLOW_HOSTED_FOR_SELFHOSTED_LANE=true"
            )
        return self

    # -------------------- Derived helpers -------------------- #
    @property
    def hosted_lane_providers(self) -> list[tuple[str, str]]:
        return _parse_priority(self.hosted_lane_priority)

    @property
    def selfhosted_lane_providers(self) -> list[tuple[str, str]]:
        return _parse_priority(self.selfhosted_lane_priority)


def _parse_priority(s: str) -> list[tuple[str, str]]:
    """Parse 'openai:gpt-4o-mini,qwen:qwen-max' → [('openai','gpt-4o-mini'), ...]."""
    out: list[tuple[str, str]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        provider, _, model = tok.partition(":")
        out.append((provider.strip(), model.strip()))
    return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Import this, not `Settings()`, everywhere."""
    s = Settings()
    _warn_override_active(s)
    return s


def _warn_override_active(s: Settings) -> None:
    if s.demo_allow_hosted_for_selfhosted_lane:
        logger.warning(
            "DEMO_ALLOW_HOSTED_FOR_SELFHOSTED_LANE=true — both lanes route to "
            "hosted providers. hosted=%s  self_hosted=%s",
            s.hosted_lane_providers,
            s.selfhosted_lane_providers,
        )
