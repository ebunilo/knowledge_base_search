"""
Core data contracts for the ingestion and retrieval pipeline.

These are the types that flow between stages. Each stage takes one shape
and returns the next shape. Everything is a pydantic model so we get
free validation, JSON-serializability, and stable schemas for testing.

Stage flow (ingestion):
    RawDocument  → ParsedDocument  → ChunkedDocument
                 (parser)           (chunker + classifier)

    ChunkedDocument  → EnrichedDocument  → EmbeddedDocument  → indexed
                      (enrichment)        (embeddings)        (writers)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Enums shared across stages
# --------------------------------------------------------------------------- #

class SensitivityLane(str, Enum):
    HOSTED_OK = "hosted_ok"
    SELF_HOSTED_ONLY = "self_hosted_only"


class Visibility(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"


class DocumentFormat(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    YAML = "yaml"
    JSON = "json"
    TEXT = "text"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Access-control payload (carried as JSONB in Postgres + payload in Qdrant)
# --------------------------------------------------------------------------- #

class AclPayload(BaseModel):
    """Access-control envelope. A query is allowed if the user satisfies:

        (visibility == public)
        OR (user.department in departments AND user.role >= min_role)
        OR (any of user.extra_grants matches this doc's source_id)
    """

    departments: list[str] = Field(default_factory=lambda: ["*"])
    min_role: str = "anonymous"
    tags: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Stage 1: RawDocument — emitted by connectors
# --------------------------------------------------------------------------- #

class RawDocument(BaseModel):
    """What a connector hands to the parser."""

    document_id: str                          # stable id, usually source_id + path hash
    source_id: str                            # matches source_inventory.json
    source_uri: str                           # where we read it from
    title: str | None = None
    content_bytes: bytes = Field(repr=False)  # raw file payload
    content_hash: str                         # sha256 of content_bytes
    format_hint: DocumentFormat = DocumentFormat.UNKNOWN
    language: str = "en"
    region: str | None = None

    # Defaults propagated from source_inventory.json; parser/classifier may override
    default_sensitivity: SensitivityLane = SensitivityLane.SELF_HOSTED_ONLY
    default_visibility: Visibility = Visibility.INTERNAL
    default_acl: AclPayload = Field(default_factory=AclPayload)

    # Connector-level metadata that survives through the pipeline
    metadata: dict[str, Any] = Field(default_factory=dict)

    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True


# --------------------------------------------------------------------------- #
# Stage 2: ParsedDocument — emitted by parsers
# --------------------------------------------------------------------------- #

class ParsedBlock(BaseModel):
    """
    A parser's unit of output. Blocks are the raw material for the chunker;
    they preserve logical structure (headings, paragraphs, code fences, tables,
    JSON/YAML paths) so the chunker can split on natural boundaries.
    """

    kind: Literal[
        "heading", "paragraph", "list_item", "code", "table",
        "json_leaf", "yaml_leaf", "raw",
    ]
    text: str
    # Ordered path of section headings leading to this block, e.g.
    # ["Introduction", "Getting Started", "Prerequisites"]. For YAML/JSON,
    # this is the key path, e.g. ["services", "auth", "timeout"].
    section_path: list[str] = Field(default_factory=list)
    # Depth of heading (1-6) or structured key (>=1). None for non-hierarchical blocks.
    level: int | None = None
    # Source-anchored hints useful for citations.
    page: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    document_id: str
    source_id: str
    source_uri: str
    title: str | None
    language: str
    content_hash: str
    format: DocumentFormat
    blocks: list[ParsedBlock]
    default_sensitivity: SensitivityLane
    default_visibility: Visibility
    default_acl: AclPayload
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("blocks")
    @classmethod
    def _non_empty(cls, v: list[ParsedBlock]) -> list[ParsedBlock]:
        if not v:
            raise ValueError("ParsedDocument must contain at least one block")
        return v


# --------------------------------------------------------------------------- #
# Stage 3: ChunkedDocument — structural + parent-child splits
# --------------------------------------------------------------------------- #

class ParentChunk(BaseModel):
    """
    Section-level chunk. This is what we return to the LLM as context.
    Typically ~1500-2500 tokens.
    """

    parent_id: str
    document_id: str
    ord: int
    content: str
    token_count: int
    section_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChildChunk(BaseModel):
    """
    Fine-grained chunk. This is what we embed and search.
    Typically ~200-400 tokens.
    """

    child_id: str
    parent_id: str
    document_id: str
    ord: int
    content: str
    token_count: int
    section_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkedDocument(BaseModel):
    document_id: str
    source_id: str
    source_uri: str
    title: str | None
    language: str
    content_hash: str
    format: DocumentFormat
    sensitivity: SensitivityLane
    visibility: Visibility
    acl: AclPayload
    parents: list[ParentChunk]
    children: list[ChildChunk]
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Stage 4: EnrichedDocument — after hypothetical questions / summaries
# --------------------------------------------------------------------------- #

class EnrichedChildChunk(ChildChunk):
    hypothetical_questions: list[str] = Field(default_factory=list)
    summary: str | None = None


class EnrichedDocument(ChunkedDocument):
    children: list[EnrichedChildChunk]  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Stage 5: EmbeddedDocument — ready for indexing
# --------------------------------------------------------------------------- #

class EmbeddedChildChunk(EnrichedChildChunk):
    # One vector per question (question-vector retrieval) + one for content
    content_vector: list[float]
    question_vectors: list[list[float]] = Field(default_factory=list)


class EmbeddedDocument(EnrichedDocument):
    children: list[EmbeddedChildChunk]  # type: ignore[assignment]
    embed_model: str
    embed_dim: int


# --------------------------------------------------------------------------- #
# Ingestion run summary (what the orchestrator returns)
# --------------------------------------------------------------------------- #

class IngestStats(BaseModel):
    source_id: str
    documents_seen: int = 0
    documents_skipped_unchanged: int = 0
    documents_processed: int = 0
    documents_failed: int = 0
    parents_written: int = 0
    children_written: int = 0
    vectors_written: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = Field(default_factory=list)
