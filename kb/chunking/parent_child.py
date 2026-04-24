"""
Parent-child chunker.

Top-level orchestration for chunking:
    1. Parents: produced by kb.chunking.structural.build_parents — section-sized
       units that the LLM will see as context at answer time.
    2. Children: each parent is sliced into fine-grained chunks that are what
       we actually embed and search. Children carry their parent_id so small-
       to-big retrieval can fetch the parent once a child is matched.

Child-splitting strategy:
    * Within a parent, we split on paragraphs first.
    * If a paragraph exceeds the child-target size, split by sentences,
      greedily packed up to the target size with a small overlap between
      consecutive children (reduces boundary-cut recall loss).
    * Code blocks are kept intact if they fit in 1.5× the target; otherwise
      split on blank lines inside the code block.

Also: this module is where the sensitivity classifier is called per-chunk.
Each ChunkedDocument carries a single sensitivity lane (the most restrictive
across its children) so retrieval/generation can route consistently.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Iterable

from kb.chunking.structural import build_parents
from kb.chunking.tokens import count_tokens
from kb.classifier.sensitivity import SensitivityClassifier
from kb.settings import get_settings
from kb.types import (
    AclPayload,
    ChildChunk,
    ChunkedDocument,
    ParentChunk,
    ParsedDocument,
    SensitivityLane,
)


logger = logging.getLogger(__name__)


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])|(?<=\n)\s*")


def chunk_document(
    parsed: ParsedDocument,
    *,
    classifier: SensitivityClassifier | None = None,
) -> ChunkedDocument:
    settings = get_settings()
    classifier = classifier or SensitivityClassifier()

    parent_dicts = build_parents(
        parsed.blocks,
        parent_target_tokens=settings.parent_chunk_target_tokens,
    )

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    per_chunk_lanes: list[SensitivityLane] = []

    for p_idx, p in enumerate(parent_dicts):
        parent_id = _mk_id(parsed.document_id, "p", p_idx)
        parents.append(
            ParentChunk(
                parent_id=parent_id,
                document_id=parsed.document_id,
                ord=p_idx,
                content=p["text"],
                token_count=p["token_count"],
                section_path=p["section_path"],
            )
        )

        for c_idx, child_text in enumerate(
            _split_into_children(
                p["text"],
                target_tokens=settings.child_chunk_target_tokens,
                overlap_tokens=settings.child_chunk_overlap_tokens,
            )
        ):
            child_id = _mk_id(parsed.document_id, "c", p_idx, c_idx)
            child = ChildChunk(
                child_id=child_id,
                parent_id=parent_id,
                document_id=parsed.document_id,
                ord=c_idx,
                content=child_text,
                token_count=count_tokens(child_text),
                section_path=p["section_path"],
                metadata={"parent_ord": p_idx},
            )
            children.append(child)

            decision = classifier.classify(
                text=child_text,
                source_id=parsed.source_id,
                source_default=parsed.default_sensitivity,
            )
            per_chunk_lanes.append(decision.lane)
            if decision.tags:
                child.metadata["sensitivity_tags"] = decision.tags

    # Document-level lane = most restrictive across children. If any child is
    # self_hosted_only, the whole document rides the self-hosted lane. This
    # keeps generation-time routing simple and safe.
    if any(l == SensitivityLane.SELF_HOSTED_ONLY for l in per_chunk_lanes):
        lane = SensitivityLane.SELF_HOSTED_ONLY
    elif per_chunk_lanes:
        lane = SensitivityLane.HOSTED_OK
    else:
        lane = parsed.default_sensitivity

    return ChunkedDocument(
        document_id=parsed.document_id,
        source_id=parsed.source_id,
        source_uri=parsed.source_uri,
        title=parsed.title,
        language=parsed.language,
        content_hash=parsed.content_hash,
        format=parsed.format,
        sensitivity=lane,
        visibility=parsed.default_visibility,
        acl=parsed.default_acl,
        parents=parents,
        children=children,
        metadata=parsed.metadata,
    )


# --------------------------------------------------------------------------- #
# Child splitting
# --------------------------------------------------------------------------- #

def _split_into_children(
    parent_text: str,
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> Iterable[str]:
    """
    Greedy paragraph-then-sentence splitter with a small overlap tail.

    1. Break parent into paragraphs (on blank lines).
    2. Greedy pack paragraphs into children up to target_tokens.
    3. If a single paragraph blows the target, split it on sentences and
       greedy-pack those.
    4. On every flush, prepend the tail of the previous child (overlap) to
       preserve context across boundaries.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", parent_text) if p.strip()]
    if not paragraphs:
        return

    buffer: list[str] = []
    buf_tokens = 0
    tail_carry = ""

    def flush():
        nonlocal buffer, buf_tokens, tail_carry
        if not buffer:
            return None
        body = "\n\n".join(buffer).strip()
        if tail_carry:
            body = f"{tail_carry}\n\n{body}"
        tail_carry = _take_tail(body, overlap_tokens)
        buffer = []
        buf_tokens = 0
        return body

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if para_tokens > target_tokens:
            # Flush what we have, then sentence-split this big paragraph.
            flushed = flush()
            if flushed:
                yield flushed
            yield from _sentence_pack(para, target_tokens, overlap_tokens, tail_carry)
            tail_carry = ""
            continue

        if buf_tokens + para_tokens > target_tokens and buffer:
            flushed = flush()
            if flushed:
                yield flushed

        buffer.append(para)
        buf_tokens += para_tokens

    flushed = flush()
    if flushed:
        yield flushed


def _sentence_pack(
    paragraph: str,
    target_tokens: int,
    overlap_tokens: int,
    initial_tail: str,
) -> Iterable[str]:
    sentences = [s.strip() for s in _SENTENCE_RE.split(paragraph) if s.strip()]
    buffer: list[str] = []
    buf_tokens = 0
    tail_carry = initial_tail

    def flush():
        nonlocal buffer, buf_tokens, tail_carry
        if not buffer:
            return None
        body = " ".join(buffer).strip()
        if tail_carry:
            body = f"{tail_carry}\n\n{body}"
        tail_carry = _take_tail(body, overlap_tokens)
        buffer = []
        buf_tokens = 0
        return body

    for sent in sentences:
        s_tokens = count_tokens(sent)
        if buf_tokens + s_tokens > target_tokens and buffer:
            flushed = flush()
            if flushed:
                yield flushed
        buffer.append(sent)
        buf_tokens += s_tokens

    flushed = flush()
    if flushed:
        yield flushed


def _take_tail(text: str, overlap_tokens: int) -> str:
    """Return the tail of `text` worth roughly `overlap_tokens` tokens."""
    if overlap_tokens <= 0 or not text:
        return ""
    # Cheap char-based approximation: 4 chars ≈ 1 token for English.
    approx_chars = overlap_tokens * 4
    if len(text) <= approx_chars:
        return text
    return text[-approx_chars:]


def _mk_id(document_id: str, kind: str, *parts: int) -> str:
    suffix = "-".join(str(p) for p in parts)
    base = f"{document_id}::{kind}::{suffix}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"{kind}_{h}"
