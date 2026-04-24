"""
Embedding client.

Two backends, chosen via `HF_EMBED_MODE`:
    * "serverless" (default in demo)
          POST https://router.huggingface.co/hf-inference/models/<model>/pipeline/feature-extraction
          Bearer token = HF_API_TOKEN.
    * "endpoint" (prod, demo-isolated)
          POST <HF_EMBED_ENDPOINT_URL>
          Used when we've provisioned a dedicated endpoint for throughput/SLA.

A third mode is planned but not wired here:
    * "local_tei" — POST http://tei:80/embed against the TEI container. Easy
      to bolt on later; just swap the URL and headers.

The client:
    * Batches with a conservative max-batch-tokens ceiling (16 384, matching
      the TEI default) so we don't exceed provider-side limits.
    * Caches the output dim on first successful call.
    * Retries transient errors with exponential backoff (1s, 2s, 4s).
    * Returns a plain `list[list[float]]` — callers decide how to layout
      vectors onto chunks and questions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from kb.chunking.tokens import count_tokens
from kb.settings import Settings, get_settings
from kb.types import EmbeddedChildChunk, EmbeddedDocument, EnrichedDocument


logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised on non-recoverable embedding failures."""


@dataclass
class _Backend:
    url: str
    token: str
    mode: str          # "serverless" or "endpoint"
    model: str


class EmbeddingClient:
    """Synchronous batched embeddings client."""

    MAX_BATCH_TOKENS = 14_000   # leave headroom below the 16 384 TEI default
    MAX_BATCH_ITEMS = 64
    TIMEOUT_S = 60

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._backend = self._build_backend(self.settings)
        self._dim: int | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def model(self) -> str:
        return self._backend.model

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Probe once with a tiny input to learn the vector dim.
            vec = self.embed(["dim probe"])[0]
            self._dim = len(vec)
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return vectors in the same order."""
        if not texts:
            return []

        out: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
        for batch_indices in self._batch_indices(texts):
            batch_texts = [texts[i] for i in batch_indices]
            vectors = self._embed_batch(batch_texts)
            if len(vectors) != len(batch_texts):
                raise EmbeddingError(
                    f"embedding backend returned {len(vectors)} vectors for "
                    f"{len(batch_texts)} inputs"
                )
            for idx, vec in zip(batch_indices, vectors):
                out[idx] = vec

        # Record the dim if we learned it implicitly
        if self._dim is None and out and out[0]:
            self._dim = len(out[0])
        return out

    # ------------------------------------------------------------------ #
    # Backend selection
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_backend(settings: Settings) -> _Backend:
        model = settings.hf_embed_model_id
        if settings.hf_embed_mode == "endpoint":
            url = settings.hf_embed_endpoint_url.rstrip("/")
            if not url:
                raise EmbeddingError(
                    "HF_EMBED_MODE=endpoint but HF_EMBED_ENDPOINT_URL is empty"
                )
            return _Backend(url=url, token=settings.hf_api_token, mode="endpoint", model=model)

        # serverless
        url = (
            f"https://router.huggingface.co/hf-inference/models/"
            f"{model}/pipeline/feature-extraction"
        )
        if not settings.hf_api_token:
            raise EmbeddingError("HF_API_TOKEN is not configured for serverless embeddings")
        return _Backend(url=url, token=settings.hf_api_token, mode="serverless", model=model)

    # ------------------------------------------------------------------ #
    # Batching
    # ------------------------------------------------------------------ #

    def _batch_indices(self, texts: list[str]):
        """
        Yield lists of indices that fit within the batch-token and batch-item
        ceilings. Preserves original order; each index appears exactly once.
        """
        buf: list[int] = []
        buf_tokens = 0
        for i, t in enumerate(texts):
            t_tokens = count_tokens(t) if t else 1
            if buf and (
                buf_tokens + t_tokens > self.MAX_BATCH_TOKENS
                or len(buf) >= self.MAX_BATCH_ITEMS
            ):
                yield buf
                buf, buf_tokens = [], 0
            buf.append(i)
            buf_tokens += t_tokens
        if buf:
            yield buf

    # ------------------------------------------------------------------ #
    # HTTP
    # ------------------------------------------------------------------ #

    def _embed_batch(self, batch_texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self._backend.token}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"inputs": batch_texts}

        last_exc: Exception | None = None
        for attempt, delay in enumerate([0, 1, 2, 4], start=1):
            if delay:
                time.sleep(delay)
            try:
                r = requests.post(
                    self._backend.url, json=payload, headers=headers,
                    timeout=self.TIMEOUT_S,
                )
                # 503 "model is loading" on serverless — retry
                if r.status_code == 503:
                    last_exc = EmbeddingError(f"503 model loading: {r.text[:120]}")
                    continue
                r.raise_for_status()
                data = r.json()
                return _normalize_vectors(data, expected=len(batch_texts))
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "embed attempt %d/%d failed: %s", attempt, 4, exc
                )

        raise EmbeddingError(f"embed failed after retries: {last_exc}")


# --------------------------------------------------------------------------- #
# Response normalisation
# --------------------------------------------------------------------------- #

def _normalize_vectors(data: Any, *, expected: int) -> list[list[float]]:
    """
    The HF Inference API (for bge-m3 via feature-extraction) returns either:
        * list[list[float]]                — one vector per input (flat)
        * list[list[list[float]]]          — per-token embeddings; we mean-pool
    Some models / endpoints may also return a single flat list if expected=1.
    """
    if not isinstance(data, list):
        raise EmbeddingError(f"unexpected embedding response type: {type(data)}")

    # Case 1: a single flat vector was returned even though we sent >1 input.
    # Very rare but has been seen on some endpoints — we reject explicitly.
    if data and isinstance(data[0], (int, float)):
        if expected == 1:
            return [list(map(float, data))]
        raise EmbeddingError(
            "backend returned a single flat vector for a multi-input batch"
        )

    # Case 2: list[list[float]]  — exactly one vector per input
    if data and isinstance(data[0], list) and data[0] and isinstance(data[0][0], (int, float)):
        if len(data) != expected:
            raise EmbeddingError(
                f"expected {expected} vectors, got {len(data)}"
            )
        return [list(map(float, v)) for v in data]

    # Case 3: list[list[list[float]]]  — per-token embeddings; mean-pool per input
    if data and isinstance(data[0], list) and data[0] and isinstance(data[0][0], list):
        pooled: list[list[float]] = []
        for per_input in data:
            pooled.append(_mean_pool(per_input))
        if len(pooled) != expected:
            raise EmbeddingError(
                f"expected {expected} pooled vectors, got {len(pooled)}"
            )
        return pooled

    raise EmbeddingError(f"unexpected embedding response shape (len={len(data)})")


def _mean_pool(token_vectors: list[list[float]]) -> list[float]:
    if not token_vectors:
        raise EmbeddingError("empty per-token output")
    dim = len(token_vectors[0])
    acc = [0.0] * dim
    for v in token_vectors:
        for i in range(dim):
            acc[i] += v[i]
    n = len(token_vectors)
    return [x / n for x in acc]


# --------------------------------------------------------------------------- #
# Convenience: embed a whole enriched document
# --------------------------------------------------------------------------- #

def embed_document(
    enriched: EnrichedDocument,
    client: EmbeddingClient | None = None,
) -> EmbeddedDocument:
    """
    Embed every child's content and its hypothetical questions, and return
    the resulting EmbeddedDocument. The caller can then hand it to the
    MultiIndexWriter.
    """
    client = client or EmbeddingClient()

    # Build a flat list of texts to embed, remembering where each came from.
    # Layout: [c0_content, c0_q0, c0_q1, ..., c1_content, c1_q0, ...]
    texts: list[str] = []
    positions: list[tuple[int, str, int]] = []  # (child_idx, 'content'|'q', q_idx)

    for c_idx, child in enumerate(enriched.children):
        texts.append(child.content)
        positions.append((c_idx, "content", -1))
        for q_idx, q in enumerate(child.hypothetical_questions):
            texts.append(q)
            positions.append((c_idx, "q", q_idx))

    vectors: list[list[float]] = client.embed(texts) if texts else []
    if len(vectors) != len(texts):
        raise EmbeddingError(
            f"embedding count mismatch: {len(vectors)} vectors for {len(texts)} texts"
        )

    # Lay out per-child vectors
    content_vecs: dict[int, list[float]] = {}
    q_vecs: dict[int, list[list[float]]] = {i: [] for i in range(len(enriched.children))}
    for (c_idx, kind, _q_idx), vec in zip(positions, vectors):
        if kind == "content":
            content_vecs[c_idx] = vec
        else:
            q_vecs[c_idx].append(vec)

    new_children: list[EmbeddedChildChunk] = []
    for c_idx, child in enumerate(enriched.children):
        new_children.append(
            EmbeddedChildChunk(
                **child.model_dump(),
                content_vector=content_vecs.get(c_idx, []),
                question_vectors=q_vecs.get(c_idx, []),
            )
        )

    base = enriched.model_dump(exclude={"children"})
    return EmbeddedDocument(
        **base,
        children=new_children,
        embed_model=client.model,
        embed_dim=client.dim,
    )
