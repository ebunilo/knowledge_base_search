"""
Cross-encoder reranker — HF Serverless `bge-reranker-v2-m3`.

The fused list out of RRF is *recall-biased*: we pulled top-N (typically
30) from two orthogonal retrievers, so anything broadly relevant tends to
be in there. The cross-encoder's job is precision — score each
(query, passage) pair with a model that sees both together and re-orders
the list by *actual* relevance, not just vector similarity.

Call pattern (matches the smoke-test reranker probe):

    POST {HF_INFERENCE_BASE}/BAAI/bge-reranker-v2-m3/pipeline/sentence-similarity
    {
      "inputs": {
        "source_sentence": "<query>",
        "sentences": ["<passage 1>", "<passage 2>", ...]
      }
    }
    → [0.824, 0.553, ...]     # one score per passage, same order

Design:
    * Degrade gracefully. If the reranker fails (503 cold-start, timeout,
      5xx), we return the input order unchanged and log a warning — the
      request still succeeds, you just lose the precision boost.
    * Batch at most `MAX_BATCH` passages per call so a long list doesn't
      exceed the provider-side request size.
    * Normalise response shape: HF sometimes returns `[score, ...]` or
      `[{"score": ...}, ...]` depending on the pipeline variant.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from kb.settings import Settings, get_settings


logger = logging.getLogger(__name__)


HF_INFERENCE_BASE = "https://router.huggingface.co/hf-inference/models"


class RerankerError(Exception):
    """Raised on unrecoverable reranker failures (the caller catches this)."""


@dataclass
class RerankResult:
    """One result entry: the original candidate + its rerank score."""
    index: int          # index into the input list
    score: float        # raw score from the cross-encoder (higher = better)


class CrossEncoderReranker:
    """Thin HTTP client around the HF cross-encoder."""

    MAX_BATCH = 32
    TIMEOUT_S = 30
    RETRIES = 2
    RETRY_BACKOFF_S = (1.0, 3.0)

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = self.settings.hf_rerank_model_id
        self._mode = self.settings.hf_rerank_mode
        self._token = self.settings.hf_api_token
        self._url = self._build_url()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def model(self) -> str:
        return self._model

    def rerank(
        self,
        *,
        query: str,
        passages: list[str],
    ) -> list[RerankResult]:
        """
        Score every (query, passage) pair and return a list of RerankResults
        sorted by descending score.

        Raises RerankerError if all retries fail.
        """
        if not passages:
            return []
        if not query.strip():
            # Everything scores the same with an empty query; degenerate to
            # input order.
            return [RerankResult(index=i, score=0.0) for i in range(len(passages))]

        scores: list[float] = [0.0] * len(passages)
        for start in range(0, len(passages), self.MAX_BATCH):
            batch = passages[start : start + self.MAX_BATCH]
            batch_scores = self._rerank_batch(query=query, passages=batch)
            if len(batch_scores) != len(batch):
                raise RerankerError(
                    f"reranker returned {len(batch_scores)} scores for "
                    f"{len(batch)} inputs"
                )
            for local_idx, s in enumerate(batch_scores):
                scores[start + local_idx] = s

        results = [RerankResult(index=i, score=s) for i, s in enumerate(scores)]
        results.sort(key=lambda r: (-r.score, r.index))
        return results

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _build_url(self) -> str:
        if self._mode == "endpoint":
            endpoint = self.settings.hf_rerank_endpoint_url.rstrip("/")
            if not endpoint:
                raise RerankerError(
                    "HF_RERANK_MODE=endpoint but HF_RERANK_ENDPOINT_URL is empty"
                )
            return endpoint
        return f"{HF_INFERENCE_BASE}/{self._model}/pipeline/sentence-similarity"

    def _rerank_batch(self, *, query: str, passages: list[str]) -> list[float]:
        payload = {
            "inputs": {"source_sentence": query, "sentences": passages},
            "options": {"wait_for_model": True},
        }
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

        last_err: Exception | None = None
        for attempt in range(self.RETRIES + 1):
            try:
                t0 = time.monotonic()
                r = requests.post(
                    self._url, headers=headers, json=payload,
                    timeout=self.TIMEOUT_S,
                )
                elapsed_ms = int((time.monotonic() - t0) * 1000)

                if r.status_code == 503:
                    # Cold-start on HF Serverless — retryable
                    logger.info(
                        "reranker cold-start (attempt %d/%d): %s",
                        attempt + 1, self.RETRIES + 1, r.text[:120],
                    )
                    last_err = RerankerError(f"503: {r.text[:200]}")
                    self._sleep(attempt)
                    continue
                if not r.ok:
                    raise RerankerError(
                        f"rerank HTTP {r.status_code}: {r.text[:200]}"
                    )

                scores = self._parse_scores(r.json(), expected=len(passages))
                logger.debug(
                    "rerank batch n=%d in %dms model=%s",
                    len(passages), elapsed_ms, self._model,
                )
                return scores
            except requests.Timeout as exc:
                last_err = exc
                logger.warning("reranker timeout (attempt %d): %s", attempt + 1, exc)
                self._sleep(attempt)
            except requests.RequestException as exc:
                last_err = exc
                logger.warning("reranker request failed (attempt %d): %s", attempt + 1, exc)
                self._sleep(attempt)

        raise RerankerError(f"reranker failed after retries: {last_err}")

    def _sleep(self, attempt: int) -> None:
        if attempt < len(self.RETRY_BACKOFF_S):
            time.sleep(self.RETRY_BACKOFF_S[attempt])

    @staticmethod
    def _parse_scores(data: Any, *, expected: int) -> list[float]:
        """
        Normalise the several shapes HF Inference returns for
        sentence-similarity. We accept any of:

            [float, float, ...]                # common
            [{"score": float}, ...]            # some router variants
            {"scores": [float, ...]}           # endpoint variant
            {"similarities": [float, ...]}     # older variant
        """
        if isinstance(data, list):
            if not data:
                return []
            if isinstance(data[0], (int, float)):
                return [float(x) for x in data]
            if isinstance(data[0], dict) and "score" in data[0]:
                return [float(x["score"]) for x in data]
        if isinstance(data, dict):
            for key in ("scores", "similarities", "results"):
                v = data.get(key)
                if isinstance(v, list):
                    if v and isinstance(v[0], (int, float)):
                        return [float(x) for x in v]
                    if v and isinstance(v[0], dict) and "score" in v[0]:
                        return [float(x["score"]) for x in v]
            if "error" in data:
                raise RerankerError(f"reranker API error: {data['error']!r:.200}")
        raise RerankerError(
            f"unexpected reranker response shape (expected {expected} scores): "
            f"{str(data)[:200]}"
        )
