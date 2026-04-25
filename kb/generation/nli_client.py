"""
NLI client — Hugging Face Serverless.

Wraps the same `pipeline/zero-shot-classification` endpoint the Phase-0
smoke test validated, plus the classic HF Inference API as a fallback.
The model is `facebook/bart-large-mnli` by default (configurable).

API usage trick: zero-shot-classification internally constructs an MNLI
input as `(premise, hypothesis_template.format(label))`. By passing
`candidate_labels=[hypothesis]` and `hypothesis_template="{}"`, the
underlying NLI input becomes literally `(premise, hypothesis)` and the
returned score is the entailment probability — exactly what we want for
faithfulness.

Response handling mirrors the smoke test (and the rerank client) — the
HF inference router emits at least three shapes for this endpoint:
    * dict:  `{"sequence": "...", "labels": ["..."], "scores": [...]}`
    * list:  `[{"label": "...", "score": ...}, ...]`
    * dict:  `{"labels": [...], "scores": [...]}`  (no "sequence")

We tolerate all three. Cold starts (HTTP 503) are retried with backoff.
On unrecoverable failure the client raises `NLIClientError`; the
checker turns that into a graceful degrade with `fallback_reason` set.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from kb.settings import Settings, get_settings


logger = logging.getLogger(__name__)


_HF_INFERENCE_BASE = "https://router.huggingface.co/hf-inference/models"
_HF_CLASSIC_BASE = "https://api-inference.huggingface.co/models"


class NLIClientError(Exception):
    """Raised when every endpoint failed for an NLI request."""


class NLIClient:
    """
    Single-pair NLI client with batching helpers.

    Thread-safe (stateless). Requests use one HTTP call per (premise,
    hypothesis) pair — the HF zero-shot pipeline doesn't natively batch
    pairs. Slice 2C will profile this on the eval harness; if it
    dominates total latency we'll switch to the raw text-classification
    endpoint with explicit `[premise] </s> [hypothesis]` batching.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model_id: str | None = None,
        timeout_s: int = 30,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_id = model_id or self.settings.hf_nli_model_id
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def entailment_score(self, premise: str, hypothesis: str) -> float:
        """
        Return the probability (0..1) that `premise` entails `hypothesis`.

        Trivial inputs (either side empty after strip) return 0.0 without
        a network round-trip — there is no claim to verify.
        """
        premise = (premise or "").strip()
        hypothesis = (hypothesis or "").strip()
        if not premise or not hypothesis:
            return 0.0

        payload = self._build_payload(premise=premise, hypothesis=hypothesis)
        data = self._post_with_fallback(payload)
        score = self._extract_score(data, hypothesis)
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------ #
    # Request shaping
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_payload(*, premise: str, hypothesis: str) -> dict[str, Any]:
        return {
            "inputs": premise,
            "parameters": {
                "candidate_labels": [hypothesis],
                # No decoration — we want the literal hypothesis as the
                # MNLI hypothesis side.
                "hypothesis_template": "{}",
                "multi_label": True,
            },
        }

    def _post_with_fallback(self, payload: dict[str, Any]) -> Any:
        """
        Try router → router/pipeline → classic, with cold-start retries.

        Same fallback chain as `scripts/smoke_test.py::test_hf_nli`,
        which we already know works against this account.
        """
        urls = [
            f"{_HF_CLASSIC_BASE}/{self.model_id}",
            f"{_HF_INFERENCE_BASE}/{self.model_id}",
            f"{_HF_INFERENCE_BASE}/{self.model_id}/pipeline/zero-shot-classification",
        ]
        last_err: Exception | None = None
        for url in urls:
            try:
                return self._post_with_retries(url, payload)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.debug("NLI endpoint %s failed: %s", url, exc)
                continue
        raise NLIClientError(f"all NLI endpoints failed; last={last_err}")

    def _post_with_retries(self, url: str, payload: dict[str, Any]) -> Any:
        """One URL, with exponential backoff on 503 cold-starts."""
        token = self.settings.hf_api_token
        if not token:
            raise NLIClientError("HF_API_TOKEN is not configured")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.max_retries):
            resp = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout_s,
            )
            if resp.status_code == 503 and attempt < self.max_retries - 1:
                wait = self.backoff_base_s * (2 ** attempt)
                logger.info(
                    "NLI 503 (cold start) on %s, retrying in %.1fs", url, wait,
                )
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                raise NLIClientError(
                    f"{resp.status_code} {resp.reason} on {url}: "
                    f"{resp.text[:200]}"
                )
            return resp.json()
        raise NLIClientError(f"NLI: max retries exhausted on {url}")

    # ------------------------------------------------------------------ #
    # Response parsing — three shapes (see module docstring)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_score(data: Any, hypothesis: str) -> float:
        """
        Pull the entailment probability out of one of the three known
        response shapes. Falls back to 0.0 on unrecognised shapes rather
        than raising — at this point the HTTP call already succeeded;
        a parse miss should be a soft signal, not an outage.
        """
        # Shape A: {"sequence": ..., "labels": [...], "scores": [...]}
        if isinstance(data, dict) and "scores" in data:
            scores = data.get("scores") or []
            labels = data.get("labels") or []
            if not scores:
                return 0.0
            # When candidate_labels=[hypothesis], labels[0] should be that
            # hypothesis. Pick by exact match if present, else first.
            for lbl, sc in zip(labels, scores):
                if isinstance(lbl, str) and lbl.strip() == hypothesis.strip():
                    return float(sc)
            return float(scores[0])

        # Shape B: [{"label": "...", "score": ...}, ...]
        if isinstance(data, list) and data and isinstance(data[0], dict) and "score" in data[0]:
            for item in data:
                if str(item.get("label", "")).strip() == hypothesis.strip():
                    return float(item.get("score", 0.0))
            return float(data[0].get("score", 0.0))

        # Shape C (rare): bare list of floats — single-pair endpoints
        # sometimes return [score].
        if isinstance(data, list) and data and isinstance(data[0], (int, float)):
            return float(data[0])

        logger.warning("NLI: unrecognised response shape: %.200s", str(data))
        return 0.0
