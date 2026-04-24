"""
Tests for `kb.retrieval.rerank`.

We mock the HTTP layer so these tests run offline. The reranker's job is
to take a list of passages and return them sorted by descending relevance
score — the tests assert that contract plus the several response shapes
the HF Inference Router emits.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kb.retrieval.rerank import (
    CrossEncoderReranker,
    RerankerError,
    RerankResult,
)
from kb.settings import Settings


def _reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(
        Settings(
            hf_api_token="test",
            hf_rerank_mode="serverless",
            hf_rerank_model_id="BAAI/bge-reranker-v2-m3",
        )
    )


class _FakeResp:
    def __init__(self, json_body, status_code: int = 200, text: str = ""):
        self._json = json_body
        self.status_code = status_code
        self.ok = status_code < 400
        self.reason = "OK" if self.ok else "ERR"
        self.text = text or repr(json_body)

    def json(self):
        return self._json


# --------------------------------------------------------------------------- #
# Happy path — multiple shapes
# --------------------------------------------------------------------------- #

class TestRerankHappyPath:
    def test_flat_list_of_floats(self):
        rr = _reranker()
        resp = _FakeResp([0.2, 0.9, 0.5])
        with patch("kb.retrieval.rerank.requests.post", return_value=resp):
            results = rr.rerank(query="q", passages=["a", "b", "c"])
        assert [r.index for r in results] == [1, 2, 0]
        assert results[0].score == pytest.approx(0.9)
        assert results[-1].score == pytest.approx(0.2)

    def test_list_of_dict_scores(self):
        rr = _reranker()
        resp = _FakeResp([{"score": 0.1}, {"score": 0.7}])
        with patch("kb.retrieval.rerank.requests.post", return_value=resp):
            results = rr.rerank(query="q", passages=["a", "b"])
        assert [r.index for r in results] == [1, 0]

    def test_dict_with_scores_key(self):
        rr = _reranker()
        resp = _FakeResp({"scores": [0.3, 0.4, 0.1]})
        with patch("kb.retrieval.rerank.requests.post", return_value=resp):
            results = rr.rerank(query="q", passages=["a", "b", "c"])
        assert [r.index for r in results] == [1, 0, 2]

    def test_preserves_order_for_ties(self):
        rr = _reranker()
        resp = _FakeResp([0.5, 0.5, 0.5])
        with patch("kb.retrieval.rerank.requests.post", return_value=resp):
            results = rr.rerank(query="q", passages=["a", "b", "c"])
        # Tie-break by input index (stable).
        assert [r.index for r in results] == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #

class TestBatching:
    def test_batches_when_over_max(self):
        rr = _reranker()
        rr.MAX_BATCH = 2  # force two batches for 3 inputs

        call_log: list[list[str]] = []

        def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
            call_log.append(json["inputs"]["sentences"])
            # Return descending scores so we can verify the merge order.
            scores = [1.0 - 0.1 * i for i in range(len(json["inputs"]["sentences"]))]
            return _FakeResp(scores)

        with patch("kb.retrieval.rerank.requests.post", side_effect=fake_post):
            results = rr.rerank(query="q", passages=["a", "b", "c"])

        assert len(call_log) == 2
        assert call_log[0] == ["a", "b"]
        assert call_log[1] == ["c"]
        assert len(results) == 3
        # Both batches started at 1.0; the first item of each batch should
        # tie but the first batch wins on stable index order.
        assert results[0].index in {0, 2}


# --------------------------------------------------------------------------- #
# Error / degenerate paths
# --------------------------------------------------------------------------- #

class TestErrorHandling:
    def test_empty_passages_short_circuits(self):
        rr = _reranker()
        # No HTTP call should be made.
        with patch("kb.retrieval.rerank.requests.post") as p:
            assert rr.rerank(query="q", passages=[]) == []
            assert p.call_count == 0

    def test_empty_query_returns_input_order(self):
        rr = _reranker()
        with patch("kb.retrieval.rerank.requests.post") as p:
            results = rr.rerank(query="   ", passages=["a", "b"])
            assert p.call_count == 0
        assert [r.index for r in results] == [0, 1]
        assert all(r.score == 0.0 for r in results)

    def test_5xx_is_raised_after_retries(self):
        rr = _reranker()
        rr.RETRIES = 1
        rr.RETRY_BACKOFF_S = (0.0, 0.0)
        resp = _FakeResp({"error": "nope"}, status_code=500, text='{"error":"nope"}')
        with patch("kb.retrieval.rerank.requests.post", return_value=resp):
            with pytest.raises(RerankerError):
                rr.rerank(query="q", passages=["a", "b"])

    def test_cold_start_503_retries_then_succeeds(self):
        rr = _reranker()
        rr.RETRIES = 2
        rr.RETRY_BACKOFF_S = (0.0, 0.0)

        responses = [
            _FakeResp({"error": "loading"}, status_code=503, text='{"error":"loading"}'),
            _FakeResp([0.1, 0.9]),
        ]

        def fake_post(*args, **kwargs):
            return responses.pop(0)

        with patch("kb.retrieval.rerank.requests.post", side_effect=fake_post):
            results = rr.rerank(query="q", passages=["a", "b"])
        assert [r.index for r in results] == [1, 0]

    def test_unknown_shape_raises(self):
        rr = _reranker()
        rr.RETRIES = 0
        resp = _FakeResp({"foo": "bar"})
        with patch("kb.retrieval.rerank.requests.post", return_value=resp):
            with pytest.raises(RerankerError):
                rr.rerank(query="q", passages=["a"])

    def test_count_mismatch_raises(self):
        rr = _reranker()
        resp = _FakeResp([0.1, 0.2])  # 2 scores for 3 inputs
        with patch("kb.retrieval.rerank.requests.post", return_value=resp):
            with pytest.raises(RerankerError):
                rr.rerank(query="q", passages=["a", "b", "c"])
