"""
Tests for `kb.generation.nli_client.NLIClient`.

We mock `requests.post` so these are offline. Coverage:
    * Trivial inputs (empty premise / hypothesis) short-circuit.
    * The three response shapes the smoke test enumerated all parse.
    * The score is clamped to [0, 1].
    * 503 cold starts retry then succeed.
    * Exhausted retries / 4xx fall through to the next URL; if every
      URL fails we raise NLIClientError.
    * The candidate label comes back as the hypothesis (no decoration).
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from kb.generation.nli_client import NLIClient, NLIClientError
from kb.settings import Settings


def _client(**kwargs) -> NLIClient:
    return NLIClient(
        Settings(hf_api_token="test-token"),
        max_retries=2, backoff_base_s=0.0, **kwargs,
    )


def _resp(status: int, body):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body
    r.text = str(body)
    r.reason = "OK" if status < 400 else "ERR"
    return r


# --------------------------------------------------------------------------- #
# Trivial inputs
# --------------------------------------------------------------------------- #

class TestTrivial:
    def test_empty_premise_returns_zero(self):
        c = _client()
        with patch("kb.generation.nli_client.requests.post") as p:
            assert c.entailment_score("", "anything") == 0.0
            assert p.call_count == 0

    def test_empty_hypothesis_returns_zero(self):
        c = _client()
        with patch("kb.generation.nli_client.requests.post") as p:
            assert c.entailment_score("anything", "  ") == 0.0
            assert p.call_count == 0


# --------------------------------------------------------------------------- #
# Response shapes
# --------------------------------------------------------------------------- #

class TestResponseShapes:
    def test_dict_with_labels_and_scores(self):
        body = {
            "sequence": "premise",
            "labels": ["the gateway uses oauth"],
            "scores": [0.83],
        }
        c = _client()
        with patch("kb.generation.nli_client.requests.post", return_value=_resp(200, body)):
            assert c.entailment_score("premise", "the gateway uses oauth") == pytest.approx(0.83)

    def test_dict_picks_score_for_matching_label(self):
        body = {
            "labels": ["foo", "bar baz"],
            "scores": [0.1, 0.7],
        }
        c = _client()
        with patch("kb.generation.nli_client.requests.post", return_value=_resp(200, body)):
            assert c.entailment_score("premise", "bar baz") == pytest.approx(0.7)

    def test_list_of_dicts(self):
        body = [
            {"label": "wrong", "score": 0.2},
            {"label": "the claim", "score": 0.6},
        ]
        c = _client()
        with patch("kb.generation.nli_client.requests.post", return_value=_resp(200, body)):
            assert c.entailment_score("p", "the claim") == pytest.approx(0.6)

    def test_bare_list_of_floats(self):
        c = _client()
        with patch("kb.generation.nli_client.requests.post", return_value=_resp(200, [0.42])):
            assert c.entailment_score("p", "h") == pytest.approx(0.42)

    def test_unknown_shape_returns_zero(self):
        c = _client()
        with patch("kb.generation.nli_client.requests.post", return_value=_resp(200, {"weird": "shape"})):
            assert c.entailment_score("p", "h") == 0.0

    def test_score_clamped_above_one(self):
        c = _client()
        with patch("kb.generation.nli_client.requests.post", return_value=_resp(200, [1.7])):
            assert c.entailment_score("p", "h") == 1.0

    def test_score_clamped_below_zero(self):
        c = _client()
        with patch("kb.generation.nli_client.requests.post", return_value=_resp(200, [-0.4])):
            assert c.entailment_score("p", "h") == 0.0


# --------------------------------------------------------------------------- #
# Retries / fallback
# --------------------------------------------------------------------------- #

class TestRetries:
    def test_503_retries_then_succeeds(self):
        c = _client()
        responses = [_resp(503, {"error": "loading"}), _resp(200, [0.55])]
        with patch("kb.generation.nli_client.requests.post", side_effect=responses) as p:
            assert c.entailment_score("p", "h") == pytest.approx(0.55)
            assert p.call_count == 2

    def test_first_url_4xx_falls_back_to_next(self):
        c = _client()
        # Three URLs are tried in order; first 400, second 200.
        responses = [_resp(400, "bad"), _resp(200, [0.31])]
        with patch("kb.generation.nli_client.requests.post", side_effect=responses) as p:
            assert c.entailment_score("p", "h") == pytest.approx(0.31)
            assert p.call_count == 2

    def test_all_urls_fail_raises(self):
        c = _client()
        with patch(
            "kb.generation.nli_client.requests.post",
            side_effect=[_resp(500, "err"), _resp(500, "err"), _resp(500, "err"),
                         _resp(500, "err"), _resp(500, "err"), _resp(500, "err")],
        ):
            with pytest.raises(NLIClientError):
                c.entailment_score("p", "h")

    def test_missing_token_raises(self):
        c = NLIClient(Settings(hf_api_token=""))
        with pytest.raises(NLIClientError):
            c.entailment_score("p", "h")


# --------------------------------------------------------------------------- #
# Payload shape
# --------------------------------------------------------------------------- #

class TestPayload:
    def test_payload_uses_identity_template(self):
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            return _resp(200, [0.9])

        c = _client()
        with patch("kb.generation.nli_client.requests.post", side_effect=fake_post):
            c.entailment_score("the source content", "the claim")

        assert captured["json"]["inputs"] == "the source content"
        params = captured["json"]["parameters"]
        assert params["candidate_labels"] == ["the claim"]
        assert params["hypothesis_template"] == "{}"
        assert params["multi_label"] is True
