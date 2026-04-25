"""
Tests for `kb.generation.segmentation`.

The splitter is a regex-based heuristic — these tests pin behaviour we
care about:
    * Sentences are returned in document order with leading/trailing
      whitespace stripped.
    * `[N]` markers are extracted per sentence, deduplicated, and kept
      in first-appearance order.
    * Common abbreviations don't end a sentence.
    * Very short fragments (citation-only leftovers) are dropped.
"""

from __future__ import annotations

from kb.generation.segmentation import split_sentences


class TestBasicSplitting:
    def test_empty_input(self):
        assert split_sentences("") == []
        assert split_sentences("   \n\n  ") == []

    def test_single_sentence(self):
        out = split_sentences("The gateway authenticates via OAuth.")
        assert len(out) == 1
        assert out[0].text == "The gateway authenticates via OAuth."
        assert out[0].markers == ()

    def test_multiple_sentences_in_order(self):
        text = (
            "The gateway authenticates via OAuth. "
            "Tokens last one hour. "
            "Refresh requires the same client id."
        )
        out = split_sentences(text)
        assert len(out) == 3
        assert out[0].text.startswith("The gateway")
        assert out[1].text == "Tokens last one hour."
        assert out[2].text.startswith("Refresh")

    def test_strips_whitespace(self):
        out = split_sentences("  Alpha.   Beta!   Gamma?  ", min_chars=2)
        texts = [s.text for s in out]
        assert texts == ["Alpha.", "Beta!", "Gamma?"]


class TestMarkers:
    def test_marker_attached_to_its_sentence(self):
        text = "The gateway uses OAuth [1]. Tokens last one hour [2]."
        out = split_sentences(text)
        assert len(out) == 2
        assert out[0].markers == (1,)
        assert out[1].markers == (2,)

    def test_marker_after_punctuation_doesnt_split(self):
        # "[1]." — the `.` belongs to the marker; we shouldn't break here.
        text = "Webhooks retry up to 8 times [1]. They use exponential backoff [2]."
        out = split_sentences(text)
        assert len(out) == 2
        assert out[0].markers == (1,)
        assert out[1].markers == (2,)

    def test_multiple_markers_in_one_sentence(self):
        text = "It works as documented [1][2] and matches the spec [3]."
        out = split_sentences(text)
        assert len(out) == 1
        assert out[0].markers == (1, 2, 3)

    def test_dedupes_repeated_markers(self):
        text = "First [1]. Then again [1] and once more [1]."
        out = split_sentences(text)
        # Two sentences; each marker list deduped within its sentence.
        assert all(m == (1,) for m in (out[0].markers, out[1].markers))


class TestAbbreviations:
    def test_eg_doesnt_end_a_sentence(self):
        text = "Use SSL terminations, e.g. NGINX or Envoy. They both work."
        out = split_sentences(text)
        assert len(out) == 2
        assert "e.g. NGINX" in out[0].text

    def test_us_doesnt_end_a_sentence(self):
        text = "The U.S. region is supported. Failover is automatic."
        out = split_sentences(text)
        assert len(out) == 2
        assert out[0].text.startswith("The U.S.")

    def test_etc_doesnt_end_a_sentence(self):
        # `etc.` is in our abbreviation list — both halves stay together.
        text = "Supports JSON, YAML, etc. The parser auto-detects format."
        out = split_sentences(text)
        assert len(out) == 1
        assert out[0].text == text


class TestFiltering:
    def test_drops_short_fragments(self):
        # The trailing "OK." is below the 10-char minimum; drop it.
        text = "The pipeline is healthy. OK."
        out = split_sentences(text)
        assert len(out) == 1
        assert out[0].text == "The pipeline is healthy."

    def test_min_chars_configurable(self):
        out = split_sentences("Hi. Hello.", min_chars=2)
        assert [s.text for s in out] == ["Hi.", "Hello."]

    def test_question_and_exclamation(self):
        text = (
            "Does it really scale well? "
            "Yes it does! "
            "The benchmark hit 10k QPS."
        )
        out = split_sentences(text)
        assert len(out) == 3
