"""
Token counting.

We use tiktoken's `cl100k_base` as a close-enough proxy for both OpenAI GPT-4o
and Qwen tokenizers. The numbers won't match perfectly (especially for CJK)
but for English technical content they are within a few percent, which is
plenty for sizing chunks.

The `count_tokens` helper is cached per-string to keep the chunker cheap
when it probes the same prefix repeatedly while greedy-packing.
"""

from __future__ import annotations

from functools import lru_cache


_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = "fallback"
    return _ENCODER


@lru_cache(maxsize=4096)
def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _get_encoder()
    if enc == "fallback":
        # Rough heuristic: ~4 chars per token for English. Overestimates for
        # code/CJK but keeps us operational when tiktoken is unavailable.
        return max(1, len(text) // 4)
    return len(enc.encode(text))
