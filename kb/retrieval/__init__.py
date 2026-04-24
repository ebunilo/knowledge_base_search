"""Query-time retrieval — hybrid dense + sparse with ACL enforcement."""

from kb.retrieval.rerank import CrossEncoderReranker, RerankerError
from kb.retrieval.retriever import Retriever, search
from kb.retrieval.rewrite import QueryRewriter, RewriteResult
from kb.retrieval.types import (
    MatchVia,
    RetrievalConfig,
    RetrievalHit,
    RetrievalResult,
    UserContext,
)

__all__ = [
    "Retriever",
    "search",
    "UserContext",
    "RetrievalConfig",
    "RetrievalHit",
    "RetrievalResult",
    "MatchVia",
    "QueryRewriter",
    "RewriteResult",
    "CrossEncoderReranker",
    "RerankerError",
]
