"""Chunk enrichment — hypothetical questions and 1-line summaries."""

from kb.enrichment.enricher import Enricher, enrich_document
from kb.enrichment.llm_client import LLMClient, LLMClientError

__all__ = ["Enricher", "enrich_document", "LLMClient", "LLMClientError"]
