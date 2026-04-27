"""Utilities for cleaning documents before placement under ``sample_docs`` (or other corpora)."""

from kb.preprocessing.pdf_margins import StripMarginsConfig, strip_pdf_margins

__all__ = ["StripMarginsConfig", "strip_pdf_margins"]
