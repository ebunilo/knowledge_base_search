"""
Parser router — dispatches RawDocument to the right concrete parser.

Phase 1 parsers:
    * PDF      — pymupdf4llm → Markdown with headings + pages, pypdf fallback.
    * YAML     — ruamel/pyyaml walker that preserves key-paths for citation.
    * JSON     — same walker, JSON variant.
    * Markdown — ATX-heading splitter.
    * HTML     — basic tag stripper (BeautifulSoup optional; simple regex fallback).
    * TEXT     — single-block passthrough.

Every parser returns a ParsedDocument with a non-empty list of ParsedBlock.
Empty inputs, unreadable content, or format mismatches raise ParserError.
"""

from __future__ import annotations

import logging

from kb.parsers.pdf import parse_pdf
from kb.parsers.structured import parse_json, parse_yaml
from kb.parsers.text import parse_html, parse_markdown, parse_text
from kb.types import DocumentFormat, ParsedDocument, RawDocument


logger = logging.getLogger(__name__)


class ParserError(Exception):
    """Raised when a parser cannot produce any usable blocks."""


def parse_document(raw: RawDocument) -> ParsedDocument:
    """Dispatch to the parser matching raw.format_hint."""
    fmt = raw.format_hint

    try:
        if fmt == DocumentFormat.PDF:
            blocks = parse_pdf(raw)
        elif fmt == DocumentFormat.MARKDOWN:
            blocks = parse_markdown(raw)
        elif fmt == DocumentFormat.HTML:
            blocks = parse_html(raw)
        elif fmt == DocumentFormat.YAML:
            blocks = parse_yaml(raw)
        elif fmt == DocumentFormat.JSON:
            blocks = parse_json(raw)
        elif fmt == DocumentFormat.TEXT:
            blocks = parse_text(raw)
        else:
            raise ParserError(f"No parser registered for format {fmt}")
    except ParserError:
        raise
    except Exception as exc:
        raise ParserError(
            f"Parser for {fmt} failed on {raw.source_uri}: {exc}"
        ) from exc

    if not blocks:
        raise ParserError(f"Parser produced zero blocks for {raw.source_uri}")

    return ParsedDocument(
        document_id=raw.document_id,
        source_id=raw.source_id,
        source_uri=raw.source_uri,
        title=raw.title,
        language=raw.language,
        content_hash=raw.content_hash,
        format=fmt,
        blocks=blocks,
        default_sensitivity=raw.default_sensitivity,
        default_visibility=raw.default_visibility,
        default_acl=raw.default_acl,
        metadata=raw.metadata,
    )
