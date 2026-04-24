"""Document parsers — convert RawDocument bytes into ParsedDocument blocks."""

from kb.parsers.router import ParserError, parse_document

__all__ = ["parse_document", "ParserError"]
