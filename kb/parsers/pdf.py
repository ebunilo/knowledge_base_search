"""
PDF parser.

Strategy:
    1. Primary: pymupdf4llm.to_markdown() — preserves headings, lists, tables,
       and produces a Markdown string we can split on ATX headings. This keeps
       the downstream chunker's section-path logic consistent across formats.
    2. Fallback: pypdf — extract raw text per page, emit one paragraph block
       per page. No heading structure, but at least keeps the document
       citable by page.

We only fall back when pymupdf4llm raises or returns empty output.
"""

from __future__ import annotations

import io
import logging
import re

from kb.types import ParsedBlock, RawDocument


logger = logging.getLogger(__name__)


_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def parse_pdf(raw: RawDocument) -> list[ParsedBlock]:
    blocks = _try_pymupdf4llm(raw)
    if blocks:
        return blocks

    logger.info("pymupdf4llm produced no blocks for %s; falling back to pypdf", raw.source_uri)
    return _try_pypdf(raw)


# --------------------------------------------------------------------------- #
# Primary path: pymupdf4llm → Markdown → block list
# --------------------------------------------------------------------------- #

def _try_pymupdf4llm(raw: RawDocument) -> list[ParsedBlock]:
    try:
        import pymupdf                 # type: ignore
        import pymupdf4llm             # type: ignore
    except ImportError:
        logger.debug("pymupdf4llm not installed; skipping primary PDF path")
        return []

    try:
        doc = pymupdf.open(stream=raw.content_bytes, filetype="pdf")
    except Exception as exc:
        logger.warning("pymupdf could not open %s: %s", raw.source_uri, exc)
        return []

    try:
        md = pymupdf4llm.to_markdown(doc, show_progress=False)
    except Exception as exc:
        logger.warning("pymupdf4llm.to_markdown failed on %s: %s", raw.source_uri, exc)
        doc.close()
        return []
    finally:
        try:
            doc.close()
        except Exception:
            pass

    if not md or not md.strip():
        return []

    return _markdown_to_blocks(md)


def _markdown_to_blocks(md: str) -> list[ParsedBlock]:
    """
    Walk a Markdown string, maintaining a heading stack so every content block
    carries its full section_path. Consecutive non-heading lines are coalesced
    into a single paragraph block until the next heading or blank separator.
    """
    blocks: list[ParsedBlock] = []
    heading_stack: list[tuple[int, str]] = []  # (level, title)
    buffer: list[str] = []

    def flush_buffer():
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        if text:
            blocks.append(
                ParsedBlock(
                    kind="paragraph",
                    text=text,
                    section_path=[h[1] for h in heading_stack],
                )
            )
        buffer.clear()

    for line in md.splitlines():
        m = _ATX_HEADING_RE.match(line)
        if m:
            flush_buffer()
            level = len(m.group(1))
            title = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            blocks.append(
                ParsedBlock(
                    kind="heading",
                    text=title,
                    section_path=[h[1] for h in heading_stack],
                    level=level,
                )
            )
        elif line.strip() == "":
            flush_buffer()
        else:
            buffer.append(line)

    flush_buffer()
    return blocks


# --------------------------------------------------------------------------- #
# Fallback: pypdf page-by-page text extraction
# --------------------------------------------------------------------------- #

def _try_pypdf(raw: RawDocument) -> list[ParsedBlock]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        logger.error("pypdf not installed and pymupdf4llm failed: %s", exc)
        return []

    reader = PdfReader(io.BytesIO(raw.content_bytes))
    blocks: list[ParsedBlock] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        blocks.append(
            ParsedBlock(
                kind="paragraph",
                text=text,
                page=page_num,
                section_path=[f"Page {page_num}"],
            )
        )
    return blocks
