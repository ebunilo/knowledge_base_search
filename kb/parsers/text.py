"""
Text-like parsers: Markdown, HTML, plain text.

Markdown gets the same ATX-heading walker used by the PDF parser so the
chunker's section_path logic is uniform. HTML is converted to Markdown-ish
text with a small regex-based tag stripper — good enough for Phase 1 and
avoids a hard BeautifulSoup dependency.
"""

from __future__ import annotations

import re

from kb.types import ParsedBlock, RawDocument


_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_HEADING_RE = re.compile(
    r"<h([1-6])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL
)
_WS_RE = re.compile(r"[ \t]+")


def parse_markdown(raw: RawDocument) -> list[ParsedBlock]:
    text = _decode(raw.content_bytes)
    return _markdown_to_blocks(text)


def parse_text(raw: RawDocument) -> list[ParsedBlock]:
    text = _decode(raw.content_bytes).strip()
    if not text:
        return []
    # Split on double-newline paragraphs — preserves natural boundaries for
    # the chunker without imposing structure that isn't there.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [
        ParsedBlock(kind="paragraph", text=p, section_path=[])
        for p in paragraphs
    ]


def parse_html(raw: RawDocument) -> list[ParsedBlock]:
    """
    Lightweight HTML → Markdown-ish conversion. Good enough for Confluence
    exports, knowledge base articles, and simple static pages. For anything
    complex, users should rely on Docling/unstructured (Phase 2).
    """
    html = _decode(raw.content_bytes)
    # Replace headings with ATX equivalents so the MD walker picks them up.
    def _repl_heading(m: re.Match) -> str:
        level = int(m.group(1))
        inner = _HTML_TAG_RE.sub("", m.group(2)).strip()
        return f"\n\n{'#' * level} {inner}\n\n"

    md = _HTML_HEADING_RE.sub(_repl_heading, html)
    # Paragraphs + line breaks
    md = re.sub(r"</p>", "\n\n", md, flags=re.IGNORECASE)
    md = re.sub(r"<br\s*/?>", "\n", md, flags=re.IGNORECASE)
    # Strip remaining tags
    md = _HTML_TAG_RE.sub("", md)
    # Collapse whitespace
    md = _WS_RE.sub(" ", md)
    return _markdown_to_blocks(md)


# --------------------------------------------------------------------------- #
# Shared: Markdown → blocks with heading stack
# --------------------------------------------------------------------------- #

def _markdown_to_blocks(md: str) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    in_code = False
    code_lang: str | None = None

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
        stripped = line.rstrip()

        # Fenced code blocks — preserve as single blocks
        if stripped.startswith("```"):
            if not in_code:
                flush_buffer()
                in_code = True
                code_lang = stripped[3:].strip() or None
                code_buf: list[str] = []
            else:
                in_code = False
                blocks.append(
                    ParsedBlock(
                        kind="code",
                        text="\n".join(code_buf),
                        section_path=[h[1] for h in heading_stack],
                        meta={"language": code_lang} if code_lang else {},
                    )
                )
            continue

        if in_code:
            code_buf.append(line)
            continue

        m = _ATX_HEADING_RE.match(stripped)
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
        elif not stripped:
            flush_buffer()
        else:
            buffer.append(line)

    flush_buffer()
    return blocks


def _decode(content_bytes: bytes) -> str:
    try:
        return content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return content_bytes.decode("utf-8", errors="replace")
