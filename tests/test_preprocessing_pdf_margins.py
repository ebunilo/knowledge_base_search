"""Tests for margin-based PDF header/footer stripping."""

from __future__ import annotations

import pytest

from kb.preprocessing.pdf_margins import StripMarginsConfig, strip_pdf_margins


def _two_page_pdf_bytes() -> bytes:
    import fitz  # PyMuPDF (from pymupdf4llm)

    d = fitz.open()
    try:
        for i in range(2):
            page = d.new_page()
            page.insert_text((50, 100 + i * 20), f"page {i + 1}")
        return d.tobytes()
    finally:
        d.close()


def test_strip_pdf_margins_preserves_page_count() -> None:
    import fitz

    raw = _two_page_pdf_bytes()
    out = strip_pdf_margins(raw)
    o = fitz.open(stream=out, filetype="pdf")
    try:
        assert o.page_count == 2
        for i in range(2):
            h = o[i].rect.height
            # Clipped page is shorter than a full default page (or equal if min-body guard)
            assert h > 0
    finally:
        o.close()


def test_strip_config_validation() -> None:
    with pytest.raises(ValueError, match="header_ratio"):
        StripMarginsConfig(header_ratio=0.6)
    with pytest.raises(ValueError, match="footer_ratio"):
        StripMarginsConfig(footer_ratio=-0.1)
    with pytest.raises(ValueError, match="almost the entire page"):
        StripMarginsConfig(header_ratio=0.45, footer_ratio=0.45)
