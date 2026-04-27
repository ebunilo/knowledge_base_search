"""
Remove top (header) and bottom (footer) bands from a PDF by clipping each
page to a center rectangle, then re-rendering those clips into a new file.

Suitable for **repetitive** running headers/footers in technical manuals. It
does **not** use OCR or read text: it is purely geometric. Multi-column
layouts, figures that span the full page, or very large margins may need
tuned ``header_ratio`` / ``footer_ratio`` (or a different tool).

Requires PyMuPDF (``import fitz``), which is pulled in with ``pymupdf4llm``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StripMarginsConfig:
    """Vertical fractions of each page to drop from the top and bottom."""

    # Typical technical PDFs: 6–10% per band (tune for your print margins).
    header_ratio: float = 0.08
    footer_ratio: float = 0.10

    # If clipping would leave less than this height in points, keep full page.
    min_body_height_points: float = 36.0  # about 0.5 inch

    def __post_init__(self) -> None:
        if not 0.0 <= self.header_ratio < 0.5:
            raise ValueError("header_ratio must be in [0, 0.5)")
        if not 0.0 <= self.footer_ratio < 0.5:
            raise ValueError("footer_ratio must be in [0, 0.5)")
        if self.header_ratio + self.footer_ratio >= 0.9:
            raise ValueError("header_ratio + footer_ratio would remove almost the entire page")


def strip_pdf_margins(
    pdf_bytes: bytes,
    config: StripMarginsConfig | None = None,
) -> bytes:
    """
    Return a new PDF as bytes with header/footer bands cut from every page.
    """
    import fitz  # PyMuPDF

    cfg = config or StripMarginsConfig()
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        dst: Any = None
        try:
            dst = fitz.open()
            for pno in range(src.page_count):
                page = src[pno]
                r = page.rect
                h = r.height
                y0 = r.y0 + h * cfg.header_ratio
                y1 = r.y1 - h * cfg.footer_ratio
                clip: Any
                if y1 - y0 < cfg.min_body_height_points:
                    logger.warning(
                        "page %d: body too short after clip (%.1fpt); using full page",
                        pno + 1,
                        y1 - y0,
                    )
                    clip = r
                else:
                    clip = fitz.Rect(r.x0, y0, r.x1, y1)

                npage = dst.new_page(width=clip.width, height=clip.height)
                npage.show_pdf_page(
                    npage.rect,
                    src,
                    pno,
                    clip=clip,
                    keep_proportion=True,
                    overlay=True,
                )
            return dst.tobytes()
        finally:
            if dst is not None:
                dst.close()
    finally:
        src.close()
