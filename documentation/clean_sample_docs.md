# Cleaning PDFs before `sample_docs`

Some manuals (for example data sheets) repeat page headers and footers. For RAG, it helps to **drop fixed top and bottom bands** on each page *before* you place files under `data/sample_docs/…`.

## What the tool does

- **Input:** one or more `.pdf` paths, or a directory (every `*.pdf` inside, non-recursive).
- **Method:** for each page, a **center rectangle** is taken (by height fractions), then the page is **re-drawn** into a new file. This is **geometric** only: it does not look for text or repeated lines. Tune `--header` / `--footer` for your print margins.
- **Output:** by default, `name.cleaned.pdf` next to each input, or a target path / directory (see the script’s `--help`).

## Usage

From the repository root, with dependencies installed (PyMuPDF comes in via `pymupdf4llm`):

```bash
python scripts/clean_pdf_margins.py path/to/doc.pdf
python scripts/clean_pdf_margins.py path/to/folder/ -o data/sample_docs/public -f
```

Optional flags: `--header`, `--footer` (each must be below 0.5, as fractions of page height), and `--min-body-pt` (safety: if the clipped body would be too small, the full page is kept for that page).

## Programmatic use

`strip_pdf_margins(pdf_bytes, StripMarginsConfig(...))` in `kb.preprocessing.pdf_margins` returns cleaned PDF **bytes** (same for Markdown or other formats is not in scope; handle those separately if needed).

## Limits

- Multi-column or full-bleed figures can be clipped; reduce ratios or pre-split pages.
- It is not a full layout or OCR pipeline; it is a **pre-dump** step for the corpus in `sample_docs`.
