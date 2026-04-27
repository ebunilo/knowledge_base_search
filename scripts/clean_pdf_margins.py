#!/usr/bin/env python3
"""
Strip top/bottom bands from one or more PDFs (geometric margin clip).

Typical use: run on files before copying them into ``data/sample_docs/...`` so
RAG chunking is less contaminated by running headers/footers.

  uv run python scripts/clean_pdf_margins.py path/to/in.pdf
  uv run python scripts/clean_pdf_margins.py data/incoming/ -o data/sample_docs/public
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

# Allow ``python scripts/clean_pdf_margins.py`` from the repo without editable install.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from kb.preprocessing.pdf_margins import StripMarginsConfig, strip_pdf_margins  # noqa: E402


def _default_out(pdf: Path) -> Path:
    return pdf.with_name(f"{pdf.stem}.cleaned{pdf.suffix}")


def _collect_inputs(paths: tuple[Path, ...]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob("*.pdf")))
        else:
            if p.suffix.lower() != ".pdf":
                raise click.ClickException(f"not a PDF: {p}")
            out.append(p)
    return out


def _resolve_out_path(
    in_pdf: Path,
    output: Path | None,
    *,
    multiple_inputs: bool,
) -> Path:
    if output is None:
        return _default_out(in_pdf)
    if not multiple_inputs and output.suffix.lower() == ".pdf":
        return output
    output.mkdir(parents=True, exist_ok=True)
    return output / _default_out(in_pdf).name


@click.command()
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path, file_okay=True, dir_okay=True),
    required=True,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path, allow_dash=True),
    default=None,
    help="Output .pdf (one file in) or a directory. Default: *.cleaned.pdf beside the input.",
)
@click.option(
    "--header",
    "header_ratio",
    type=float,
    default=0.08,
    show_default=True,
    help="Fraction of page height to remove from the top.",
)
@click.option(
    "--footer",
    "footer_ratio",
    type=float,
    default=0.10,
    show_default=True,
    help="Fraction of page height to remove from the bottom.",
)
@click.option(
    "--min-body-pt",
    "min_body_height_points",
    type=float,
    default=36.0,
    show_default=True,
    help="If the clipped body is shorter (points), keep the full page.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Overwrite an existing output file.",
)
def main(
    paths: tuple[Path, ...],
    output: str | None,
    header_ratio: float,
    footer_ratio: float,
    min_body_height_points: float,
    force: bool,
) -> None:
    """Strip header/footer *bands* from PDFs (geometric, not text detection)."""
    if output in (None, "-"):
        out_path: Path | None = None
    else:
        out_path = Path(output)
    del output

    inputs = _collect_inputs(paths)
    if not inputs:
        raise click.ClickException("no PDFs matched")
    if len(inputs) > 1 and out_path is not None and out_path.suffix.lower() == ".pdf":
        raise click.ClickException("with multiple inputs, -o must be a directory, not a .pdf path")

    config = StripMarginsConfig(
        header_ratio=header_ratio,
        footer_ratio=footer_ratio,
        min_body_height_points=min_body_height_points,
    )

    n_in = len(inputs)
    multiple = n_in > 1
    for pdf_path in inputs:
        target = _resolve_out_path(pdf_path, out_path, multiple_inputs=multiple)
        if target.suffix.lower() != ".pdf":
            raise click.ClickException(f"output must be a .pdf path: {target}")
        if target.exists() and not force:
            raise click.ClickException(f"refusing to overwrite (use -f): {target}")
        raw = pdf_path.read_bytes()
        cleaned = strip_pdf_margins(raw, config)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(cleaned)
        click.echo(f"wrote {target} ({len(cleaned)} bytes)")


if __name__ == "__main__":
    main()
