"""
YAML and JSON parsers.

Structured documents (OpenAPI specs, CI/CD configs, runbooks-as-YAML,
product catalogs) are common in our corpus. Treating them as plain text
destroys the most useful retrieval signal — the key path. Instead, we walk
the parsed tree and emit a `yaml_leaf` / `json_leaf` block for each scalar
value, with its key path as section_path. Lists and maps produce heading
blocks so the chunker can group related leaves under the same parent.

Example:
    {"services": {"auth": {"timeout": 30}}}
produces:
    heading("services", level=1)
    heading("auth",     level=2, section_path=["services"])
    json_leaf("timeout = 30", section_path=["services","auth"])
"""

from __future__ import annotations

import json
import logging
from typing import Any

from kb.types import ParsedBlock, RawDocument


logger = logging.getLogger(__name__)


def parse_yaml(raw: RawDocument) -> list[ParsedBlock]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for YAML parsing") from exc

    try:
        text = raw.content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.content_bytes.decode("utf-8", errors="replace")

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        logger.warning("YAML parse failed for %s: %s — treating as plain text", raw.source_uri, exc)
        return _fallback_plain(text)

    if data is None:
        return _fallback_plain(text)

    return list(_walk(data, path=[], leaf_kind="yaml_leaf"))


def parse_json(raw: RawDocument) -> list[ParsedBlock]:
    try:
        text = raw.content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.content_bytes.decode("utf-8", errors="replace")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed for %s: %s — treating as plain text", raw.source_uri, exc)
        return _fallback_plain(text)

    return list(_walk(data, path=[], leaf_kind="json_leaf"))


# --------------------------------------------------------------------------- #
# Shared tree walker
# --------------------------------------------------------------------------- #

# Scalars grouped under the same parent that together fit under this threshold
# are coalesced into a single leaf block. Keeps chunk counts manageable for
# dense key-value files (e.g. OpenAPI).
_COALESCE_CHAR_BUDGET = 400


def _walk(node: Any, path: list[str], leaf_kind: str):
    if isinstance(node, dict):
        # Emit a heading for this level (root dict → no heading, just its keys)
        if path:
            yield ParsedBlock(
                kind="heading",
                text=path[-1],
                section_path=list(path),
                level=len(path),
            )
        yield from _walk_mapping(node, path, leaf_kind)

    elif isinstance(node, list):
        if path:
            yield ParsedBlock(
                kind="heading",
                text=path[-1],
                section_path=list(path),
                level=len(path),
            )
        for idx, item in enumerate(node):
            child_path = [*path, f"[{idx}]"]
            if isinstance(item, (dict, list)):
                yield from _walk(item, child_path, leaf_kind)
            else:
                yield ParsedBlock(
                    kind=leaf_kind,  # type: ignore[arg-type]
                    text=f"{path[-1] if path else '<root>'}[{idx}] = {_fmt_scalar(item)}",
                    section_path=child_path,
                    level=len(child_path),
                )
    else:
        yield ParsedBlock(
            kind=leaf_kind,  # type: ignore[arg-type]
            text=f"{path[-1] if path else '<root>'} = {_fmt_scalar(node)}",
            section_path=list(path) or ["<root>"],
            level=len(path) or 1,
        )


def _walk_mapping(mapping: dict, path: list[str], leaf_kind: str):
    """
    For each key:
      - if value is a container, recurse
      - if value is scalar, collect into a coalesce buffer
    Flush the buffer whenever it exceeds the char budget or before recursing
    into a container.
    """
    buffer: list[str] = []
    buffer_chars = 0

    def flush():
        nonlocal buffer, buffer_chars
        if not buffer:
            return None
        block = ParsedBlock(
            kind=leaf_kind,  # type: ignore[arg-type]
            text="\n".join(buffer),
            section_path=list(path) or ["<root>"],
            level=len(path) or 1,
        )
        buffer = []
        buffer_chars = 0
        return block

    for key, value in mapping.items():
        key_str = str(key)
        if isinstance(value, (dict, list)):
            flushed = flush()
            if flushed is not None:
                yield flushed
            yield from _walk(value, [*path, key_str], leaf_kind)
        else:
            line = f"{key_str} = {_fmt_scalar(value)}"
            if buffer and buffer_chars + len(line) + 1 > _COALESCE_CHAR_BUDGET:
                flushed = flush()
                if flushed is not None:
                    yield flushed
            buffer.append(line)
            buffer_chars += len(line) + 1

    flushed = flush()
    if flushed is not None:
        yield flushed


def _fmt_scalar(v: Any) -> str:
    if isinstance(v, str):
        return v if len(v) <= 200 else v[:197] + "..."
    if v is None:
        return "null"
    return str(v)


def _fallback_plain(text: str) -> list[ParsedBlock]:
    text = text.strip()
    if not text:
        return []
    return [ParsedBlock(kind="raw", text=text, section_path=[])]
