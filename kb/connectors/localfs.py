"""
Local-filesystem connector.

Walks a directory tree and yields a RawDocument for every file whose extension
matches the configured include-list. This is the workhorse for demos and
unit tests: drop files in data/sample_docs/ and point the connector at it.

connector_config shape (from source_inventory.json):
    {
        "base_path": "./data/sample_docs",
        "include_extensions": [".pdf", ".md", ".yml", ".yaml", ".json"],
        "exclude_globs": ["**/.git/**", "**/node_modules/**"]
    }
"""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Iterator
from pathlib import Path

from kb.connectors.base import Connector, ConnectorError
from kb.types import DocumentFormat, RawDocument


logger = logging.getLogger(__name__)


_EXT_TO_FORMAT: dict[str, DocumentFormat] = {
    ".pdf":  DocumentFormat.PDF,
    ".md":   DocumentFormat.MARKDOWN,
    ".markdown": DocumentFormat.MARKDOWN,
    ".html": DocumentFormat.HTML,
    ".htm":  DocumentFormat.HTML,
    ".yml":  DocumentFormat.YAML,
    ".yaml": DocumentFormat.YAML,
    ".json": DocumentFormat.JSON,
    ".txt":  DocumentFormat.TEXT,
}

_DEFAULT_INCLUDE = list(_EXT_TO_FORMAT.keys())
_DEFAULT_EXCLUDE = ["**/.git/**", "**/.venv/**", "**/node_modules/**", "**/__pycache__/**"]


class LocalFilesystemConnector(Connector):
    """Walk a directory tree and yield RawDocument for each matching file."""

    def __init__(self, source_id, source_config, connector_config):
        super().__init__(source_id, source_config, connector_config)

        base_path_str = self.connector_config.get("base_path")
        if not base_path_str:
            raise ConnectorError(
                f"localfs source {source_id!r} missing connector_config.base_path"
            )

        self.base_path = Path(base_path_str).expanduser().resolve()
        self.include_extensions: list[str] = [
            e.lower() for e in self.connector_config.get("include_extensions", _DEFAULT_INCLUDE)
        ]
        self.exclude_globs: list[str] = self.connector_config.get(
            "exclude_globs", _DEFAULT_EXCLUDE
        )
        self.max_bytes: int = self.connector_config.get("max_bytes_per_doc", 25_000_000)

    def iter_documents(self) -> Iterator[RawDocument]:
        if not self.base_path.exists():
            logger.warning(
                "localfs base_path does not exist: %s (source=%s)",
                self.base_path, self.source_id,
            )
            return

        if not self.base_path.is_dir():
            raise ConnectorError(f"localfs base_path is not a directory: {self.base_path}")

        logger.info("Scanning %s for source %s", self.base_path, self.source_id)

        for path in sorted(self.base_path.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.include_extensions:
                continue
            if self._is_excluded(path):
                continue

            try:
                stat = path.stat()
                if stat.st_size > self.max_bytes:
                    logger.warning(
                        "Skipping %s (size %d exceeds max_bytes=%d)",
                        path, stat.st_size, self.max_bytes,
                    )
                    continue

                content_bytes = path.read_bytes()
            except OSError as exc:
                logger.error("Failed to read %s: %s", path, exc)
                continue

            rel = path.relative_to(self.base_path).as_posix()
            fmt = _EXT_TO_FORMAT.get(path.suffix.lower(), DocumentFormat.UNKNOWN)

            yield self._build_raw_document(
                source_uri=f"file://{path}",
                content_bytes=content_bytes,
                title=path.stem,
                format_hint=fmt,
                metadata={
                    "relative_path": rel,
                    "size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                },
            )

    def _is_excluded(self, path: Path) -> bool:
        p = path.as_posix()
        return any(fnmatch.fnmatch(p, pat) for pat in self.exclude_globs)
