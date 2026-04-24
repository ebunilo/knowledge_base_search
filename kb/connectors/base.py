"""
Connector framework.

A Connector is an iterable of RawDocument. It is responsible for:
  * discovering documents in the upstream system
  * fetching raw bytes
  * computing a stable content hash (for incremental indexing)
  * stamping defaults from source_inventory.json (sensitivity, visibility, acl)

Connectors are deliberately thin — they do NOT parse, chunk, or classify.
That keeps them cheap to rerun and easy to test.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from kb.types import AclPayload, RawDocument, SensitivityLane, Visibility


logger = logging.getLogger(__name__)


class ConnectorError(Exception):
    """Raised when a connector cannot be constructed or a fetch fails hard."""


class Connector(ABC):
    """Abstract base class for all connectors."""

    def __init__(
        self,
        source_id: str,
        source_config: dict[str, Any],
        connector_config: dict[str, Any] | None = None,
    ) -> None:
        self.source_id = source_id
        self.source_config = source_config
        self.connector_config = connector_config or {}

        # Defaults from source_inventory.json — propagated onto every RawDocument
        self._default_sensitivity = SensitivityLane(
            source_config.get("sensitivity_lane", "self_hosted_only")
        )
        self._default_visibility = Visibility(
            source_config.get("default_visibility", "internal")
        )
        default_acl = source_config.get("default_acl") or {}
        self._default_acl = AclPayload(
            departments=default_acl.get("departments", ["*"]),
            min_role=default_acl.get("min_role", "anonymous"),
            tags=default_acl.get("tags", []),
        )
        self._region = source_config.get("region")
        self._language = source_config.get("language", "en")

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #

    @abstractmethod
    def iter_documents(self) -> Iterator[RawDocument]:
        """Yield RawDocument objects lazily. Implementations should be generators."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Helpers shared by concrete connectors
    # ------------------------------------------------------------------ #

    def _build_raw_document(
        self,
        *,
        source_uri: str,
        content_bytes: bytes,
        title: str | None = None,
        format_hint: Any = None,
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
    ) -> RawDocument:
        """Assemble a RawDocument with defaults filled in from source config."""
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        if document_id is None:
            document_id = self._make_document_id(source_uri, content_hash)

        return RawDocument(
            document_id=document_id,
            source_id=self.source_id,
            source_uri=source_uri,
            title=title,
            content_bytes=content_bytes,
            content_hash=content_hash,
            format_hint=format_hint or "unknown",
            language=self._language,
            region=self._region,
            default_sensitivity=self._default_sensitivity,
            default_visibility=self._default_visibility,
            default_acl=self._default_acl,
            metadata=metadata or {},
        )

    def _make_document_id(self, source_uri: str, content_hash: str) -> str:
        """
        Stable id scheme: `<source_id>::<sha1(source_uri)[:16]>`.

        Keyed on the URI (not the content) so that when a document is updated
        its id stays constant and we overwrite rather than create duplicates.
        The content_hash is carried separately and used by RecordManager to
        detect unchanged documents.
        """
        uri_hash = hashlib.sha1(source_uri.encode("utf-8")).hexdigest()[:16]
        return f"{self.source_id}::{uri_hash}"
