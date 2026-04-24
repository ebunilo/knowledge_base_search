"""Source connectors."""

from kb.connectors.base import Connector, ConnectorError
from kb.connectors.localfs import LocalFilesystemConnector

__all__ = ["Connector", "ConnectorError", "LocalFilesystemConnector", "get_connector"]


def get_connector(source_id: str, source_config: dict) -> Connector:
    """
    Factory: build a Connector from a source_inventory.json entry.

    Supported connector_type values (Phase 1):
        * localfs       — walk a directory on disk (demo workhorse)

    More connectors (confluence, github, s3, jira) are stubbed in the roadmap
    and will be added in Phase 2.
    """
    connector_type = source_config["connector_type"]
    connector_config = source_config.get("connector_config", {})

    if connector_type == "localfs":
        return LocalFilesystemConnector(
            source_id=source_id,
            source_config=source_config,
            connector_config=connector_config,
        )

    raise ConnectorError(
        f"Unsupported connector_type {connector_type!r} for source {source_id!r}. "
        "Phase 1 supports: localfs."
    )
