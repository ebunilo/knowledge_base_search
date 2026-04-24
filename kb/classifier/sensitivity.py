"""
Sensitivity classifier (rules-based v1).

Decides, for each chunk, whether it is allowed to leave the network for
generation (hosted_ok) or must stay on the self-hosted lane (self_hosted_only).

Signals in priority order:
    1. Source-level default from source_inventory.json (the baseline).
    2. Explicit per-document override in raw.metadata["sensitivity_override"].
    3. Content patterns: PII regexes, classification tags, explicit keywords
       like "CONFIDENTIAL", "INTERNAL ONLY", NDA references, secret-looking
       tokens (AWS keys, private keys, bearer tokens).
    4. Deny-list of source_ids that are always self_hosted_only
       (src_legal_contracts, src_internal_docs, src_support_tickets, …).

Upgrade path (Phase 2):
    * Swap the regex layer for Microsoft Presidio (PII NER + analyzers).
    * Add a small classifier model trained on flagged historical tickets.
    * Add per-chunk NLI check against a "sensitive-if" hypothesis set.

The classifier is deliberately conservative: when in doubt it returns
SELF_HOSTED_ONLY. Downgrading later is cheaper than a leak.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from kb.types import SensitivityLane


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Sources that must never be hosted-ok regardless of chunk contents.
# Mirrors the `sensitivity_lane` field in data/source_inventory.json.
# --------------------------------------------------------------------------- #
_ALWAYS_SELF_HOSTED_SOURCES: frozenset[str] = frozenset({
    "src_internal_docs",
    "src_github_private",
    "src_support_tickets",
    "src_legal_contracts",
    "src_product_specs",
})


# --------------------------------------------------------------------------- #
# Pattern library.
# Each entry: (tag, compiled regex, description)
# Keep patterns narrow to avoid false positives that would block harmless text.
# --------------------------------------------------------------------------- #

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("pii_email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("pii_phone_e164", re.compile(r"\+?\d{1,3}[-\s]?\(?\d{1,4}\)?[-\s]?\d{3,4}[-\s]?\d{3,4}\b")),
    ("pii_ssn_us", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("pii_iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    ("pii_credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("secret_aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("secret_bearer", re.compile(r"(?i)\b(?:bearer|token)\s+[A-Za-z0-9._\-]{20,}")),
    ("secret_private_key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),
    ("label_confidential", re.compile(r"\b(CONFIDENTIAL|STRICTLY\s+CONFIDENTIAL|INTERNAL\s+ONLY|DO\s+NOT\s+DISTRIBUTE|NDA|PRIVILEGED\s+AND\s+CONFIDENTIAL)\b", re.IGNORECASE)),
    ("label_gdpr", re.compile(r"\b(GDPR|PHI|PII|HIPAA|SOC\s?2|PCI\s?DSS)\b", re.IGNORECASE)),
]

# Any of these fired → classification is forced to self_hosted_only
_FORCE_SELF_HOSTED_TAGS: frozenset[str] = frozenset({
    "secret_aws_key", "secret_bearer", "secret_private_key",
    "label_confidential", "pii_ssn_us", "pii_credit_card",
    "pii_iban", "pii_private_key",
})


@dataclass
class SensitivityDecision:
    lane: SensitivityLane
    tags: list[str] = field(default_factory=list)
    reason: str = ""


class SensitivityClassifier:
    """Stateless classifier; shared across threads safely."""

    def classify(
        self,
        *,
        text: str,
        source_id: str,
        source_default: SensitivityLane,
        document_override: SensitivityLane | None = None,
    ) -> SensitivityDecision:
        # 1. Explicit per-document override wins.
        if document_override is not None:
            return SensitivityDecision(
                lane=document_override,
                reason="document_override",
            )

        # 2. Always-self-hosted sources — short-circuit before scanning content.
        if source_id in _ALWAYS_SELF_HOSTED_SOURCES:
            return SensitivityDecision(
                lane=SensitivityLane.SELF_HOSTED_ONLY,
                reason=f"source {source_id} is in always-self-hosted list",
            )

        # 3. Pattern scan.
        tags = self._scan(text)
        if tags & _FORCE_SELF_HOSTED_TAGS:
            return SensitivityDecision(
                lane=SensitivityLane.SELF_HOSTED_ONLY,
                tags=sorted(tags),
                reason=f"patterns: {sorted(tags & _FORCE_SELF_HOSTED_TAGS)}",
            )

        # 4. Fall back to the source default.
        return SensitivityDecision(
            lane=source_default,
            tags=sorted(tags),
            reason="source_default",
        )

    @staticmethod
    def _scan(text: str) -> set[str]:
        if not text:
            return set()
        hits: set[str] = set()
        for tag, pattern in _PATTERNS:
            if pattern.search(text):
                hits.add(tag)
        return hits
