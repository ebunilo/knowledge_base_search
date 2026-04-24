"""
Access-control decisions and filter builders.

Rules (applied on every hit before it leaves the retriever):

    A document is visible to the user if ANY of the following holds:

        (a) document.visibility == "public"
        (b) document.source_id is in user.granted_source_ids
        (c) user.tenant_id matches the tenant of the private collection
            AND (user.department in document.acl.departments OR "*" in …)
            AND user's role_rank >= document.acl.min_role_rank

Rule (a) covers customer-facing docs. Rule (b) is the escape hatch for
extra grants (e.g. an SRE with read access to `src_github_private`).
Rule (c) is the standard RBAC path for internal users.

Role hierarchy (from staff_directory.json):
    anonymous < employee < manager < admin

The module exposes both:
    * `build_qdrant_filter(user, collection)` — pushdown filter for Qdrant
      so the vector DB never returns points the user cannot see.
    * `hit_allowed(user, payload)` — Python predicate used for the BM25
      path (in-process index has no pushdown).

Both paths use the same rules; keeping them next to each other makes
divergence obvious if the rules ever change.
"""

from __future__ import annotations

from typing import Any

from kb.retrieval.types import UserContext
from kb.settings import Settings, get_settings


# Canonical role ordering. Unknown roles collapse to anonymous.
_ROLE_RANK: dict[str, int] = {
    "anonymous": 0,
    "customer":  0,          # alias — external users
    "employee":  1,
    "manager":   2,
    "admin":     3,
}


# --------------------------------------------------------------------------- #
# Collection routing
# --------------------------------------------------------------------------- #

def accessible_collections(
    user: UserContext,
    settings: Settings | None = None,
) -> list[str]:
    """
    Return the Qdrant collections this user may search.

    External / anonymous users see only the public collection. Staff
    (tenant matches the configured internal tenant) also see the private
    collection; ACL payload filtering further narrows results.
    """
    settings = settings or get_settings()
    collections = [settings.app_public_collection]
    if (
        user.tenant_id
        and user.tenant_id == settings.app_tenant_internal_id
        and _rank(user.role) >= _ROLE_RANK["employee"]
    ):
        collections.append(settings.app_private_collection)
    return collections


def is_private_collection(collection: str, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return collection == settings.app_private_collection


# --------------------------------------------------------------------------- #
# Qdrant filter (pushdown)
# --------------------------------------------------------------------------- #

def build_qdrant_filter(
    user: UserContext,
    collection: str,
    *,
    source_allowlist: list[str] | None = None,
    allowed_match_kinds: list[str] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any] | None:
    """
    Build a Qdrant `filter` object enforcing ACL + optional restrictions.

    For the public collection (everything is visibility=public) we return
    either None or only the source/kind restrictions — no ACL needed.

    For the private collection we AND together:
        * ACL clause   — public OR department-match OR extra-grant source
        * source clause — if allowlist provided
        * kind clause   — content / question point filter
    """
    settings = settings or get_settings()
    must: list[dict[str, Any]] = []

    if is_private_collection(collection, settings):
        must.append(_private_acl_clause(user))
    # Public collection: no ACL clause; every point is public by construction.

    if source_allowlist:
        must.append({"key": "source_id", "match": {"any": list(source_allowlist)}})

    if allowed_match_kinds and len(allowed_match_kinds) < 2:
        must.append({"key": "kind", "match": {"any": list(allowed_match_kinds)}})

    if not must:
        return None
    return {"must": must}


def _private_acl_clause(user: UserContext) -> dict[str, Any]:
    """
    Build the core ACL clause for a private collection:

        (visibility = "public")
         OR (department match AND role match)
         OR (source_id in extra_grants)
    """
    satisfied_roles = [
        r for r, rank in _ROLE_RANK.items() if rank <= _rank(user.role)
    ]

    should: list[dict[str, Any]] = [
        {"key": "visibility", "match": {"value": "public"}},
        {
            "must": [
                {
                    "key": "acl_departments",
                    "match": {
                        "any": list(_dept_match_tokens(user)),
                    },
                },
                {
                    "key": "acl_min_role",
                    "match": {"any": satisfied_roles},
                },
            ]
        },
    ]

    grants = user.granted_source_ids
    if grants:
        should.append({"key": "source_id", "match": {"any": grants}})

    return {"should": should}


def _dept_match_tokens(user: UserContext) -> list[str]:
    """
    Department tokens the user matches.

    "*" is always included so docs with a wildcard ACL are visible to any
    internal user. The user's own department is added when present.
    """
    out = ["*"]
    if user.department:
        out.append(user.department)
    return out


# --------------------------------------------------------------------------- #
# Python-side predicate (BM25 path, audit, defence-in-depth)
# --------------------------------------------------------------------------- #

def hit_allowed(user: UserContext, payload: dict[str, Any]) -> bool:
    """
    Return True iff the user may see the hit described by `payload`.

    `payload` must carry the shape written by QdrantWriter / BM25Writer:
        visibility, sensitivity, source_id,
        acl_departments, acl_min_role, acl_tags
    (BM25 stores the same info under payload["acl"] and top-level fields;
    we read both shapes.)
    """
    # --- Rule (a): public ---
    visibility = payload.get("visibility")
    if visibility == "public":
        return True

    source_id = payload.get("source_id")
    # --- Rule (b): extra grants ---
    if source_id and source_id in user.granted_source_ids:
        return True

    # --- Rule (c): RBAC ---
    if user.is_external:
        return False

    acl_departments = payload.get("acl_departments")
    acl_min_role = payload.get("acl_min_role")
    if acl_departments is None or acl_min_role is None:
        acl = payload.get("acl") or {}
        acl_departments = acl.get("departments", [])
        acl_min_role = acl.get("min_role", "anonymous")

    # Role check
    if _rank(user.role) < _rank(acl_min_role):
        return False

    # Department check
    if "*" in (acl_departments or []):
        return True
    if user.department and user.department in acl_departments:
        return True
    return False


def _rank(role: str | None) -> int:
    if not role:
        return 0
    return _ROLE_RANK.get(role.lower(), 0)


# --------------------------------------------------------------------------- #
# Convenience — load a user from staff_directory.json
# --------------------------------------------------------------------------- #

def load_user(
    user_ref: str,
    *,
    directory_path: str = "data/staff_directory.json",
) -> UserContext:
    """
    Build a UserContext from an id / email in `staff_directory.json`.

    Falls back to an anonymous context if the ref is "anonymous" or empty.
    """
    import json

    if not user_ref or user_ref == "anonymous":
        return UserContext()

    with open(directory_path, "r", encoding="utf-8") as f:
        dir_ = json.load(f)

    tenant_id = dir_.get("tenant_model", {}).get("internal_tenant_id")
    for u in dir_.get("users", []):
        if u.get("user_id") == user_ref or u.get("email") == user_ref:
            return UserContext(
                user_id=u["user_id"],
                tenant_id=tenant_id,
                department=u.get("department"),
                role=u.get("role", "employee"),
                extra_grants=u.get("extra_grants", []) or [],
                region=u.get("region"),
            )
    raise KeyError(f"user {user_ref!r} not found in {directory_path}")
