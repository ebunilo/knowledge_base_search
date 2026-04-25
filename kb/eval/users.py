"""
Map `golden_set.json` examples to a concrete `UserContext`.

Priority:
    1. `eval_user` field on the example
    2. Hard override table for one-off test semantics (e.g. ACL-denial rows)
    3. `expected_acl` {min_role, departments?} heuristics + staff_directory.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kb.retrieval.acl import load_user
from kb.retrieval.types import UserContext


# Role that must *fail* the ACL gate for a row (f005) — the golden case
# checks that a generic employee in engineering is denied a legal-only doc.
_QID_DENIAL_EMPLOYEE: frozenset[str] = frozenset({"f005"})

# QIDs where we need a *non-legal* staffer for the negative ACL scenario.
_QID_ACL_EMP_NON_LEGAL: frozenset[str] = frozenset({"neg002"})

_ROLE_ORDER = ("anonymous", "customer", "employee", "manager", "admin")


def _rank(role: str) -> int:
    r = (role or "anonymous").lower()
    try:
        return _ROLE_ORDER.index(r) if r in _ROLE_ORDER else 0
    except ValueError:
        return 0


def _users_from_directory(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("users", [])


def _pick_by_acl(
    expected_acl: dict[str, Any],
    users: list[dict[str, Any]],
) -> str | None:
    """Return user_id of the first user satisfying expected_acl, or None."""
    min_r = (expected_acl or {}).get("min_role", "employee")
    want_rank = _rank(min_r)
    depts = (expected_acl or {}).get("departments") or []
    all_dept = depts in ([], ["*"], ["any"])

    for u in users:
        if _rank(u.get("role", "anonymous")) < want_rank:
            continue
        dep = (u.get("department") or "").lower()
        if not all_dept and depts:
            dset = {d.lower() for d in depts if isinstance(d, str)}
            if dep and dep not in dset:
                continue
        return u.get("user_id")
    return None


def user_for_qid(
    ex,
    *,
    staff_directory: Path = Path("data/staff_directory.json"),
) -> UserContext:
    """Build UserContext for one golden example."""
    if ex.eval_user:
        return load_user(ex.eval_user, directory_path=str(staff_directory))
    if ex.qid in _QID_ACL_EMP_NON_LEGAL:
        return load_user("u_002", directory_path=str(staff_directory))
    if ex.qid in _QID_DENIAL_EMPLOYEE:
        return load_user("u_002", directory_path=str(staff_directory))

    users = _users_from_directory(staff_directory)
    if not (ex.expected_acl or {}).get("min_role") or (ex.expected_acl or {}).get(
        "min_role"
    ) == "anonymous":
        return UserContext()

    uid = _pick_by_acl(ex.expected_acl, users)
    if uid is None and (ex.expected_acl or {}).get("min_role", "") != "anonymous":
        # Fallback: first internal user with at least the minimum role
        for u in users:
            if _rank(u.get("role", "")) >= _rank(
                (ex.expected_acl or {}).get("min_role", "employee"),
            ):
                uid = u.get("user_id")
                break
    if uid is None:
        return UserContext()
    return load_user(uid, directory_path=str(staff_directory))
