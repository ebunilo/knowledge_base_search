"""
Load `data/staff_directory.json` for the user picker. Falls back gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kb.retrieval.acl import load_user
from kb.retrieval.types import UserContext


def _path() -> Path:
    p = Path("data/staff_directory.json")
    return p


def load_user_options() -> list[dict[str, str]]:
    """`{id, label, department, role}` for UI dropdowns."""
    path = _path()
    if not path.exists():
        return [
            {
                "id": "anonymous",
                "label": "Anonymous (public only)",
                "department": "",
                "role": "anonymous",
            },
        ]
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    out: list[dict[str, str]] = [
        {
            "id": "anonymous",
            "label": "Anonymous (public only)",
            "department": "",
            "role": "anonymous",
        },
    ]
    for u in data.get("users", []):
        uid = u.get("user_id", "")
        if not uid:
            continue
        label = (u.get("display_name") or uid) + (
            f" — {u.get('email', '')}" if u.get("email") else ""
        )
        out.append(
            {
                "id": uid,
                "label": label,
                "department": (u.get("department") or ""),
                "role": (u.get("role") or "employee"),
            },
        )
    return out


def build_user(p) -> UserContext:
    from kb.web.models import UserPayload

    p = p if isinstance(p, UserPayload) else UserPayload.model_validate(p)
    if (p.user_id or "anonymous") in ("", "anonymous"):
        return UserContext()
    path = _path()
    dir_path = str(path) if path.exists() else "data/staff_directory.json"
    try:
        u = load_user(p.user_id, directory_path=dir_path)
    except (FileNotFoundError, KeyError):
        u = UserContext(user_id=p.user_id)
    if p.role is not None:
        u = u.model_copy(update={"role": p.role})
    if p.department is not None:
        u = u.model_copy(update={"department": p.department})
    return u
