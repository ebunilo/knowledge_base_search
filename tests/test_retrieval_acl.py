"""
Tests for `kb.retrieval.acl`.

These cover the *decision rules* that keep internal docs away from
external users, and the Qdrant filter shape we push down to the vector DB.
No network / DB is exercised.
"""

from __future__ import annotations

import pytest

from kb.retrieval.acl import (
    accessible_collections,
    build_qdrant_filter,
    hit_allowed,
)
from kb.retrieval.types import UserContext
from kb.settings import Settings


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_public_collection="public_v1",
        app_private_collection="tenant_acme_staff_private_v1",
        app_tenant_internal_id="tenant_acme_staff",
    )


@pytest.fixture
def anonymous() -> UserContext:
    return UserContext()


@pytest.fixture
def eng_employee() -> UserContext:
    return UserContext(
        user_id="u_002",
        tenant_id="tenant_acme_staff",
        department="engineering",
        role="employee",
    )


@pytest.fixture
def ops_manager_with_grant() -> UserContext:
    return UserContext(
        user_id="u_006",
        tenant_id="tenant_acme_staff",
        department="operations",
        role="manager",
        extra_grants=["src_github_private:read"],
    )


# --------------------------------------------------------------------------- #
# Collection routing
# --------------------------------------------------------------------------- #

class TestAccessibleCollections:
    def test_anonymous_gets_public_only(self, anonymous, settings):
        cols = accessible_collections(anonymous, settings)
        assert cols == ["public_v1"]

    def test_employee_gets_both(self, eng_employee, settings):
        cols = accessible_collections(eng_employee, settings)
        assert cols == ["public_v1", "tenant_acme_staff_private_v1"]

    def test_wrong_tenant_is_external(self, settings):
        # Valid role, but tenant is not ours — treat as external.
        u = UserContext(tenant_id="tenant_other", role="manager")
        cols = accessible_collections(u, settings)
        assert cols == ["public_v1"]

    def test_role_anonymous_with_tenant_still_blocks(self, settings):
        u = UserContext(tenant_id="tenant_acme_staff", role="anonymous")
        cols = accessible_collections(u, settings)
        assert cols == ["public_v1"]


# --------------------------------------------------------------------------- #
# Python-side predicate (BM25 path, defence in depth)
# --------------------------------------------------------------------------- #

class TestHitAllowed:
    PUBLIC = {
        "visibility": "public",
        "source_id": "src_public_kb",
        "acl_departments": ["*"],
        "acl_min_role": "anonymous",
    }
    RESTRICTED_ENG = {
        "visibility": "restricted",
        "source_id": "src_internal_docs",
        "acl_departments": ["engineering"],
        "acl_min_role": "manager",
    }
    INTERNAL_WILDCARD = {
        "visibility": "internal",
        "source_id": "src_internal_docs",
        "acl_departments": ["*"],
        "acl_min_role": "employee",
    }

    def test_public_is_always_allowed(self, anonymous):
        assert hit_allowed(anonymous, self.PUBLIC) is True

    def test_external_cannot_see_internal(self, anonymous):
        assert hit_allowed(anonymous, self.INTERNAL_WILDCARD) is False
        assert hit_allowed(anonymous, self.RESTRICTED_ENG) is False

    def test_employee_wildcard_ok(self, eng_employee):
        assert hit_allowed(eng_employee, self.INTERNAL_WILDCARD) is True

    def test_employee_blocked_by_role(self, eng_employee):
        # Right department, but role < manager
        assert hit_allowed(eng_employee, self.RESTRICTED_ENG) is False

    def test_manager_blocked_by_department(self):
        u = UserContext(
            user_id="u_005",
            tenant_id="tenant_acme_staff",
            department="product",
            role="manager",
        )
        assert hit_allowed(u, self.RESTRICTED_ENG) is False

    def test_manager_in_department_allowed(self):
        u = UserContext(
            user_id="u_001",
            tenant_id="tenant_acme_staff",
            department="engineering",
            role="manager",
        )
        assert hit_allowed(u, self.RESTRICTED_ENG) is True

    def test_extra_grant_overrides_acl(self, ops_manager_with_grant):
        # Restricted doc from a source the user has an extra grant on.
        payload = {
            "visibility": "restricted",
            "source_id": "src_github_private",
            "acl_departments": ["engineering"],  # NOT ops
            "acl_min_role": "admin",             # NOT manager
        }
        assert hit_allowed(ops_manager_with_grant, payload) is True

    def test_legacy_acl_dict_shape(self, eng_employee):
        # BM25 writer stores ACL as a nested dict under "acl".
        payload = {
            "visibility": "internal",
            "source_id": "src_internal_docs",
            "acl": {"departments": ["engineering"], "min_role": "employee"},
        }
        assert hit_allowed(eng_employee, payload) is True


# --------------------------------------------------------------------------- #
# Qdrant filter builder (pushdown)
# --------------------------------------------------------------------------- #

class TestBuildQdrantFilter:
    def test_public_collection_no_filter(self, anonymous, settings):
        filt = build_qdrant_filter(anonymous, "public_v1", settings=settings)
        assert filt is None

    def test_public_with_source_allowlist(self, anonymous, settings):
        filt = build_qdrant_filter(
            anonymous,
            "public_v1",
            source_allowlist=["src_public_kb"],
            settings=settings,
        )
        assert filt == {"must": [{"key": "source_id", "match": {"any": ["src_public_kb"]}}]}

    def test_private_collection_for_employee(self, eng_employee, settings):
        filt = build_qdrant_filter(
            eng_employee, "tenant_acme_staff_private_v1", settings=settings,
        )
        assert filt is not None
        should = filt["must"][0]["should"]
        # public escape hatch is always present
        assert {"key": "visibility", "match": {"value": "public"}} in should
        # AND clause for department × role
        and_clause = [s for s in should if "must" in s][0]["must"]
        depts = next(c for c in and_clause if c["key"] == "acl_departments")
        roles = next(c for c in and_clause if c["key"] == "acl_min_role")
        assert set(depts["match"]["any"]) == {"*", "engineering"}
        assert set(roles["match"]["any"]) >= {"anonymous", "employee"}
        assert "manager" not in roles["match"]["any"]
        assert "admin" not in roles["match"]["any"]

    def test_private_with_extra_grants(self, ops_manager_with_grant, settings):
        filt = build_qdrant_filter(
            ops_manager_with_grant,
            "tenant_acme_staff_private_v1",
            settings=settings,
        )
        should = filt["must"][0]["should"]
        grants = [s for s in should if s.get("key") == "source_id"]
        assert grants == [{"key": "source_id", "match": {"any": ["src_github_private"]}}]

    def test_kind_filter_only_when_restricted(self, anonymous, settings):
        # Both kinds allowed → no clause
        filt = build_qdrant_filter(
            anonymous, "public_v1",
            allowed_match_kinds=["content", "question"], settings=settings,
        )
        assert filt is None

        # Only one kind → emit clause
        filt = build_qdrant_filter(
            anonymous, "public_v1",
            allowed_match_kinds=["content"], settings=settings,
        )
        assert filt == {"must": [{"key": "kind", "match": {"any": ["content"]}}]}
