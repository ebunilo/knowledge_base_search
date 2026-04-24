-- =============================================================================
-- Bootstrap schema for the KB search doc store + metadata + audit log.
-- Runs once on first container start (Postgres initdb lifecycle).
-- Idempotent where possible (IF NOT EXISTS).
-- =============================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- trigram index for fuzzy metadata lookups

-- Schemas ---------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS kb;
CREATE SCHEMA IF NOT EXISTS audit;

-- ============================================================================
--  Record manager (LangChain-compatible incremental indexing)
-- ============================================================================
-- The app layer will also manage this via LangChain's SQLRecordManager,
-- but we pre-create it so we own the schema explicitly.
CREATE TABLE IF NOT EXISTS kb.upsertion_record (
    uuid TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    namespace TEXT NOT NULL,
    group_id TEXT,
    updated_at DOUBLE PRECISION NOT NULL,
    UNIQUE (namespace, key)
);
CREATE INDEX IF NOT EXISTS idx_upsertion_record_updated_at
    ON kb.upsertion_record (updated_at);
CREATE INDEX IF NOT EXISTS idx_upsertion_record_group_id
    ON kb.upsertion_record (group_id);

-- ============================================================================
--  Documents + parent chunks (the "doc store" half of parent-child retrieval)
-- ============================================================================
CREATE TABLE IF NOT EXISTS kb.document (
    document_id     TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL,
    source_uri      TEXT NOT NULL,
    title           TEXT,
    content_hash    TEXT NOT NULL,
    language        TEXT DEFAULT 'en',
    region          TEXT,
    sensitivity_lane TEXT NOT NULL CHECK (sensitivity_lane IN ('hosted_ok','self_hosted_only')),
    visibility      TEXT NOT NULL CHECK (visibility IN ('public','internal','restricted')),
    acl             JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_document_source_id   ON kb.document (source_id);
CREATE INDEX IF NOT EXISTS idx_document_visibility  ON kb.document (visibility);
CREATE INDEX IF NOT EXISTS idx_document_acl_gin     ON kb.document USING GIN (acl);
CREATE INDEX IF NOT EXISTS idx_document_metadata_gin ON kb.document USING GIN (metadata);

CREATE TABLE IF NOT EXISTS kb.parent_chunk (
    parent_id       TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES kb.document(document_id) ON DELETE CASCADE,
    ord             INTEGER NOT NULL,
    content         TEXT NOT NULL,
    token_count     INTEGER,
    section_path    TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_parent_chunk_document_id ON kb.parent_chunk (document_id);

CREATE TABLE IF NOT EXISTS kb.child_chunk (
    child_id        TEXT PRIMARY KEY,
    parent_id       TEXT NOT NULL REFERENCES kb.parent_chunk(parent_id) ON DELETE CASCADE,
    document_id     TEXT NOT NULL REFERENCES kb.document(document_id) ON DELETE CASCADE,
    ord             INTEGER NOT NULL,
    content         TEXT NOT NULL,
    token_count     INTEGER,
    hypothetical_questions JSONB DEFAULT '[]'::jsonb,
    summary         TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_child_chunk_parent_id   ON kb.child_chunk (parent_id);
CREATE INDEX IF NOT EXISTS idx_child_chunk_document_id ON kb.child_chunk (document_id);

-- ============================================================================
--  Tenancy + ACL (mirrored from staff_directory.json for lookups)
-- ============================================================================
CREATE TABLE IF NOT EXISTS kb.department (
    department_id    TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    data_residency   TEXT
);

CREATE TABLE IF NOT EXISTS kb.app_user (
    user_id          TEXT PRIMARY KEY,
    email            TEXT UNIQUE NOT NULL,
    display_name     TEXT,
    department_id    TEXT REFERENCES kb.department(department_id),
    role             TEXT NOT NULL,
    extra_grants     JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================================
--  Feedback (thumbs + structured)
-- ============================================================================
CREATE TABLE IF NOT EXISTS kb.feedback (
    feedback_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id       TEXT,
    user_id          TEXT,
    query            TEXT NOT NULL,
    rewritten_query  TEXT,
    answer           TEXT,
    cited_chunk_ids  JSONB NOT NULL DEFAULT '[]'::jsonb,
    rating           SMALLINT CHECK (rating IN (-1, 0, 1)),
    category         TEXT,
    comment          TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_feedback_user       ON kb.feedback (user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating     ON kb.feedback (rating);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON kb.feedback (created_at);

-- ============================================================================
--  Audit log (every query)
-- ============================================================================
CREATE TABLE IF NOT EXISTS audit.query_log (
    log_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ts               TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id          TEXT,
    tenant_id        TEXT,
    session_id       TEXT,
    intent           TEXT,
    lane             TEXT CHECK (lane IN ('hosted_ok','self_hosted_only')),
    lane_isolation   TEXT,                -- 'enforced' | 'demo_override'
    models_used      JSONB NOT NULL DEFAULT '[]'::jsonb,
    query            TEXT NOT NULL,
    rewritten_query  TEXT,
    doc_ids_returned JSONB NOT NULL DEFAULT '[]'::jsonb,
    confidence       REAL,
    is_best_guess    BOOLEAN,
    latency_ms       INTEGER,
    outcome          TEXT                 -- 'ok'|'refused'|'acl_denied'|'error'
);
CREATE INDEX IF NOT EXISTS idx_audit_ts       ON audit.query_log (ts);
CREATE INDEX IF NOT EXISTS idx_audit_user     ON audit.query_log (user_id);
CREATE INDEX IF NOT EXISTS idx_audit_outcome  ON audit.query_log (outcome);

-- ============================================================================
--  Grants
-- ============================================================================
GRANT USAGE ON SCHEMA kb, audit TO CURRENT_USER;
GRANT ALL   ON ALL TABLES IN SCHEMA kb, audit TO CURRENT_USER;
