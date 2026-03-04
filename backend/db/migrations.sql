-- Run this in Supabase SQL Editor
-- Safe incremental migration — adds only what is missing.
-- No DROP statements. No table recreations. Idempotent (safe to re-run).
--
-- Existing tables confirmed present:
--   customer_profiles, chat_sessions, chat_messages, complaint_logs,
--   offers_given, dispatch_checklists, wallet_transactions,
--   refund_requests, replacement_requests
--
-- Problems fixed here:
--   1. NAME MISMATCH: customer_profiles → customers  (code hardcodes "customers")
--   2. MISSING TABLE:  fraud_flags                   (fraud_escalation_agent.py)
--   3. MISSING COLUMNS on 4 existing tables          (see sections below)


-- ===========================================================================
-- 1. NAME MISMATCH: rename customer_profiles → customers
--
--    customer_repo.py:16   _TABLE = "customers"
--    analytics_repo.py:175 client.table("customers")
--
--    ALTER TABLE RENAME preserves all data, indexes, sequences, and
--    FK constraints (Postgres tracks FKs by OID, not by name).
--    Any existing FK constraints on other tables that reference
--    customer_profiles(user_id) will continue to work after the rename.
-- ===========================================================================

ALTER TABLE IF EXISTS customer_profiles RENAME TO customers;


-- ===========================================================================
-- 2. MISSING COLUMNS on customers
--
--    customer_repo.py inserts/updates all of these fields via model_dump().
--    A generic customer_profiles table is unlikely to have had these.
--
--    Column                 | Used in
--    -----------------------|----------------------------------------------------
--    avg_spend_inr          | _compute_segment() bulk threshold check
--    complaint_count        | _compute_segment(), update_customer_stats()
--    last_complaint_date    | _compute_segment() churn check, update_customer_stats()
--    satisfaction_score     | get_customer_satisfaction_score(), update_customer_stats()
--    customer_segment       | create_customer_profile(), update_customer_stats()
--    updated_at             | update_customer_stats() (set on every update)
-- ===========================================================================

ALTER TABLE customers
    ADD COLUMN IF NOT EXISTS avg_spend_inr       NUMERIC      NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS complaint_count     INTEGER      NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_complaint_date TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS satisfaction_score  NUMERIC
                                                     CHECK (satisfaction_score >= 0.0
                                                        AND satisfaction_score <= 5.0),
    ADD COLUMN IF NOT EXISTS customer_segment    TEXT         NOT NULL DEFAULT 'new'
                                                     CHECK (customer_segment IN (
                                                         'new', 'regular', 'bulk',
                                                         'churning', 'frequent_complainer'
                                                     )),
    ADD COLUMN IF NOT EXISTS updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW();

-- Index on segment (used by _compute_segment fan-out queries)
CREATE INDEX IF NOT EXISTS idx_customers_segment ON customers (customer_segment);
-- Index on phone (customer identification lookups)
CREATE INDEX IF NOT EXISTS idx_customers_phone   ON customers (phone)
    WHERE phone IS NOT NULL;

-- Auto-update updated_at on every row change
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_customers_updated_at ON customers;
CREATE TRIGGER trg_customers_updated_at
    BEFORE UPDATE ON customers
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ===========================================================================
-- 3. MISSING COLUMNS on chat_sessions
--
--    Column            | Used in
--    ------------------|-------------------------------------------------------
--    ended_at          | ChatSession model (nullable, set when session closes)
--    emotion_detected  | ChatSession model (NLP pipeline output)
--    language_detected | ChatSession model (lingua language detector output)
--
--    resolution_status is likely already present since it is a core session
--    field, but ADD COLUMN IF NOT EXISTS is a no-op if it already exists.
-- ===========================================================================

ALTER TABLE chat_sessions
    ADD COLUMN IF NOT EXISTS ended_at           TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS emotion_detected   TEXT,
    ADD COLUMN IF NOT EXISTS language_detected  TEXT,
    ADD COLUMN IF NOT EXISTS resolution_status  TEXT  NOT NULL DEFAULT 'pending'
                                                    CHECK (resolution_status IN (
                                                        'resolved', 'unresolved',
                                                        'escalated', 'pending'
                                                    ));

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions (user_id);


-- ===========================================================================
-- 4. MISSING COLUMNS on chat_messages
--
--    analytics_repo.log_interaction() inserts these exact keys for BOTH the
--    'user' row and the 'bot' row on every turn:
--
--    Column                | Inserted value
--    ----------------------|----------------------------------------------------
--    agent_used            | name of the specialist agent that handled the turn
--    tools_called          | TEXT[] of LangChain tool names called
--    latency_ms            | INTEGER, None for user row / actual ms for bot row
--    tokens_used           | INTEGER, None for user row / token count for bot row
--    hallucination_flagged | BOOLEAN, False for user row / actual flag for bot row
--
--    get_daily_stats() selects: user_id, timestamp, hallucination_flagged, role
--    so hallucination_flagged must exist for that query to work at all.
-- ===========================================================================

ALTER TABLE chat_messages
    ADD COLUMN IF NOT EXISTS agent_used            TEXT,
    ADD COLUMN IF NOT EXISTS tools_called          TEXT[]   NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS latency_ms            INTEGER,
    ADD COLUMN IF NOT EXISTS tokens_used           INTEGER,
    ADD COLUMN IF NOT EXISTS hallucination_flagged BOOLEAN  NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages (session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_user_id    ON chat_messages (user_id);
-- timestamp index is critical for get_daily_stats() .gte("timestamp", cutoff) filter
CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp  ON chat_messages (timestamp);


-- ===========================================================================
-- 5. MISSING COLUMNS on complaint_logs
--
--    analytics_repo.get_complaint_analytics() reads these columns via SELECT *:
--
--    Column          | Used in
--    ----------------|----------------------------------------------------------
--    product_name    | get_complaint_analytics() → by_product aggregation
--    resolution_type | get_complaint_analytics() → resolution_rate calculation
--    resolved_at     | get_complaint_analytics() → avg_resolution_time calculation
-- ===========================================================================

ALTER TABLE complaint_logs
    ADD COLUMN IF NOT EXISTS product_name    TEXT,
    ADD COLUMN IF NOT EXISTS resolution_type TEXT  NOT NULL DEFAULT 'none'
                                                 CHECK (resolution_type IN (
                                                     'refund', 'replacement',
                                                     'wallet', 'offer', 'none'
                                                 )),
    ADD COLUMN IF NOT EXISTS resolved_at     TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_complaint_logs_user_id        ON complaint_logs (user_id);
CREATE INDEX IF NOT EXISTS idx_complaint_logs_session_id     ON complaint_logs (session_id);
-- complaint_type index: used by get_complaint_analytics() and get_common_complaints()
CREATE INDEX IF NOT EXISTS idx_complaint_logs_complaint_type ON complaint_logs (complaint_type);
-- created_at index: used by get_daily_stats() .gte("created_at", cutoff)
CREATE INDEX IF NOT EXISTS idx_complaint_logs_created_at     ON complaint_logs (created_at);


-- ===========================================================================
-- 6. MISSING TABLE: fraud_flags
--
--    fraud_escalation_agent.py:39 inserts into this table.
--    It does NOT exist in the user's current Supabase schema.
--
--    Exact insert payload from _log_fraud_flag():
--        flag_id, user_id, session_id, reason, image_path,
--        status ("pending_review"), created_at
-- ===========================================================================

CREATE TABLE IF NOT EXISTS fraud_flags (
    flag_id     TEXT         PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    user_id     TEXT         NOT NULL,
    session_id  TEXT         NOT NULL,
    reason      TEXT         NOT NULL,
    image_path  TEXT         NOT NULL,
    status      TEXT         NOT NULL DEFAULT 'pending_review'
                    CHECK (status IN ('pending_review', 'confirmed', 'dismissed')),
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- user_id: lookup all fraud events for a customer
CREATE INDEX IF NOT EXISTS idx_fraud_flags_user_id    ON fraud_flags (user_id);
-- session_id: look up fraud events within a session
CREATE INDEX IF NOT EXISTS idx_fraud_flags_session_id ON fraud_flags (session_id);
-- status: QA dashboard filter for pending_review items
CREATE INDEX IF NOT EXISTS idx_fraud_flags_status     ON fraud_flags (status);


-- ===========================================================================
-- 7. MISSING TABLE: security_events
--
--    security/safety_layer.log_security_event() inserts into this table
--    whenever a prompt-injection or red-team probe is detected.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS security_events (
    id          TEXT         PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    event_type  TEXT         NOT NULL,
    user_id     TEXT,
    session_id  TEXT,
    detail      TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_security_events_user_id    ON security_events (user_id);
CREATE INDEX IF NOT EXISTS idx_security_events_event_type ON security_events (event_type);
CREATE INDEX IF NOT EXISTS idx_security_events_created_at ON security_events (created_at);


-- ===========================================================================
-- 8. MISSING TABLE: orders
--
--    tools/order_lookup_tool.py:33 queries client.table("orders")
--    This table is not present in the original schema — add it here.
--
--    Columns match exactly what order_lookup_tool returns:
--        order_id, user_id, status, items, amount_inr,
--        placed_at, estimated_delivery, delivery_partner, tracking_url
-- ===========================================================================

CREATE TABLE IF NOT EXISTS orders (
    order_id            TEXT         PRIMARY KEY,
    user_id             TEXT         NOT NULL,
    status              TEXT         NOT NULL DEFAULT 'processing'
                            CHECK (status IN (
                                'processing', 'out_for_delivery',
                                'delivered', 'delayed', 'cancelled'
                            )),
    items               JSONB        NOT NULL DEFAULT '[]',
    amount_inr          NUMERIC      NOT NULL DEFAULT 0.0,
    placed_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    estimated_delivery  TIMESTAMPTZ,
    delivery_partner    TEXT,
    tracking_url        TEXT,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders (user_id);
CREATE INDEX IF NOT EXISTS idx_orders_status  ON orders (status);


-- ===========================================================================
-- NOTE: wallet_transactions, refund_requests, replacement_requests
--
--    These tables exist in your Supabase schema but are NOT referenced by
--    the four files audited here (customer_repo, analytics_repo,
--    fraud_escalation_agent, dispatch_checklist_tool).
--    They are likely used by tools/refund_tool.py, tools/replacement_tool.py,
--    and tools/wallet_credit_tool.py — check those files separately if
--    column additions are needed there.
-- ===========================================================================
