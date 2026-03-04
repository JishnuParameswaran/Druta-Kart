-- Run this in Supabase SQL Editor
-- Creates all tables for the Druta Kart agentic support backend.
-- Tables are ordered by FK dependency so there are no forward references.
-- gen_random_uuid() is available natively in PostgreSQL 13+ (all Supabase projects).

-- ===========================================================================
-- 1. customers
--    Source: db/models.py (CustomerProfile), db/customer_repo.py (_TABLE)
--            analytics_repo.py (client.table("customers"))
-- ===========================================================================

CREATE TABLE IF NOT EXISTS customers (
    user_id              TEXT         PRIMARY KEY,
    name                 TEXT         NOT NULL,
    phone                TEXT,
    total_orders         INTEGER      NOT NULL DEFAULT 0,
    avg_spend_inr        NUMERIC      NOT NULL DEFAULT 0.0,
    complaint_count      INTEGER      NOT NULL DEFAULT 0,
    last_complaint_date  TIMESTAMPTZ,
    satisfaction_score   NUMERIC      CHECK (satisfaction_score >= 0.0 AND satisfaction_score <= 5.0),
    customer_segment     TEXT         NOT NULL DEFAULT 'new'
                             CHECK (customer_segment IN (
                                 'new', 'regular', 'bulk', 'churning', 'frequent_complainer'
                             )),
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Segment queries in _compute_segment / get_customer_segment
CREATE INDEX IF NOT EXISTS idx_customers_segment ON customers (customer_segment);
-- Phone lookups for customer identification
CREATE INDEX IF NOT EXISTS idx_customers_phone   ON customers (phone);

-- Trigger to keep updated_at current on every UPDATE
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
-- 2. chat_sessions
--    Source: db/models.py (ChatSession), analytics_repo.py (_SESSIONS_TABLE)
-- ===========================================================================

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id         TEXT         PRIMARY KEY,
    user_id            TEXT         NOT NULL REFERENCES customers (user_id),
    started_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    ended_at           TIMESTAMPTZ,
    resolution_status  TEXT         NOT NULL DEFAULT 'pending'
                           CHECK (resolution_status IN (
                               'resolved', 'unresolved', 'escalated', 'pending'
                           )),
    emotion_detected   TEXT,
    language_detected  TEXT
);

-- Frequent join/filter in analytics and per-customer session history
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions (user_id);


-- ===========================================================================
-- 3. chat_messages
--    Source: db/models.py (ChatMessage), analytics_repo.py (_MESSAGES_TABLE)
--            log_interaction() inserts rows for both 'user' and 'bot' roles.
--            get_daily_stats() queries: user_id, timestamp, hallucination_flagged, role
-- ===========================================================================

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id            TEXT         PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    session_id            TEXT         NOT NULL REFERENCES chat_sessions (session_id),
    user_id               TEXT         NOT NULL REFERENCES customers (user_id),
    role                  TEXT         NOT NULL CHECK (role IN ('user', 'bot')),
    content               TEXT         NOT NULL,
    timestamp             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    emotion               TEXT,
    language              TEXT,
    agent_used            TEXT,
    tools_called          TEXT[]       NOT NULL DEFAULT '{}',
    latency_ms            INTEGER,
    tokens_used           INTEGER,
    hallucination_flagged BOOLEAN      NOT NULL DEFAULT FALSE
);

-- session_id: retrieved per-session in websocket replay / log_interaction
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages (session_id);
-- user_id: per-customer message history
CREATE INDEX IF NOT EXISTS idx_chat_messages_user_id    ON chat_messages (user_id);
-- timestamp: range filter in get_daily_stats() (.gte("timestamp", cutoff))
CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp  ON chat_messages (timestamp);


-- ===========================================================================
-- 4. complaint_logs
--    Source: db/models.py (ComplaintLog), analytics_repo.py (_COMPLAINTS_TABLE)
--            get_complaint_analytics() selects *, filters on created_at
--            get_common_complaints()   selects complaint_type
--            get_daily_stats()         selects created_at (.gte filter)
-- ===========================================================================

CREATE TABLE IF NOT EXISTS complaint_logs (
    complaint_id    TEXT         PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    session_id      TEXT         NOT NULL REFERENCES chat_sessions (session_id),
    user_id         TEXT         NOT NULL REFERENCES customers (user_id),
    complaint_type  TEXT         NOT NULL
                        CHECK (complaint_type IN (
                            'damaged', 'missing', 'wrong', 'late', 'payment'
                        )),
    product_name    TEXT,
    resolution_type TEXT         NOT NULL DEFAULT 'none'
                        CHECK (resolution_type IN (
                            'refund', 'replacement', 'wallet', 'offer', 'none'
                        )),
    resolved_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- user_id: per-customer complaint history, update_customer_stats complaint counter
CREATE INDEX IF NOT EXISTS idx_complaint_logs_user_id        ON complaint_logs (user_id);
-- session_id: join back to session for resolution context
CREATE INDEX IF NOT EXISTS idx_complaint_logs_session_id     ON complaint_logs (session_id);
-- complaint_type: aggregation in get_complaint_analytics() and get_common_complaints()
CREATE INDEX IF NOT EXISTS idx_complaint_logs_complaint_type ON complaint_logs (complaint_type);
-- created_at: range filter in get_daily_stats() and get_complaint_analytics()
CREATE INDEX IF NOT EXISTS idx_complaint_logs_created_at     ON complaint_logs (created_at);


-- ===========================================================================
-- 5. offers_given
--    Source: db/models.py (OfferGiven)
--            Offer safety caps enforced in config.py: max ₹200 wallet credit,
--            max 35% discount, max 2 free items — NOT enforced here by CHECK
--            intentionally (caps live in application config only).
-- ===========================================================================

CREATE TABLE IF NOT EXISTS offers_given (
    offer_id          TEXT         PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    user_id           TEXT         NOT NULL REFERENCES customers (user_id),
    session_id        TEXT         NOT NULL REFERENCES chat_sessions (session_id),
    offer_type        TEXT         NOT NULL,   -- 'wallet_credit' | 'discount' | 'free_item'
    offer_value       NUMERIC      NOT NULL,   -- INR amount or percentage
    offer_description TEXT         NOT NULL,
    accepted          BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Per-customer offer history (retention / churn analysis)
CREATE INDEX IF NOT EXISTS idx_offers_given_user_id    ON offers_given (user_id);
-- Per-session offer lookup
CREATE INDEX IF NOT EXISTS idx_offers_given_session_id ON offers_given (session_id);


-- ===========================================================================
-- 6. dispatch_checklists
--    Source: db/models.py (DispatchChecklist),
--            tools/dispatch_checklist_tool.py (client.table("dispatch_checklists"))
--            Inserts: checklist_id, session_id, user_id, issues_reported,
--                     checklist_items, sent_at
-- ===========================================================================

CREATE TABLE IF NOT EXISTS dispatch_checklists (
    checklist_id    TEXT         PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    session_id      TEXT         NOT NULL REFERENCES chat_sessions (session_id),
    user_id         TEXT         NOT NULL REFERENCES customers (user_id),
    issues_reported TEXT[]       NOT NULL DEFAULT '{}',
    checklist_items TEXT[]       NOT NULL DEFAULT '{}',
    sent_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- session_id: look up checklists generated within a session
CREATE INDEX IF NOT EXISTS idx_dispatch_checklists_session_id ON dispatch_checklists (session_id);
-- user_id: per-customer dispatch history
CREATE INDEX IF NOT EXISTS idx_dispatch_checklists_user_id    ON dispatch_checklists (user_id);


-- ===========================================================================
-- 7. fraud_flags
--    Source: agents/fraud_escalation_agent.py (_log_fraud_flag)
--            Inserts: flag_id, user_id, session_id, reason, image_path,
--                     status ('pending_review'), created_at
-- ===========================================================================

CREATE TABLE IF NOT EXISTS fraud_flags (
    flag_id     TEXT         PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    user_id     TEXT         NOT NULL REFERENCES customers (user_id),
    session_id  TEXT         NOT NULL REFERENCES chat_sessions (session_id),
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
-- status: filter pending_review items for the QA dashboard
CREATE INDEX IF NOT EXISTS idx_fraud_flags_status     ON fraud_flags (status);
