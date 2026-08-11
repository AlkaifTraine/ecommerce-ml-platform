-- =====================================================================
-- OLTP schema: the storefront's system of record.
--
-- This models the transactional side of the business - the tables a real
-- store would write to on every click. It is deliberately normalised, indexed
-- for row-level access, and constrained. Analytics does NOT run here; that is
-- what the warehouse is for, and the contrast is the point.
--
-- RETENTION: session_events is a rolling hot window (default 7 days of DATA
-- time, not wall time). Older rows are archived to the lake and deleted, which
-- is what keeps an OLTP store fast and small. History is never lost - it lives
-- in Parquet - but it is not queryable from here, which is exactly the
-- constraint that forces a warehouse to exist.
-- =====================================================================

CREATE SCHEMA IF NOT EXISTS storefront;
SET search_path TO storefront, public;

-- ---------------------------------------------------------------------
-- The replay clock. Single authoritative row shared by every service, so
-- "what time does the system think it is" has exactly one answer.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS replay_clock (
    id                  SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    current_data_time   TIMESTAMP    NOT NULL,
    phase               TEXT         NOT NULL DEFAULT 'backfill'
                                     CHECK (phase IN ('backfill', 'live', 'stopped')),
    speed_multiplier    DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    events_emitted      BIGINT       NOT NULL DEFAULT 0,
    started_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- Dimensions
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id         BIGINT PRIMARY KEY,
    first_seen_at   TIMESTAMP NOT NULL,
    last_seen_at    TIMESTAMP NOT NULL,
    total_sessions  INTEGER   NOT NULL DEFAULT 0 CHECK (total_sessions >= 0),
    total_orders    INTEGER   NOT NULL DEFAULT 0 CHECK (total_orders  >= 0),
    lifetime_value  NUMERIC(14,2) NOT NULL DEFAULT 0 CHECK (lifetime_value >= 0)
);

CREATE TABLE IF NOT EXISTS products (
    product_id      BIGINT PRIMARY KEY,
    category_id     BIGINT,
    category_code   TEXT,
    brand           TEXT,
    current_price   NUMERIC(12,2) CHECK (current_price >= 0),
    updated_at      TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_products_category ON products (category_id);

-- ---------------------------------------------------------------------
-- Sessions and the high-volume event stream
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    session_key     TEXT PRIMARY KEY,
    user_id         BIGINT NOT NULL REFERENCES users (user_id),
    started_at      TIMESTAMP NOT NULL,
    last_event_at   TIMESTAMP NOT NULL,
    n_events        INTEGER   NOT NULL DEFAULT 0 CHECK (n_events >= 0),
    status          TEXT      NOT NULL DEFAULT 'open'
                              CHECK (status IN ('open', 'closed')),
    CONSTRAINT sessions_time_order CHECK (last_event_at >= started_at)
);
CREATE INDEX IF NOT EXISTS ix_sessions_user    ON sessions (user_id);
CREATE INDEX IF NOT EXISTS ix_sessions_started ON sessions (started_at);
CREATE INDEX IF NOT EXISTS ix_sessions_open    ON sessions (status) WHERE status = 'open';

CREATE TABLE IF NOT EXISTS session_events (
    event_id    BIGSERIAL PRIMARY KEY,
    session_key TEXT      NOT NULL REFERENCES sessions (session_key) ON DELETE CASCADE,
    seq         INTEGER   NOT NULL CHECK (seq > 0),
    event_time  TIMESTAMP NOT NULL,
    event_type  TEXT      NOT NULL
                CHECK (event_type IN ('view','cart','remove_from_cart','purchase')),
    product_id  BIGINT    NOT NULL,
    price       NUMERIC(12,2),
    UNIQUE (session_key, seq)
);
CREATE INDEX IF NOT EXISTS ix_events_session ON session_events (session_key, seq);
CREATE INDEX IF NOT EXISTS ix_events_time    ON session_events (event_time);

-- ---------------------------------------------------------------------
-- Carts and orders
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS carts (
    cart_id     BIGSERIAL PRIMARY KEY,
    session_key TEXT   NOT NULL REFERENCES sessions (session_key) ON DELETE CASCADE,
    user_id     BIGINT NOT NULL REFERENCES users (user_id),
    created_at  TIMESTAMP NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active','abandoned','converted')),
    UNIQUE (session_key)
);

CREATE TABLE IF NOT EXISTS cart_items (
    cart_id     BIGINT NOT NULL REFERENCES carts (cart_id) ON DELETE CASCADE,
    product_id  BIGINT NOT NULL,
    qty         INTEGER NOT NULL DEFAULT 1 CHECK (qty > 0),
    unit_price  NUMERIC(12,2) NOT NULL CHECK (unit_price >= 0),
    added_at    TIMESTAMP NOT NULL,
    PRIMARY KEY (cart_id, product_id)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id     BIGSERIAL PRIMARY KEY,
    session_key  TEXT   NOT NULL REFERENCES sessions (session_key) ON DELETE CASCADE,
    user_id      BIGINT NOT NULL REFERENCES users (user_id),
    ordered_at   TIMESTAMP NOT NULL,
    total_amount NUMERIC(14,2) NOT NULL CHECK (total_amount >= 0),
    status       TEXT NOT NULL DEFAULT 'placed'
                 CHECK (status IN ('placed','shipped','returned','cancelled'))
);
CREATE INDEX IF NOT EXISTS ix_orders_session ON orders (session_key);
CREATE INDEX IF NOT EXISTS ix_orders_time    ON orders (ordered_at);

CREATE TABLE IF NOT EXISTS order_items (
    order_id   BIGINT NOT NULL REFERENCES orders (order_id) ON DELETE CASCADE,
    product_id BIGINT NOT NULL,
    qty        INTEGER NOT NULL DEFAULT 1 CHECK (qty > 0),
    unit_price NUMERIC(12,2) NOT NULL CHECK (unit_price >= 0),
    PRIMARY KEY (order_id, product_id)
);

-- ---------------------------------------------------------------------
-- ML surface: what the model was asked, what it said, and what we did.
--
-- scored_at_data_time is the REPLAY clock, not wall time. Monitoring joins on
-- it, so drift analysis stays aligned with the simulated calendar.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id       BIGSERIAL PRIMARY KEY,
    session_key         TEXT      NOT NULL,
    model_name          TEXT      NOT NULL,
    model_version       TEXT      NOT NULL,
    model_alias         TEXT      NOT NULL DEFAULT 'champion'
                                  CHECK (model_alias IN ('champion','challenger','shadow')),
    cutoff_seq          INTEGER   NOT NULL CHECK (cutoff_seq > 0),
    scored_at_data_time TIMESTAMP NOT NULL,
    scored_at_wall_time TIMESTAMPTZ NOT NULL DEFAULT now(),
    score               DOUBLE PRECISION NOT NULL CHECK (score BETWEEN 0 AND 1),
    latency_ms          DOUBLE PRECISION,
    feature_snapshot    JSONB,
    UNIQUE (session_key, model_alias, cutoff_seq)
);
CREATE INDEX IF NOT EXISTS ix_pred_data_time ON predictions (scored_at_data_time);
CREATE INDEX IF NOT EXISTS ix_pred_alias     ON predictions (model_alias, scored_at_data_time);

-- Labels land later than predictions. This table is what makes the
-- delayed-label problem explicit rather than hidden.
CREATE TABLE IF NOT EXISTS prediction_outcomes (
    session_key    TEXT PRIMARY KEY,
    label          SMALLINT  NOT NULL CHECK (label IN (0,1)),
    resolved_at    TIMESTAMP NOT NULL,
    revenue        NUMERIC(14,2) NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS interventions (
    intervention_id BIGSERIAL PRIMARY KEY,
    session_key     TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('discount','free_shipping','email','none')),
    score_at_send   DOUBLE PRECISION NOT NULL,
    sent_at         TIMESTAMP NOT NULL,
    -- a small random holdout stays untreated so treatment effect remains
    -- estimable; without it the intervention policy poisons its own training data
    is_holdout      BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS ix_interventions_session ON interventions (session_key);

-- ---------------------------------------------------------------------
-- CDC watermark: how far the lake extractor has drained the OLTP store.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cdc_watermark (
    table_name       TEXT PRIMARY KEY,
    last_event_id    BIGINT    NOT NULL DEFAULT 0,
    last_data_time   TIMESTAMP,
    extracted_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO replay_clock (id, current_data_time, phase)
VALUES (1, TIMESTAMP '2019-10-01 00:00:00', 'stopped')
ON CONFLICT (id) DO NOTHING;

INSERT INTO cdc_watermark (table_name) VALUES
    ('session_events'), ('orders'), ('predictions')
ON CONFLICT (table_name) DO NOTHING;
