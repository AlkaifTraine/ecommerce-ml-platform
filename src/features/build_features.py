"""Truncation-point feature builder - the core of the modelling contract.

The prediction task
-------------------
Stand inside a live session immediately after its k-th event. Using ONLY those
k events plus history that predates the session, predict whether a purchase
happens later in the same session.

Eligibility rules (these are what make the numbers real):

1. The session must have at least k events. You cannot make a k-event
   prediction on a session that never reached k events.
2. No purchase may occur within the first k events. If the customer already
   bought, there is nothing left to predict and including those rows would
   hand the model the answer.
3. The session must not touch a quarantined day. The source pipeline lost
   purchase events on 2019-11-14..17 (see scripts/data_quality_audit.py), so
   labels there are unknowable, not negative.
4. The label looks strictly at events k+1..end. Nothing at or before the
   cutoff contributes to the label; nothing at or after the cutoff contributes
   to the features.

`--hard-mode` additionally strips cart signals from the features. Cart activity
before the cutoff is legitimately observable in production, so the default
keeps it. Hard mode matters more here than it first appears: cart events are
badly under-recorded in October (buy/cart ~69% vs ~34% in November), so
cart-derived features are not comparable across the two months.

Execution notes
---------------
Built as staged temp tables rather than one large CTE chain. The CTE version
referenced `ranked` three times, and DuckDB recomputed the 110M-row window
function each time. Pre-filtering to sessions that can possibly qualify (via
the session index) before ranking cuts the sort down substantially.
"""

from __future__ import annotations

import argparse
import time

import duckdb

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)


def _con(settings) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def build(settings, k: int, hard_mode: bool = False, until: str | None = None) -> int:
    con = _con(settings)
    events = f"read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')"
    sessions = f"read_parquet('{(settings.features_dir / 'sessions.parquet').as_posix()}')"

    time_filter = f"AND e.event_time < TIMESTAMP '{until}'" if until else ""
    if until:
        log.info("clock enforced: events at or after %s are invisible", until)

    # A session is quarantined if any quarantined day falls inside its span.
    # Checking the span (not just the start date) also catches sessions that
    # begin before the outage and run into it.
    if settings.quarantine_dates:
        q_pred = " OR ".join(
            f"(DATE '{d}' BETWEEN CAST(session_start AS DATE) AND CAST(session_end AS DATE))"
            for d in settings.quarantine_dates
        )
        quarantine_filter = f"AND NOT ({q_pred})"
        log.info("quarantining %d day(s): %s",
                 len(settings.quarantine_dates), ", ".join(settings.quarantine_dates))
    else:
        quarantine_filter = ""

    cart_features = (
        "0 AS n_cart_events, 0.0 AS cart_ratio"
        if hard_mode
        else """
        sum(CASE WHEN event_type='cart' THEN 1 ELSE 0 END)     AS n_cart_events,
        avg(CASE WHEN event_type='cart' THEN 1.0 ELSE 0.0 END) AS cart_ratio
        """
    )

    suffix = f"k{k}" + ("_hard" if hard_mode else "")
    out = settings.features_dir / f"train_{suffix}.parquet"
    log.info("building features k=%d hard_mode=%s -> %s", k, hard_mode, out.name)
    t0 = time.time()

    def step(label: str, sql: str) -> None:
        t = time.time()
        con.execute(sql)
        log.info("  [%5.1fs] %s", time.time() - t, label)

    # 1. Narrow to sessions that could possibly qualify BEFORE doing any
    #    ranking. This is the single biggest saving: ~23M sessions drop to the
    #    few million long enough to reach k events.
    step(
        "candidate sessions (length + quarantine)",
        f"""
        CREATE OR REPLACE TEMP TABLE cand AS
        SELECT session_key, user_id, session_start, session_end,
               user_prior_sessions, user_prior_purchases, user_prior_revenue,
               hours_since_last_session
        FROM {sessions}
        WHERE n_events >= {k} {quarantine_filter}
        """,
    )
    n_cand = con.execute("SELECT count(*) FROM cand").fetchone()[0]
    log.info("           candidates: %s sessions", f"{n_cand:,}")

    # 2. Rank only those sessions' events, once, into a materialised table.
    # ORDERING MUST BE TOTAL, or row_number() is not reproducible.
    #
    # An earlier version ordered by (event_time, product_id) alone. Timestamps
    # have one-second granularity and ~0.2% of rows are exact duplicates, so
    # that key ties outright - a view and a purchase of the same product in the
    # same second are indistinguishable to the sort. row_number() then assigns
    # ranks arbitrarily, differently on each run, and the leakage audit caught
    # exactly one session out of 5,153,372 where the feature builder and the
    # audit disagreed about which event was 5th.
    #
    # Fix has two parts:
    #   1. collapse exact duplicate events - a repeated (session, time, type,
    #      product) row is a collection artifact, not two real actions;
    #   2. order by (event_time, product_id, event_type), which after that
    #      dedup is unique within a session.
    #
    # The dedup is a GROUP BY, not a second window function. Deduping with
    # row_number() OVER (PARTITION BY ...) needs a second full sort of ~70M
    # rows and pushed the build back into OOM; a hash aggregate spills to disk
    # and min() over the payload columns keeps the choice deterministic.
    step(
        "rank events within candidate sessions",
        f"""
        CREATE OR REPLACE TEMP TABLE ranked AS
        SELECT session_key, event_time, event_type, product_id,
               category_id, category_code, brand, price,
               row_number() OVER (PARTITION BY session_key
                                  ORDER BY event_time, product_id, event_type) AS rn
        FROM (
            SELECT c.session_key, e.event_time, e.event_type, e.product_id,
                   min(e.category_id)   AS category_id,
                   min(e.category_code) AS category_code,
                   min(e.brand)         AS brand,
                   min(e.price)         AS price
            FROM {events} e
            JOIN cand c ON e.user_session = c.session_key
            WHERE 1=1 {time_filter}
            GROUP BY 1, 2, 3, 4
        )
        """,
    )

    # 3. Drop sessions that already purchased inside the prefix.
    step(
        "apply pre-cutoff purchase rule",
        f"""
        CREATE OR REPLACE TEMP TABLE eligible AS
        SELECT session_key FROM ranked
        GROUP BY session_key
        HAVING count(*) >= {k}
           AND sum(CASE WHEN rn <= {k} AND event_type='purchase' THEN 1 ELSE 0 END) = 0
        """,
    )
    n_elig = con.execute("SELECT count(*) FROM eligible").fetchone()[0]
    log.info("           eligible: %s sessions", f"{n_elig:,}")

    # Timestamps in this dataset have one-second granularity, so events inside
    # a session frequently tie. Ordering by (event_time, product_id) makes the
    # prefix deterministic, but it leaves a genuine ambiguity at the boundary:
    # an event ranked k+1 can share the cutoff second with the event ranked k.
    #
    # Labelling such an event as "after the cutoff" is optimistic - at the
    # instant we score, it may already have happened. So the label window is
    # defined by TIME, strictly after cutoff_time, not by rank. Events sharing
    # the cutoff second are excluded from both the features and the label.
    # This costs a few positives and removes the ambiguity entirely.
    step(
        "split prefix / label window",
        f"""
        CREATE OR REPLACE TEMP TABLE prefix AS
        SELECT r.* FROM ranked r JOIN eligible g ON g.session_key = r.session_key
        WHERE r.rn <= {k};
        CREATE OR REPLACE TEMP TABLE cutoffs AS
        SELECT session_key, max(event_time) AS cutoff_time FROM prefix GROUP BY 1;
        CREATE OR REPLACE TEMP TABLE lbl AS
        SELECT r.session_key,
               max(CASE WHEN r.event_type='purchase' THEN 1 ELSE 0 END) AS y
        FROM ranked r
        JOIN cutoffs c ON c.session_key = r.session_key
        WHERE r.event_time > c.cutoff_time
        GROUP BY r.session_key;
        """,
    )

    # `ranked` holds every event of every candidate session (tens of millions of
    # rows). Once the prefix and the label window are extracted it is dead
    # weight, and on a machine with ~5GB free it is the difference between
    # finishing and an OOM. Drop it before the aggregation stage.
    step("release ranked events", "DROP TABLE ranked; DROP TABLE eligible;")

    # count(DISTINCT x) GROUP BY session_key builds one hash set per session -
    # five million of them - and DuckDB cannot spill those, which is exactly
    # how the previous version died at 7.4GiB. Two-stage aggregation
    # (DISTINCT, then COUNT) is an ordinary hash aggregate and spills fine.
    step(
        "distinct counts (two-stage, spillable)",
        """
        CREATE OR REPLACE TEMP TABLE per_product AS
            SELECT session_key, product_id, count(*) AS views FROM prefix GROUP BY 1,2;
        CREATE OR REPLACE TEMP TABLE revisit AS
            SELECT session_key,
                   count(*)    AS n_distinct_products,
                   max(views)  AS max_views_same_product,
                   avg(views)  AS mean_views_per_product
            FROM per_product GROUP BY 1;
        DROP TABLE per_product;
        CREATE OR REPLACE TEMP TABLE dcat AS
            SELECT session_key, count(*) AS n_distinct_categories
            FROM (SELECT DISTINCT session_key, category_id FROM prefix) GROUP BY 1;
        -- brand is NULL for ~14% of events. count(DISTINCT brand) ignores
        -- NULLs, so the DISTINCT subquery must exclude them too or a session
        -- of unbranded items would report one "distinct brand". Sessions with
        -- no branded item drop out here and are restored by a LEFT JOIN below.
        CREATE OR REPLACE TEMP TABLE dbrand AS
            SELECT session_key, count(*) AS n_distinct_brands
            FROM (SELECT DISTINCT session_key, brand FROM prefix WHERE brand IS NOT NULL)
            GROUP BY 1;
        """,
    )

    step(
        "aggregate prefix features",
        f"""
        CREATE OR REPLACE TEMP TABLE gaps AS
            SELECT session_key, avg(gap) AS mean_gap_sec, median(gap) AS median_gap_sec,
                   min(gap) AS min_gap_sec, max(gap) AS max_gap_sec
            FROM (
                SELECT session_key,
                       date_diff('second',
                           lag(event_time) OVER (PARTITION BY session_key ORDER BY rn),
                           event_time) AS gap
                FROM prefix
            ) WHERE gap IS NOT NULL GROUP BY 1;
        CREATE OR REPLACE TEMP TABLE base AS
            SELECT session_key,
                   date_diff('second', min(event_time), max(event_time)) AS prefix_duration_sec,
                   avg(price) AS price_mean, min(price) AS price_min, max(price) AS price_max,
                   COALESCE(stddev_pop(price), 0.0) AS price_std,
                   max(price) - min(price) AS price_range,
                   avg(CASE WHEN brand IS NULL THEN 1.0 ELSE 0.0 END) AS null_brand_ratio,
                   avg(CASE WHEN category_code IS NULL THEN 1.0 ELSE 0.0 END) AS null_category_ratio,
                   {cart_features},
                   arg_max(price, rn) AS price_at_cutoff,
                   max(event_time)    AS cutoff_time
            FROM prefix GROUP BY session_key;
        DROP TABLE prefix;
        """,
    )

    step(
        "assemble and write training table",
        f"""
        COPY (
            SELECT
                b.session_key, c.user_id, c.session_start, b.cutoff_time,
                CAST(c.session_start AS DATE) AS session_date,

                r.n_distinct_products, dc.n_distinct_categories,
                COALESCE(db.n_distinct_brands, 0) AS n_distinct_brands,
                {k} - r.n_distinct_products AS n_repeat_views,
                r.max_views_same_product, r.mean_views_per_product,
                1.0 - (r.n_distinct_products * 1.0 / {k}) AS repeat_product_ratio,

                b.prefix_duration_sec, g.mean_gap_sec, g.median_gap_sec,
                g.min_gap_sec, g.max_gap_sec,
                CASE WHEN b.prefix_duration_sec > 0
                     THEN {k} * 60.0 / b.prefix_duration_sec ELSE NULL END AS events_per_minute,

                b.price_mean, b.price_min, b.price_max, b.price_std,
                b.price_range, b.price_at_cutoff,
                b.null_brand_ratio, b.null_category_ratio,
                b.n_cart_events, b.cart_ratio,

                hour(c.session_start)      AS hour_of_day,
                dayofweek(c.session_start) AS day_of_week,
                CASE WHEN dayofweek(c.session_start) IN (0,6) THEN 1 ELSE 0 END AS is_weekend,

                c.user_prior_sessions, c.user_prior_purchases,
                CASE WHEN c.user_prior_sessions > 0
                     THEN c.user_prior_purchases * 1.0 / c.user_prior_sessions
                     ELSE NULL END AS user_prior_conv_rate,
                c.user_prior_revenue, c.hours_since_last_session,
                CASE WHEN c.user_prior_sessions = 0 THEN 1 ELSE 0 END AS is_new_user,

                COALESCE(l.y, 0) AS y
            FROM base b
            JOIN      cand    c  ON c.session_key  = b.session_key
            JOIN      revisit r  ON r.session_key  = b.session_key
            JOIN      dcat    dc ON dc.session_key = b.session_key
            LEFT JOIN dbrand  db ON db.session_key = b.session_key
            LEFT JOIN gaps    g  ON g.session_key  = b.session_key
            LEFT JOIN lbl     l  ON l.session_key  = b.session_key
        )
        TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
    )

    n, pos = con.execute(
        f"SELECT count(*), avg(y) FROM read_parquet('{out.as_posix()}')"
    ).fetchone()
    log.info("k=%d rows=%s positive rate=%.3f%%  (total %.1fs)",
             k, f"{int(n):,}", float(pos) * 100, time.time() - t0)
    con.close()
    return int(n)


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=None, help="single k; default builds all configured k")
    ap.add_argument("--hard-mode", action="store_true", help="strip cart signals from features")
    ap.add_argument("--until", default=None, help="clock cutoff timestamp")
    args = ap.parse_args()

    ks = [args.k] if args.k else list(settings.truncation_ks)
    for k in ks:
        build(settings, k=k, hard_mode=args.hard_mode, until=args.until)


if __name__ == "__main__":
    main()
