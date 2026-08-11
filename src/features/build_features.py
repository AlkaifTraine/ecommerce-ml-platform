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
3. The label looks strictly at events k+1..end. Nothing at or before the
   cutoff contributes to the label; nothing at or after the cutoff contributes
   to the features.

`--hard-mode` additionally strips cart / remove_from_cart signals from the
features. Cart activity before the cutoff is legitimately observable in
production, so the default keeps it - hard mode exists to demonstrate that the
model still works when its single strongest feature is taken away.
"""

from __future__ import annotations

import argparse

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

    time_filter = f"WHERE event_time < TIMESTAMP '{until}'" if until else ""
    if until:
        log.info("clock enforced: events at or after %s are invisible", until)

    cart_features = (
        "0 AS n_cart_events, 0 AS n_remove_events, 0.0 AS cart_ratio"
        if hard_mode
        else """
        sum(CASE WHEN event_type='cart' THEN 1 ELSE 0 END)            AS n_cart_events,
        sum(CASE WHEN event_type='remove_from_cart' THEN 1 ELSE 0 END) AS n_remove_events,
        avg(CASE WHEN event_type='cart' THEN 1.0 ELSE 0.0 END)         AS cart_ratio
        """
    )

    suffix = f"k{k}" + ("_hard" if hard_mode else "")
    out = settings.features_dir / f"train_{suffix}.parquet"
    log.info("building features k=%d hard_mode=%s -> %s", k, hard_mode, out.name)

    con.execute(
        f"""
        COPY (
        WITH ranked AS (
            SELECT e.*,
                   s.session_key,
                   row_number() OVER (PARTITION BY s.session_key ORDER BY e.event_time, e.product_id)
                       AS rn
            FROM {events} e
            JOIN {sessions} s ON e.user_session = s.session_key
            {time_filter}
        ),
        -- Rule 1 + 2: long enough, and not already purchased before the cutoff.
        eligible AS (
            SELECT session_key
            FROM ranked
            GROUP BY session_key
            HAVING count(*) >= {k}
               AND sum(CASE WHEN rn <= {k} AND event_type='purchase' THEN 1 ELSE 0 END) = 0
        ),
        prefix AS (   -- strictly BEFORE the cutoff
            SELECT r.* FROM ranked r
            JOIN eligible g USING (session_key)
            WHERE r.rn <= {k}
        ),
        suffix_label AS (  -- strictly AFTER the cutoff
            SELECT r.session_key,
                   max(CASE WHEN r.event_type='purchase' THEN 1 ELSE 0 END) AS y
            FROM ranked r
            JOIN eligible g USING (session_key)
            WHERE r.rn > {k}
            GROUP BY r.session_key
        ),
        per_product AS (   -- how often the same product is revisited pre-cutoff
            SELECT session_key, product_id, count(*) AS views
            FROM prefix GROUP BY 1,2
        ),
        revisit AS (
            SELECT session_key,
                   max(views)  AS max_views_same_product,
                   avg(views)  AS mean_views_per_product
            FROM per_product GROUP BY 1
        ),
        gaps AS (
            SELECT session_key,
                   avg(gap_sec)    AS mean_gap_sec,
                   median(gap_sec) AS median_gap_sec,
                   min(gap_sec)    AS min_gap_sec,
                   max(gap_sec)    AS max_gap_sec
            FROM (
                SELECT session_key,
                       date_diff('second',
                           lag(event_time) OVER (PARTITION BY session_key ORDER BY rn),
                           event_time) AS gap_sec
                FROM prefix
            ) WHERE gap_sec IS NOT NULL
            GROUP BY 1
        ),
        agg AS (
            SELECT
                session_key,
                count(DISTINCT product_id)                    AS n_distinct_products,
                count(DISTINCT category_id)                   AS n_distinct_categories,
                count(DISTINCT brand)                         AS n_distinct_brands,
                {k} - count(DISTINCT product_id)              AS n_repeat_views,
                date_diff('second', min(event_time), max(event_time)) AS prefix_duration_sec,
                avg(price)                                    AS price_mean,
                min(price)                                    AS price_min,
                max(price)                                    AS price_max,
                COALESCE(stddev_pop(price), 0.0)              AS price_std,
                max(price) - min(price)                       AS price_range,
                avg(CASE WHEN brand IS NULL THEN 1.0 ELSE 0.0 END)         AS null_brand_ratio,
                avg(CASE WHEN category_code IS NULL THEN 1.0 ELSE 0.0 END) AS null_category_ratio,
                {cart_features},
                arg_max(price, rn)                            AS price_at_cutoff,
                arg_max(category_id, rn)                      AS category_at_cutoff,
                min(event_time)                               AS prefix_start,
                max(event_time)                               AS cutoff_time
            FROM prefix
            GROUP BY session_key
        )
        SELECT
            a.session_key,
            s.user_id,
            s.session_start,
            a.cutoff_time,
            CAST(s.session_start AS DATE)                     AS session_date,

            -- breadth vs depth of browsing
            a.n_distinct_products,
            a.n_distinct_categories,
            a.n_distinct_brands,
            a.n_repeat_views,
            r.max_views_same_product,
            r.mean_views_per_product,
            1.0 - (a.n_distinct_products * 1.0 / {k})         AS repeat_product_ratio,

            -- pace
            a.prefix_duration_sec,
            g.mean_gap_sec,
            g.median_gap_sec,
            g.min_gap_sec,
            g.max_gap_sec,
            CASE WHEN a.prefix_duration_sec > 0
                 THEN {k} * 60.0 / a.prefix_duration_sec ELSE NULL END AS events_per_minute,

            -- price context
            a.price_mean, a.price_min, a.price_max, a.price_std,
            a.price_range, a.price_at_cutoff,

            -- data-quality signals (browsers hit sparse listings more often)
            a.null_brand_ratio,
            a.null_category_ratio,

            -- intent signals
            a.n_cart_events, a.n_remove_events, a.cart_ratio,

            -- calendar context
            hour(s.session_start)                             AS hour_of_day,
            dayofweek(s.session_start)                        AS day_of_week,
            CASE WHEN dayofweek(s.session_start) IN (0,6) THEN 1 ELSE 0 END AS is_weekend,

            -- point-in-time user history (from sessions.parquet, prior sessions only)
            s.user_prior_sessions,
            s.user_prior_purchases,
            CASE WHEN s.user_prior_sessions > 0
                 THEN s.user_prior_purchases * 1.0 / s.user_prior_sessions
                 ELSE NULL END                                AS user_prior_conv_rate,
            s.user_prior_revenue,
            s.hours_since_last_session,
            CASE WHEN s.user_prior_sessions = 0 THEN 1 ELSE 0 END AS is_new_user,

            COALESCE(l.y, 0)                                  AS y
        FROM agg a
        -- explicit ON rather than USING: `agg` and `sessions` both expose
        -- session_key, which makes USING ambiguous from the second join onward
        JOIN      {sessions}   s ON s.session_key = a.session_key
        LEFT JOIN revisit      r ON r.session_key = a.session_key
        LEFT JOIN gaps         g ON g.session_key = a.session_key
        LEFT JOIN suffix_label l ON l.session_key = a.session_key
        )
        TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    n, pos = con.execute(
        f"SELECT count(*), avg(y) FROM read_parquet('{out.as_posix()}')"
    ).fetchone()
    log.info("k=%d rows=%s  positive rate=%.3f%%", k, f"{int(n):,}", float(pos) * 100)
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
