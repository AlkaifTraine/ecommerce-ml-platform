"""Build the analytical warehouse: a star schema over the event archive.

This is the OLAP half of the platform. Postgres answers "what is in this
customer's cart right now" with a row lookup; it cannot answer "how did
conversion move by category across two months" because it only retains seven
days. That question is what this warehouse exists for.

Modelled as a star rather than a copy of the source:

    dim_date        one row per calendar day
    dim_product     one row per product, with its category and brand
    dim_user        one row per user, with lifetime aggregates
    fct_session     one row per session - the grain analysis actually uses
    agg_daily       daily conversion and volume, the drift monitor's input
    agg_category_daily  category mix over time

Written to a persistent DuckDB file so it can be queried by BI tools, the
drift monitor, and dbt without recomputing from Parquet each time.
"""

from __future__ import annotations

import argparse
import time

import duckdb

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)


def build(settings, out_path=None) -> None:
    out_path = out_path or (settings.data_root / "warehouse.duckdb")
    events = f"read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')"
    sessions_pq = settings.features_dir / "sessions.parquet"
    if not sessions_pq.exists():
        raise SystemExit(
            f"missing {sessions_pq} - run `python -m src.features.sessionize` first"
        )
    sessions = f"read_parquet('{sessions_pq.as_posix()}')"

    con = duckdb.connect(str(out_path))
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")

    quarantine = ", ".join(f"DATE '{d}'" for d in settings.quarantine_dates) or "NULL"

    def step(label: str, sql: str) -> None:
        t = time.time()
        con.execute(sql)
        log.info("  [%6.1fs] %s", time.time() - t, label)

    log.info("building warehouse -> %s", out_path)
    t0 = time.time()

    step(
        "dim_date",
        f"""
        CREATE OR REPLACE TABLE dim_date AS
        SELECT event_date                                   AS date_key,
               dayofweek(event_date)                        AS day_of_week,
               CASE WHEN dayofweek(event_date) IN (0,6) THEN TRUE ELSE FALSE END AS is_weekend,
               weekofyear(event_date)                       AS week_of_year,
               monthname(event_date)                        AS month_name,
               event_date IN ({quarantine})                 AS is_quarantined
        FROM (SELECT DISTINCT event_date FROM {events})
        ORDER BY 1
        """,
    )

    step(
        "dim_product",
        f"""
        CREATE OR REPLACE TABLE dim_product AS
        -- min() rather than any_value() throughout: a product's category or
        -- brand can differ across rows, and any_value() would resolve that
        -- arbitrarily, making the dimension table non-reproducible between
        -- builds. Deterministic beats "probably the same".
        SELECT product_id,
               min(category_id)                             AS category_id,
               min(category_code)                           AS category_code,
               split_part(min(category_code), '.', 1)       AS category_l1,
               min(brand)                                   AS brand,
               median(price)                                AS typical_price,
               min(price)                                   AS min_price,
               max(price)                                   AS max_price,
               count(*)                                     AS n_events,
               sum(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) AS n_purchases
        FROM {events}
        GROUP BY product_id
        """,
    )

    step(
        "dim_user",
        f"""
        CREATE OR REPLACE TABLE dim_user AS
        SELECT user_id,
               min(session_start)                           AS first_seen,
               max(session_end)                             AS last_seen,
               count(*)                                     AS n_sessions,
               sum(bought)                                  AS n_buying_sessions,
               sum(revenue)                                 AS lifetime_revenue,
               sum(bought) * 1.0 / count(*)                 AS conversion_rate
        FROM {sessions}
        GROUP BY user_id
        """,
    )

    # The grain that matters: one row per visit. Everything downstream - drift
    # monitoring, cohort analysis, the training set - is derived from here.
    step(
        "fct_session",
        f"""
        CREATE OR REPLACE TABLE fct_session AS
        SELECT s.session_key,
               s.user_id,
               CAST(s.session_start AS DATE)                AS date_key,
               s.session_start, s.session_end,
               s.n_events, s.duration_sec,
               s.bought, s.carted, s.revenue,
               s.user_prior_sessions, s.user_prior_purchases,
               CAST(s.session_start AS DATE) IN ({quarantine}) AS is_quarantined
        FROM {sessions} s
        """,
    )

    step(
        "agg_daily",
        f"""
        CREATE OR REPLACE TABLE agg_daily AS
        SELECT e.event_date                                              AS date_key,
               count(*)                                                  AS events,
               count(DISTINCT e.user_session)                            AS sessions,
               count(DISTINCT e.user_id)                                 AS users,
               sum(CASE WHEN e.event_type='view'     THEN 1 ELSE 0 END)  AS views,
               sum(CASE WHEN e.event_type='cart'     THEN 1 ELSE 0 END)  AS carts,
               sum(CASE WHEN e.event_type='purchase' THEN 1 ELSE 0 END)  AS purchases,
               sum(CASE WHEN e.event_type='purchase' THEN e.price ELSE 0 END) AS revenue,
               avg(e.price)                                              AS avg_price,
               median(e.price)                                           AS median_price,
               sum(CASE WHEN e.event_type='purchase' THEN 1 ELSE 0 END) * 1.0
                   / count(DISTINCT e.user_session)                      AS purchases_per_session,
               -- the ratio that exposed the 2019-11-14..17 collection outage
               sum(CASE WHEN e.event_type='purchase' THEN 1 ELSE 0 END) * 1.0
                   / NULLIF(sum(CASE WHEN e.event_type='cart' THEN 1 ELSE 0 END), 0) AS buy_per_cart
        FROM {events} e
        GROUP BY 1 ORDER BY 1
        """,
    )

    step(
        "agg_category_daily",
        f"""
        CREATE OR REPLACE TABLE agg_category_daily AS
        SELECT event_date                                     AS date_key,
               COALESCE(split_part(category_code, '.', 1), '(unknown)') AS category_l1,
               count(*)                                       AS events,
               sum(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) AS purchases,
               avg(price)                                     AS avg_price
        FROM {events}
        GROUP BY 1, 2 ORDER BY 1, 2
        """,
    )

    log.info("=" * 62)
    for tbl in ("dim_date", "dim_product", "dim_user", "fct_session",
                "agg_daily", "agg_category_daily"):
        n = con.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
        log.info("  %-22s %14s rows", tbl, f"{n:,}")
    log.info("=" * 62)

    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    log.info("warehouse built in %.1fs, %.0f MB -> %s", time.time() - t0, size_mb, out_path)
    con.close()


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from pathlib import Path

    build(settings, Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
