"""Reconcile OLTP state against the source archive.

After a replay, the Postgres store should agree with the Parquet archive on the
things that are not subject to retention: how many sessions existed, how many
of them carted, how many ordered. `session_events` is deliberately excluded
from equality checks because it is a rolling hot window by design.

Run after a replay:
    python -m scripts.reconcile_oltp
"""

from __future__ import annotations

import sys

import duckdb
import psycopg2

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)


def source_counts(settings) -> dict[str, int]:
    con = duckdb.connect()
    try:
        src = f"read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')"
        row = con.execute(
            f"""
            SELECT
                count(DISTINCT user_session)                                        AS sessions,
                count(DISTINCT CASE WHEN event_type='cart'     THEN user_session END) AS cart_sessions,
                count(DISTINCT CASE WHEN event_type='purchase' THEN user_session END) AS purchase_sessions,
                count(DISTINCT user_id)                                             AS users,
                count(DISTINCT product_id)                                          AS products,
                count(*)                                                            AS events,
                count(*) FILTER (WHERE event_type='purchase')                       AS purchase_events
            FROM {src}
            """
        ).fetchone()
    finally:
        con.close()
    keys = ["sessions", "cart_sessions", "purchase_sessions", "users",
            "products", "events", "purchase_events"]
    return dict(zip(keys, (int(x) for x in row)))


def oltp_counts(settings) -> dict[str, int]:
    conn = psycopg2.connect(settings.postgres_dsn)
    try:
        cur = conn.cursor()
        out: dict[str, int] = {}
        for name, sql in [
            ("sessions", "SELECT count(*) FROM storefront.sessions"),
            ("cart_sessions", "SELECT count(*) FROM storefront.carts"),
            ("purchase_sessions", "SELECT count(DISTINCT session_key) FROM storefront.orders"),
            ("users", "SELECT count(*) FROM storefront.users"),
            ("products", "SELECT count(*) FROM storefront.products"),
            ("hot_events", "SELECT count(*) FROM storefront.session_events"),
            ("order_rows", "SELECT count(*) FROM storefront.orders"),
        ]:
            cur.execute(sql)
            out[name] = int(cur.fetchone()[0])
        cur.close()
    finally:
        conn.close()
    return out


def main() -> None:
    settings = get_settings()
    log.info("reading source archive at %s", settings.events_dir)
    src = source_counts(settings)
    log.info("reading OLTP at %s:%s", settings.postgres_host, settings.postgres_port)
    db = oltp_counts(settings)

    # These must match exactly - they are not affected by the retention window.
    exact = ["sessions", "cart_sessions", "purchase_sessions", "users", "products"]

    log.info("=" * 66)
    log.info("%-22s %12s %12s %10s", "metric", "source", "oltp", "delta")
    log.info("-" * 66)
    failures = []
    for k in exact:
        s, d = src[k], db[k]
        delta = d - s
        log.info("%-22s %12s %12s %10s", k, f"{s:,}", f"{d:,}", f"{delta:+,}")
        if delta != 0:
            failures.append((k, s, d))

    log.info("-" * 66)
    log.info("%-22s %12s %12s %10s", "events (archive)", f"{src['events']:,}", "-", "-")
    log.info("%-22s %12s %12s %10s", "events (hot window)", "-", f"{db['hot_events']:,}",
             f"retained {100*db['hot_events']/max(src['events'],1):.1f}%")
    log.info("=" * 66)

    if failures:
        for k, s, d in failures:
            log.error("MISMATCH %s: source=%s oltp=%s", k, f"{s:,}", f"{d:,}")
        sys.exit(1)
    log.info("reconciliation passed: OLTP agrees with the archive on all durable counts")


if __name__ == "__main__":
    main()
