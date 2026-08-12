"""Reconcile the OLTP store against a specific replayed window.

`reconcile_oltp.py` compares against the whole archive, which is only valid
after a full replay. After replaying a window, the OLTP store legitimately
holds a subset, and comparing it to 110M events reports mismatches that are
not real. This compares like with like.

Usage:
    python -m scripts.reconcile_window --from "2019-11-24 00:00:00" --to "2019-11-24 08:00:00"
"""

from __future__ import annotations

import argparse
import sys

import duckdb
import psycopg2

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="lo", required=True)
    ap.add_argument("--to", dest="hi", required=True)
    args = ap.parse_args()

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
    src = f"read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')"

    archive = con.execute(
        f"""
        SELECT count(*),
               count(DISTINCT user_session),
               count(DISTINCT user_id),
               count(DISTINCT product_id),
               count(DISTINCT CASE WHEN event_type='purchase' THEN user_session END),
               count(DISTINCT CASE WHEN event_type='cart'     THEN user_session END)
        FROM {src}
        WHERE user_session IS NOT NULL
          AND event_time >= TIMESTAMP '{args.lo}' AND event_time < TIMESTAMP '{args.hi}'
        """
    ).fetchone()

    # The archive itself contains genuinely-exact duplicate events (~0.1-0.2%
    # of rows, measured by scripts/data_quality_audit.py). A faithful replay
    # therefore REPRODUCES them, and comparing the OLTP duplicate count against
    # zero flags correct behaviour as a defect - which is exactly what an
    # earlier version of this script did. The meaningful comparison is against
    # the archive's own duplicate count over the same window.
    archive_dupes = con.execute(
        f"""
        SELECT count(*) FROM (
            SELECT user_session, event_time, event_type, product_id
            FROM {src}
            WHERE user_session IS NOT NULL
              AND event_time >= TIMESTAMP '{args.lo}' AND event_time < TIMESTAMP '{args.hi}'
            GROUP BY 1,2,3,4 HAVING count(*) > 1
        ) d
        """
    ).fetchone()[0]
    con.close()

    conn = psycopg2.connect(settings.postgres_dsn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT (SELECT count(*) FROM storefront.session_events
                 WHERE event_time >= %s AND event_time < %s),
               (SELECT count(*) FROM storefront.sessions),
               (SELECT count(*) FROM storefront.users),
               (SELECT count(*) FROM storefront.products),
               (SELECT count(DISTINCT session_key) FROM storefront.orders),
               (SELECT count(*) FROM storefront.carts)
        """,
        (args.lo, args.hi),
    )
    oltp = cur.fetchone()

    cur.execute(
        """
        SELECT count(*) FROM (
            SELECT session_key, event_time, event_type, product_id
            FROM storefront.session_events GROUP BY 1,2,3,4 HAVING count(*) > 1
        ) d
        """
    )
    dupes = int(cur.fetchone()[0])
    cur.close()
    conn.close()

    names = ["events", "sessions", "users", "products",
             "sessions that purchased", "sessions that carted"]
    log.info("window %s .. %s", args.lo, args.hi)
    log.info("=" * 72)
    log.info("%-26s %14s %14s %10s", "metric", "archive", "oltp", "delta")
    log.info("-" * 72)
    failures = []
    for name, a, o in zip(names, archive, oltp):
        a, o = int(a), int(o)
        log.info("%-26s %14s %14s %+10d", name, f"{a:,}", f"{o:,}", o - a)
        if o != a:
            failures.append(name)
    log.info("%-26s %14s %14s %+10d", "duplicate event groups",
             f"{int(archive_dupes):,}", f"{dupes:,}", dupes - int(archive_dupes))
    log.info("=" * 72)

    if dupes != int(archive_dupes):
        log.error(
            "duplicate groups differ from the archive (%s vs %s) - the hot window "
            "was probably not cleared before replay; see StorefrontConsumer "
            "docstring and use --truncate",
            f"{dupes:,}", f"{int(archive_dupes):,}",
        )
        failures.append("duplicate event groups")
    else:
        log.info("duplicate groups match the archive exactly - these are the "
                 "source's own duplicates, faithfully reproduced, not double-writes")

    if failures:
        log.error("mismatched: %s", ", ".join(failures))
        sys.exit(1)
    log.info("reconciliation passed: OLTP matches the archive over this window")


if __name__ == "__main__":
    main()
