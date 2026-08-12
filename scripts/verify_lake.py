"""Verify the CDC lake matches the OLTP source it was drained from."""

from __future__ import annotations

import sys

import duckdb
import psycopg2

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    lake = settings.parquet_dir / "lake"
    if not lake.exists():
        raise SystemExit(f"no lake at {lake} - run the warehouse_refresh DAG")

    con = duckdb.connect()
    conn = psycopg2.connect(settings.postgres_dsn)
    cur = conn.cursor()

    ok = True
    for table, pk in (("session_events", "event_id"), ("orders", "order_id"),
                      ("predictions", "prediction_id")):
        d = lake / table
        if not d.exists():
            log.warning("%-16s no lake directory", table)
            continue

        rows, distinct, lo, hi = con.execute(
            f"""
            SELECT count(*), count(DISTINCT {pk}), min({pk}), max({pk})
            FROM read_parquet('{d.as_posix()}/*.parquet')
            """
        ).fetchone()

        cur.execute(f"SELECT last_event_id FROM storefront.cdc_watermark WHERE table_name = %s",
                    (table,))
        wm = int(cur.fetchone()[0])
        cur.execute(f"SELECT count(*) FROM storefront.{table} WHERE {pk} <= %s", (wm,))
        src = int(cur.fetchone()[0])

        dupes = int(rows) - int(distinct)
        status = "OK"
        if dupes:
            status = f"FAIL {dupes} duplicate ids"
            ok = False
        elif int(rows) != src:
            status = f"FAIL lake {rows} vs source {src} at watermark"
            ok = False

        log.info("%-16s lake=%9s distinct=%9s ids=%s..%s watermark=%-9s %s",
                 table, f"{int(rows):,}", f"{int(distinct):,}", lo, hi, wm, status)

    cur.close()
    conn.close()
    con.close()

    if not ok:
        log.error("lake does not match its source")
        sys.exit(1)
    log.info("lake verified: no duplicate ids, row counts match the source at each watermark")


if __name__ == "__main__":
    main()
