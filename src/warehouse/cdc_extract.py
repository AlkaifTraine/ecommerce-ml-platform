"""Incremental extraction from the OLTP store into the Parquet lake.

Why this exists
---------------
Postgres holds only a rolling 7-day hot window (see the storefront consumer).
Everything older is trimmed. If history is going to be queryable at all, it has
to be drained out before the retention job deletes it - and it has to be
drained without running analytical scans against the transactional database.

The monotonic-id trap
---------------------
The obvious implementation is "select rows with event_id > watermark". That is
subtly wrong. `BIGSERIAL` values are handed out when a statement executes, not
when its transaction commits, so a row with id 100 can become visible *after*
a row with id 101 has already been extracted. Advance the watermark past 101
and row 100 is lost forever - silently, and only under concurrency, which is
why it usually survives testing and fails in production.

Two defences here:

1. **Safety lag.** Never extract closer than `--lag-seconds` to the newest row.
   Anything still in flight has had time to commit before we look at it.
2. **Watermark advanced only to a verified contiguous point.** The watermark
   moves to the highest id actually read in this batch, inside the same
   transaction as the read, so a crash mid-extract replays rather than skips.

The genuinely correct fix is logical replication (Debezium reading the WAL),
which observes commit order rather than id order. That is the production
answer and is noted in the README as the intended cloud path; this file is the
pragmatic local equivalent, with its limitation stated rather than hidden.
"""

from __future__ import annotations

import argparse
from datetime import timedelta

import duckdb
import psycopg2
import pyarrow as pa

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)

# table -> (primary key column, timestamp column used for lake partitioning)
SOURCES = {
    "session_events": ("event_id", "event_time"),
    "orders": ("order_id", "ordered_at"),
    "predictions": ("prediction_id", "scored_at_data_time"),
}


def _lake_dir(settings, table: str):
    return settings.parquet_dir / "lake" / table


def extract(settings, table: str, batch_size: int, lag_seconds: int) -> int:
    if table not in SOURCES:
        raise SystemExit(f"unknown table {table!r}; expected one of {list(SOURCES)}")
    pk, ts_col = SOURCES[table]

    conn = psycopg2.connect(settings.postgres_dsn)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        cur.execute(
            "SELECT last_event_id FROM storefront.cdc_watermark WHERE table_name = %s FOR UPDATE",
            (table,),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO storefront.cdc_watermark (table_name) VALUES (%s)", (table,)
            )
            watermark = 0
        else:
            watermark = int(row[0])

        # Ceiling: ignore anything newer than the safety lag, measured against
        # the DATA clock rather than wall time so it behaves the same at 1x and
        # at 40,000x replay speed.
        cur.execute("SELECT current_data_time FROM storefront.replay_clock WHERE id = 1")
        clock = cur.fetchone()[0]
        ceiling = clock - timedelta(seconds=lag_seconds)

        # Detect a source reset. If the table's highest id is BELOW the stored
        # watermark, the source was truncated and its identity restarted - the
        # watermark now points past the end of a table that begins again at 1,
        # so every future extract silently returns nothing.
        #
        # This is not hypothetical: a scheduled run drained the OLTP store,
        # the store was then truncated and replayed, and the lake ended up
        # holding two overlapping id ranges - stale rows and fresh rows with
        # the same ids, indistinguishable afterwards.
        cur.execute(f"SELECT max({pk}) FROM storefront.{table}")
        table_max = cur.fetchone()[0]
        if table_max is not None and int(table_max) < watermark:
            conn.rollback()
            raise SystemExit(
                f"{table}: source reset detected - max({pk})={table_max} is below the "
                f"watermark {watermark}. The table was truncated after a previous "
                f"extract, so the lake holds rows whose ids now refer to different "
                f"events.\n"
                f"Resolve deliberately: delete data/parquet/lake/{table}, reset the "
                f"watermark to 0, and re-extract."
            )

        cur.execute(
            f"""
            SELECT max({pk}) FROM storefront.{table}
            WHERE {pk} > %s AND {ts_col} <= %s
            """,
            (watermark, ceiling),
        )
        max_id = cur.fetchone()[0]
        if max_id is None:
            log.info("%s: nothing new below the safety ceiling %s", table, ceiling)
            conn.rollback()
            return 0

        # The watermark may only advance to a point where EVERY row at or below
        # it has already been extracted. Ids are not ordered by event_time here
        # - the consumer inserts batched by session, not chronologically - so a
        # low id can carry a timestamp past the ceiling. Advancing to
        # max(id where ts <= ceiling) then skips those rows permanently.
        #
        # Measured: 734 of 421,627 session_events silently lost that way, found
        # by scripts/verify_lake.py comparing lake rows against source rows at
        # the watermark. Nothing errored.
        #
        # So the ceiling for this batch is one below the FIRST id whose
        # timestamp exceeds the cutoff - everything under that is safe to take.
        cur.execute(
            f"SELECT min({pk}) FROM storefront.{table} WHERE {pk} > %s AND {ts_col} > %s",
            (watermark, ceiling),
        )
        first_excluded = cur.fetchone()[0]

        upper = min(int(max_id), watermark + batch_size)
        if first_excluded is not None:
            upper = min(upper, int(first_excluded) - 1)
        if upper <= watermark:
            log.info(
                "%s: next row (id %s) is above the safety ceiling %s - waiting",
                table, first_excluded, ceiling,
            )
            conn.rollback()
            return 0

        cur.execute(
            f"""
            SELECT count(*) FROM storefront.{table}
            WHERE {pk} > %s AND {pk} <= %s AND {ts_col} <= %s
            """,
            (watermark, upper, ceiling),
        )
        n = int(cur.fetchone()[0])
        if n == 0:
            conn.rollback()
            return 0

        out = _lake_dir(settings, table)
        out.mkdir(parents=True, exist_ok=True)
        target = out / f"{table}_{watermark + 1}_{upper}.parquet"

        # Read through DuckDB's postgres scanner if available; otherwise pull
        # via psycopg2 and hand the rows to DuckDB. Either way the analytical
        # write happens outside the transactional database.
        rows = _fetch(cur, table, pk, ts_col, watermark, upper, ceiling)
        _write_parquet(settings, rows, target)

        cur.execute(
            """
            UPDATE storefront.cdc_watermark
               SET last_event_id = %s, last_data_time = %s, extracted_at = now()
             WHERE table_name = %s
            """,
            (upper, ceiling, table),
        )
        conn.commit()
        log.info("%s: extracted %s rows (ids %s..%s) -> %s",
                 table, f"{n:,}", f"{watermark + 1:,}", f"{upper:,}", target.name)
        return n

    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def _fetch(cur, table, pk, ts_col, lo, hi, ceiling):
    cur.execute(
        f"""
        SELECT * FROM storefront.{table}
        WHERE {pk} > %s AND {pk} <= %s AND {ts_col} <= %s
        ORDER BY {pk}
        """,
        (lo, hi, ceiling),
    )
    return cur.description, cur.fetchall()


def _write_parquet(settings, fetched, target) -> None:
    description, rows = fetched
    cols = [d[0] for d in description]
    con = duckdb.connect()
    try:
        con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
        arrays = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
        tbl = pa.table(arrays)
        con.register("batch_arrow", tbl)
        con.execute(
            f"COPY (SELECT * FROM batch_arrow) TO '{target.as_posix()}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        con.close()


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="all", help="one of session_events, orders, predictions, or all")
    ap.add_argument("--batch-size", type=int, default=500_000)
    ap.add_argument("--lag-seconds", type=int, default=300,
                    help="never extract closer than this to the data clock")
    ap.add_argument("--loop", action="store_true", help="keep draining until nothing is left")
    args = ap.parse_args()

    tables = list(SOURCES) if args.table == "all" else [args.table]
    total = 0
    while True:
        moved = 0
        for t in tables:
            moved += extract(settings, t, args.batch_size, args.lag_seconds)
        total += moved
        if not args.loop or moved == 0:
            break
    log.info("extracted %s rows in total", f"{total:,}")


if __name__ == "__main__":
    main()
