"""Characterise the single NULL user_session found by the dbt test."""

from __future__ import annotations

import duckdb

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
    src = f"read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')"

    n = con.execute(
        f"SELECT count(*) FROM {src} WHERE user_session IS NULL"
    ).fetchone()[0]
    log.info("raw events with NULL user_session: %s", f"{n:,}")

    if n:
        log.info("the offending rows:")
        for r in con.execute(
            f"""
            SELECT event_time, event_type, product_id, price, user_id
            FROM {src} WHERE user_session IS NULL ORDER BY event_time LIMIT 10
            """
        ).fetchall():
            log.info("   %s  %-9s product=%-10s price=%-8s user=%s",
                     r[0], r[1], r[2], r[3], r[4])

    # Did it reach the training table?
    train = settings.features_dir / "train_k5.parquet"
    if train.exists():
        m = con.execute(
            f"SELECT count(*) FROM read_parquet('{train.as_posix()}') "
            "WHERE session_key IS NULL"
        ).fetchone()[0]
        log.info("rows in train_k5 with NULL session_key: %s", m)

    sess = settings.features_dir / "sessions.parquet"
    if sess.exists():
        s = con.execute(
            f"SELECT count(*), sum(n_events), sum(bought) "
            f"FROM read_parquet('{sess.as_posix()}') WHERE session_key IS NULL"
        ).fetchone()
        log.info("session index rows with NULL key: %s (events=%s, bought=%s)",
                 s[0], s[1], s[2])
    con.close()


if __name__ == "__main__":
    main()
