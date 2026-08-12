"""Carve a small, committable test fixture out of the REAL dataset.

Why this exists
---------------
The test suite needs data it can load in seconds, and CI cannot download 14GB
per run. The obvious shortcut is to generate fake sessions - but generated data
encodes whatever assumptions the generator's author had, which is exactly what
tests are supposed to catch. An earlier synthetic generator here gave browsers
and buyers barely-overlapping click gaps, so a single feature separated them
perfectly and the fixture could no longer detect a bug in anything else.

So the fixture is a real slice instead. No synthetic data exists in this
project.

Sampling rules
--------------
* Sample by SESSION, never by event. Splitting a session mid-way would corrupt
  every feature that depends on within-session ordering, and would silently
  break the truncation-point logic the tests exist to verify.
* Selection is a deterministic hash of `user_session`, so the same sessions are
  chosen on every machine and the fixture is reproducible.
* A date range is fixed explicitly rather than "the first N days", so the
  fixture does not change when new source files are added.

Usage:
    python -m scripts.make_test_fixture --from 2019-11-05 --to 2019-11-08 --rate 100
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import duckdb

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "events"


def build(settings, date_from: str, date_to: str, rate: int, out_dir: Path) -> dict:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
    src = f"read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1-in-`rate` sessions, chosen by hash so the pick is stable everywhere.
    selector = f"(hash(user_session) % {rate}) = 0"
    window = (
        f"event_date >= DATE '{date_from}' AND event_date <= DATE '{date_to}'"
    )

    log.info("carving fixture: %s..%s, 1-in-%d sessions", date_from, date_to, rate)
    con.execute(
        f"""
        COPY (
            SELECT event_time, event_date, event_type, product_id, category_id,
                   category_code, brand, price, user_id, user_session
            FROM {src}
            WHERE {window} AND {selector}
            ORDER BY event_time, user_session
        )
        TO '{out_dir.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (event_date),
         COMPRESSION ZSTD, OVERWRITE_OR_IGNORE true,
         FILENAME_PATTERN 'fixture_{{i}}')
        """
    )

    stats = con.execute(
        f"""
        WITH s AS (
            SELECT user_session,
                   count(*) AS n,
                   max(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) AS bought,
                   max(CASE WHEN event_type='cart'     THEN 1 ELSE 0 END) AS carted
            FROM read_parquet('{out_dir.as_posix()}/**/*.parquet')
            GROUP BY user_session
        )
        SELECT (SELECT count(*) FROM read_parquet('{out_dir.as_posix()}/**/*.parquet')),
               count(*), avg(bought), avg(carted),
               median(n), max(n),
               sum(CASE WHEN n >= 5  THEN 1 ELSE 0 END),
               sum(CASE WHEN n >= 10 THEN 1 ELSE 0 END)
        FROM s
        """
    ).fetchone()
    con.close()

    size_mb = sum(f.stat().st_size for f in out_dir.rglob("*.parquet")) / 1e6
    report = {
        "events": int(stats[0]),
        "sessions": int(stats[1]),
        "purchase_rate": float(stats[2]),
        "cart_rate": float(stats[3]),
        "median_events": float(stats[4]),
        "max_events": int(stats[5]),
        "sessions_ge_5": int(stats[6]),
        "sessions_ge_10": int(stats[7]),
        "size_mb": size_mb,
    }

    log.info("=" * 60)
    log.info("events           %s", f"{report['events']:,}")
    log.info("sessions         %s", f"{report['sessions']:,}")
    log.info("purchase rate    %.3f%%", report["purchase_rate"] * 100)
    log.info("cart rate        %.3f%%", report["cart_rate"] * 100)
    log.info("events/session   median %.0f, max %d", report["median_events"], report["max_events"])
    log.info("eligible k=5     %s sessions", f"{report['sessions_ge_5']:,}")
    log.info("eligible k=10    %s sessions", f"{report['sessions_ge_10']:,}")
    log.info("on disk          %.1f MB", report["size_mb"])
    log.info("=" * 60)

    if report["sessions_ge_5"] < 500:
        log.warning("only %d sessions reach k=5 - consider a lower --rate for a richer fixture",
                    report["sessions_ge_5"])
    if report["size_mb"] > 40:
        log.warning("%.1f MB is large for a committed fixture - consider a higher --rate",
                    report["size_mb"])
    return report


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="date_from", default="2019-11-05")
    ap.add_argument("--to", dest="date_to", default="2019-11-08")
    ap.add_argument("--rate", type=int, default=100, help="keep 1 session in N")
    ap.add_argument("--out", default=str(FIXTURE_DIR))
    args = ap.parse_args()

    build(settings, args.date_from, args.date_to, args.rate, Path(args.out))
    log.info("fixture written to %s", args.out)


if __name__ == "__main__":
    main()
