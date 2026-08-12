"""Convert raw clickstream CSV into date-partitioned Parquet.

Design notes
------------
* DuckDB streams the conversion so peak RAM stays near `duckdb_memory_limit`
  rather than scaling with the 9GB November file.
* `event_time` in the REES46 export carries a trailing " UTC" (e.g.
  "2019-10-01 00:00:00 UTC") which is not a parseable timestamp literal. We
  sniff the header/first row and strip it when present instead of assuming.
* Output is partitioned by `event_date`. Partitioning is what later lets the
  replayer and the training jobs read a bounded time window cheaply, which is
  the whole basis of the "no peeking into the future" guarantee.
* The source CSV is optionally deleted after a verified conversion to keep
  disk usage inside the ~41GB budget on D:.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)

EXPECTED_COLUMNS = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
]


def connect(settings) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def sniff(csv_path: Path) -> tuple[list[str], bool]:
    """Return (header columns, whether event_time has a ' UTC' suffix)."""
    with csv_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        first = fh.readline()
    has_utc = " UTC" in first.split(",")[0]
    return header, has_utc


def convert(csv_path: Path, out_dir: Path, settings, delete_source: bool = False) -> int:
    header, has_utc = sniff(csv_path)
    missing = [c for c in EXPECTED_COLUMNS if c not in header]
    if missing:
        raise SystemExit(
            f"{csv_path.name}: missing expected columns {missing}. Found: {header}"
        )
    log.info("%s: header OK, event_time has ' UTC' suffix = %s", csv_path.name, has_utc)

    ts_expr = (
        "strptime(replace(event_time, ' UTC', ''), '%Y-%m-%d %H:%M:%S')"
        if has_utc
        else "CAST(event_time AS TIMESTAMP)"
    )

    con = connect(settings)
    out_dir.mkdir(parents=True, exist_ok=True)

    # read_csv with explicit types: category_id is a ~19-digit integer, and
    # user_session is a UUID string that must not be coerced to something else.
    src = f"""
        read_csv(
            '{csv_path.as_posix()}',
            header=true,
            columns={{
                'event_time':    'VARCHAR',
                'event_type':    'VARCHAR',
                'product_id':    'BIGINT',
                'category_id':   'BIGINT',
                'category_code': 'VARCHAR',
                'brand':         'VARCHAR',
                'price':         'DOUBLE',
                'user_id':       'BIGINT',
                'user_session':  'VARCHAR'
            }}
        )
    """

    log.info("converting %s -> %s (partitioned by event_date)", csv_path.name, out_dir)
    con.execute(
        f"""
        COPY (
            SELECT
                {ts_expr}                     AS event_time,
                CAST({ts_expr} AS DATE)       AS event_date,
                event_type,
                product_id,
                category_id,
                category_code,
                brand,
                price,
                user_id,
                user_session
            FROM {src}
        )
        TO '{out_dir.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (event_date),
         COMPRESSION ZSTD, OVERWRITE_OR_IGNORE true,
         FILENAME_PATTERN '{csv_path.stem}_{{i}}')
        """
    )

    # Count ONLY the partitions this file produced, via its FILENAME_PATTERN
    # stem. Counting every parquet under out_dir would include earlier files
    # and could "verify" a conversion that actually wrote nothing - not a
    # sound basis for deleting a multi-gigabyte source.
    written = con.execute(
        f"SELECT count(*) FROM read_parquet('{out_dir.as_posix()}/**/{csv_path.stem}_*.parquet')"
    ).fetchone()[0]
    total = con.execute(
        f"SELECT count(*) FROM read_parquet('{out_dir.as_posix()}/**/*.parquet')"
    ).fetchone()[0]
    log.info("%s: wrote %s rows (parquet total now %s)",
             csv_path.name, f"{written:,}", f"{total:,}")

    if written == 0:
        raise SystemExit(
            f"{csv_path.name}: conversion produced 0 rows - refusing to continue. "
            "Source file left untouched."
        )

    if delete_source:
        size_gb = csv_path.stat().st_size / 1e9
        csv_path.unlink()
        log.info("deleted source %s, reclaimed %.2f GB", csv_path.name, size_gb)

    con.close()
    return written


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default="*.csv", help="glob within data/raw")
    ap.add_argument(
        "--delete-source",
        action="store_true",
        help="remove each CSV after a verified conversion (disk is tight)",
    )
    args = ap.parse_args()

    csvs = sorted(settings.raw_dir.glob(args.pattern))
    if not csvs:
        raise SystemExit(f"no CSV matching {args.pattern!r} in {settings.raw_dir}")

    log.info("found %d CSV file(s) to convert", len(csvs))
    for csv_path in csvs:
        log.info("--- %s (%.2f GB) ---", csv_path.name, csv_path.stat().st_size / 1e9)
        convert(csv_path, settings.events_dir, settings, delete_source=args.delete_source)

    log.info("conversion complete -> %s", settings.events_dir)


if __name__ == "__main__":
    main()
