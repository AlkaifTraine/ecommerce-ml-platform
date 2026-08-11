"""Profile the converted Parquet and emit an honest data-quality report.

This runs before any modelling. Its job is to replace assumptions with measured
facts: the real row count, the real date coverage, the real purchase base rate.
Everything downstream quotes these numbers rather than estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)


def _con(settings) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
    return con


def profile(events_glob: str, settings) -> dict:
    con = _con(settings)
    src = f"read_parquet('{events_glob}')"
    report: dict = {}

    log.info("counting rows and date coverage ...")
    row = con.execute(
        f"""
        SELECT count(*)                        AS n_rows,
               min(event_time)                 AS ts_min,
               max(event_time)                 AS ts_max,
               count(DISTINCT event_date)      AS n_days,
               count(DISTINCT user_id)         AS n_users,
               count(DISTINCT user_session)    AS n_sessions,
               count(DISTINCT product_id)      AS n_products,
               count(DISTINCT category_id)     AS n_categories,
               count(DISTINCT brand)           AS n_brands
        FROM {src}
        """
    ).fetchone()
    report["overview"] = {
        "n_rows": int(row[0]),
        "ts_min": str(row[1]),
        "ts_max": str(row[2]),
        "n_days": int(row[3]),
        "n_users": int(row[4]),
        "n_sessions": int(row[5]),
        "n_products": int(row[6]),
        "n_categories": int(row[7]),
        "n_brands": int(row[8]),
    }

    log.info("event_type distribution ...")
    report["event_types"] = [
        {"event_type": r[0], "n": int(r[1]), "pct": round(float(r[2]), 4)}
        for r in con.execute(
            f"""
            SELECT event_type, count(*) AS n,
                   100.0 * count(*) / sum(count(*)) OVER () AS pct
            FROM {src} GROUP BY 1 ORDER BY n DESC
            """
        ).fetchall()
    ]

    log.info("null rates ...")
    report["null_rates_pct"] = {
        col: round(
            float(
                con.execute(
                    f"SELECT 100.0*sum(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END)/count(*) FROM {src}"
                ).fetchone()[0]
            ),
            4,
        )
        for col in (
            "event_time",
            "event_type",
            "product_id",
            "category_id",
            "category_code",
            "brand",
            "price",
            "user_id",
            "user_session",
        )
    }

    log.info("daily volumes ...")
    report["daily"] = [
        {
            "date": str(r[0]),
            "events": int(r[1]),
            "sessions": int(r[2]),
            "purchases": int(r[3]),
        }
        for r in con.execute(
            f"""
            SELECT event_date,
                   count(*),
                   count(DISTINCT user_session),
                   sum(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END)
            FROM {src} GROUP BY 1 ORDER BY 1
            """
        ).fetchall()
    ]

    log.info("price distribution ...")
    p = con.execute(
        f"""
        SELECT min(price), quantile_cont(price,0.25), median(price),
               quantile_cont(price,0.75), quantile_cont(price,0.99), max(price), avg(price)
        FROM {src} WHERE price IS NOT NULL
        """
    ).fetchone()
    report["price"] = dict(
        zip(["min", "p25", "median", "p75", "p99", "max", "mean"], [float(x) for x in p])
    )

    # The single most important number: what fraction of sessions ever buy.
    log.info("session-level purchase base rate ...")
    b = con.execute(
        f"""
        WITH s AS (
            SELECT user_session,
                   max(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) AS bought,
                   max(CASE WHEN event_type='cart'     THEN 1 ELSE 0 END) AS carted,
                   count(*) AS n_events
            FROM {src} GROUP BY 1
        )
        SELECT count(*), avg(bought), avg(carted),
               avg(CASE WHEN carted=1 THEN bought END),
               median(n_events), quantile_cont(n_events,0.9)
        FROM s
        """
    ).fetchone()
    report["base_rates"] = {
        "n_sessions": int(b[0]),
        "session_purchase_rate": round(float(b[1]), 5),
        "session_cart_rate": round(float(b[2]), 5),
        "cart_to_purchase_rate": round(float(b[3]), 5) if b[3] is not None else None,
        "median_events_per_session": float(b[4]),
        "p90_events_per_session": float(b[5]),
    }

    con.close()
    return report


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="where to write the JSON report")
    args = ap.parse_args()

    glob = f"{settings.events_dir.as_posix()}/**/*.parquet"
    report = profile(glob, settings)

    out = Path(args.out) if args.out else settings.artifacts_dir / "data_profile.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    ov = report["overview"]
    br = report["base_rates"]
    log.info("=" * 68)
    log.info("ROWS              %s", f"{ov['n_rows']:,}")
    log.info("DATE RANGE        %s  ->  %s  (%d days)", ov["ts_min"], ov["ts_max"], ov["n_days"])
    log.info("USERS / SESSIONS  %s / %s", f"{ov['n_users']:,}", f"{ov['n_sessions']:,}")
    log.info("PRODUCTS          %s", f"{ov['n_products']:,}")
    log.info("PURCHASE RATE     %.3f%% of sessions", br["session_purchase_rate"] * 100)
    log.info("CART RATE         %.3f%% of sessions", br["session_cart_rate"] * 100)
    log.info("CART -> PURCHASE  %.2f%%", (br["cart_to_purchase_rate"] or 0) * 100)
    log.info("=" * 68)
    log.info("report written to %s", out)


if __name__ == "__main__":
    main()
