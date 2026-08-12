"""GATE 1: does the real data support the task at all?

Three questions decide whether the rest of the project is worth building:

1. How long are sessions? The truncation-point task needs sessions that reach
   k events. If the median session is 3 events, k=10 discards almost everything.
2. Among ELIGIBLE sessions (long enough, no purchase before the cutoff), what
   is the positive rate? The headline base rate across all sessions is not the
   number the model actually faces.
3. Is the Black Friday regime change visible in the raw daily series? The whole
   continuous-retraining story depends on it being real.

This runs on the raw archive - no feature build required - so it fails fast.
"""

from __future__ import annotations

import argparse
import json

import duckdb

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ks", default="5,10,20")
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
    src = f"read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')"

    log.info("building per-session summary over the full archive ...")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE sess AS
        SELECT user_session,
               count(*)                                                  AS n_events,
               max(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END)    AS bought,
               min(CASE WHEN event_type='purchase' THEN rn END)          AS first_purchase_rn
        FROM (
            SELECT user_session, event_type,
                   row_number() OVER (PARTITION BY user_session
                                      ORDER BY event_time, product_id) AS rn
            FROM {src}
        )
        GROUP BY user_session
        """
    )

    # ---- 1. session length distribution ----------------------------------
    q = con.execute(
        """
        SELECT count(*), avg(n_events), median(n_events),
               quantile_cont(n_events, 0.25), quantile_cont(n_events, 0.75),
               quantile_cont(n_events, 0.90), quantile_cont(n_events, 0.99),
               max(n_events)
        FROM sess
        """
    ).fetchone()
    log.info("=" * 72)
    log.info("SESSION LENGTH")
    log.info("  sessions        %s", f"{int(q[0]):,}")
    log.info("  mean            %.2f events", float(q[1]))
    log.info("  p25 / median    %.0f / %.0f", float(q[3]), float(q[2]))
    log.info("  p75 / p90 / p99 %.0f / %.0f / %.0f", float(q[4]), float(q[5]), float(q[6]))
    log.info("  max             %s", f"{int(q[7]):,}")

    # ---- 2. eligibility and positive rate per k --------------------------
    log.info("=" * 72)
    log.info("ELIGIBILITY BY TRUNCATION POINT")
    log.info("  %-4s %14s %9s %14s %12s", "k", "eligible", "% of all", "positives", "pos rate")
    results = {}
    for k in ks:
        r = con.execute(
            f"""
            SELECT count(*),
                   sum(CASE WHEN bought = 1 AND first_purchase_rn > {k} THEN 1 ELSE 0 END)
            FROM sess
            WHERE n_events >= {k}
              AND (first_purchase_rn IS NULL OR first_purchase_rn > {k})
            """
        ).fetchone()
        n_elig, n_pos = int(r[0]), int(r[1] or 0)
        rate = n_pos / n_elig if n_elig else 0.0
        share = 100.0 * n_elig / int(q[0])
        results[k] = {"eligible": n_elig, "positives": n_pos, "pos_rate": rate}
        log.info("  %-4d %14s %8.1f%% %14s %11.3f%%",
                 k, f"{n_elig:,}", share, f"{n_pos:,}", rate * 100)

    # ---- 3. the Black Friday signature -----------------------------------
    log.info("=" * 72)
    log.info("DAILY SERIES AROUND BLACK FRIDAY (2019-11-29)")
    rows = con.execute(
        f"""
        SELECT event_date,
               count(*)                                               AS events,
               count(DISTINCT user_session)                           AS sessions,
               sum(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) AS purchases,
               avg(price)                                             AS avg_price
        FROM {src}
        WHERE event_date BETWEEN DATE '2019-11-22' AND DATE '2019-11-30'
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    log.info("  %-12s %12s %12s %12s %11s %10s", "date", "events", "sessions", "purchases",
             "conv/sess", "avg price")
    for d, ev, ss, pu, ap_ in rows:
        marker = "  <-- BLACK FRIDAY" if str(d) == "2019-11-29" else ""
        log.info("  %-12s %12s %12s %12s %10.3f%% %10.2f%s",
                 str(d), f"{ev:,}", f"{ss:,}", f"{pu:,}", 100.0 * pu / ss, float(ap_), marker)

    baseline = con.execute(
        f"""
        SELECT avg(price),
               sum(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) * 1.0
                   / count(DISTINCT user_session)
        FROM {src} WHERE event_date BETWEEN DATE '2019-11-01' AND DATE '2019-11-20'
        """
    ).fetchone()
    log.info("  %-12s %12s %12s %12s %10.3f%% %10.2f",
             "Nov 1-20 avg", "-", "-", "-", float(baseline[1]) * 100, float(baseline[0]))
    log.info("=" * 72)

    out = settings.artifacts_dir / "gate1_signal_check.json"
    out.write_text(json.dumps({
        "session_length": {
            "sessions": int(q[0]), "mean": float(q[1]), "median": float(q[2]),
            "p90": float(q[5]), "p99": float(q[6]), "max": int(q[7]),
        },
        "eligibility": results,
        "black_friday_window": [
            {"date": str(d), "events": int(ev), "sessions": int(ss),
             "purchases": int(pu), "avg_price": float(ap_)}
            for d, ev, ss, pu, ap_ in rows
        ],
    }, indent=2), encoding="utf-8")
    log.info("written -> %s", out)
    con.close()


if __name__ == "__main__":
    main()
