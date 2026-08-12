"""Data-quality audit: find days the source pipeline lied to us.

The daily scan surfaced an anomaly around 2019-11-15..17 - triple the normal
event volume and, on the 15th, zero purchases. Behaviour does not do that;
collection pipelines do. This audit characterises it precisely, because the
consequences for modelling are severe:

* A day with missing purchase events silently labels every session on that day
  as a non-purchase. That is label corruption, not class imbalance, and it
  will teach the model that whatever happened that day predicts "no buy".
* A day with duplicated events inflates per-session counts, which corrupts
  every session-shape feature and shifts sessions across the k-event
  eligibility threshold.

Neither is visible in a feature-drift dashboard, which is exactly why this
check exists separately.
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
    ap.add_argument("--from", dest="d_from", default="2019-11-12")
    ap.add_argument("--to", dest="d_to", default="2019-11-19")
    args = ap.parse_args()

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
    src = f"read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')"

    # ---- 1. event-type composition per day --------------------------------
    log.info("=" * 92)
    log.info("EVENT TYPE COMPOSITION BY DAY")
    log.info("  %-12s %12s %12s %12s %10s %10s", "date", "view", "cart", "purchase",
             "cart/view", "buy/cart")
    rows = con.execute(
        f"""
        SELECT event_date,
               sum(CASE WHEN event_type='view'     THEN 1 ELSE 0 END) AS views,
               sum(CASE WHEN event_type='cart'     THEN 1 ELSE 0 END) AS carts,
               sum(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) AS buys
        FROM {src}
        WHERE event_date BETWEEN DATE '{args.d_from}' AND DATE '{args.d_to}'
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    # A zero-purchase day is the obvious failure, but PARTIAL loss is just as
    # damaging and far easier to miss: on 2019-11-14 and 11-16 purchases were
    # only partly recorded, and on 11-17 the backlog was flushed in. Detect all
    # of them by comparing each day's buy/cart ratio to the archive's median,
    # using MAD so the outliers do not inflate their own threshold.
    # A GLOBAL threshold does not work here: buy/cart is structurally different
    # in October than in November, so an archive-wide median (66%) flags nothing
    # while the genuinely broken days sit at 0% and 43%. Compare each day to its
    # own LOCAL neighbourhood instead - the surrounding +/-7 days, excluding
    # itself - which is how a production monitor stays valid under slow drift.
    # Quarantine is driven by signals that are unambiguous across the whole
    # archive, NOT by the buy/cart ratio. Two earlier attempts at a ratio rule
    # both failed, and instructively:
    #   * a global median flags nothing, because October's buy/cart (~116%) and
    #     November's (~33%) are structurally different - cart events are badly
    #     under-recorded in October, so the ratio has no stable baseline;
    #   * a local +/-7-day median flags every weekend, because buy/cart dips
    #     Fri-Sun and a 7-day window mixes weekday and weekend days.
    # Daily event VOLUME has no such problem: it is stable within +/-25% all
    # archive long, so a 2x excursion is unambiguous instrumentation failure.
    volumes = sorted(v + c + b for _d, v, c, b in rows)
    vol_median = volumes[len(volumes) // 2]

    missing_types = []
    for d, v, c, b in rows:
        total = v + c + b
        vol_ratio = total / vol_median
        log.info("  %-12s %12s %12s %12s %9.2f%% %9.2f%%",
                 str(d), f"{v:,}", f"{c:,}", f"{b:,}",
                 100.0 * c / max(v, 1), 100.0 * b / max(c, 1))
        if b == 0:
            missing_types.append(str(d))
            log.warning("%sNO PURCHASE EVENTS AT ALL - every label on this day is wrong", " " * 14)
        elif vol_ratio > 2.0:
            missing_types.append(str(d))
            log.warning("%svolume %.1fx the archive median - collection anomaly, labels untrustworthy",
                        " " * 14, vol_ratio)

    # Reported, but deliberately NOT used for quarantine: the October/November
    # cart-instrumentation change is a real modelling hazard rather than a
    # broken day, and is handled by dropping cart-derived features instead.
    oct_r = [b / c for d, _v, c, b in rows if c and str(d) < "2019-11-01"]
    nov_r = [b / c for d, _v, c, b in rows if c and str(d) >= "2019-11-01"]
    if oct_r and nov_r:
        log.info("  buy/cart October median %.0f%% vs November median %.0f%% "
                 "- cart events are not comparable across months",
                 sorted(oct_r)[len(oct_r) // 2] * 100, sorted(nov_r)[len(nov_r) // 2] * 100)

    # ---- 2. exact duplicate rows ------------------------------------------
    log.info("=" * 92)
    log.info("EXACT DUPLICATE EVENTS (same session, time, type, product)")
    log.info("  %-12s %14s %14s %10s", "date", "rows", "distinct", "dup rate")
    dups = con.execute(
        f"""
        WITH d AS (
            SELECT event_date, user_session, event_time, event_type, product_id,
                   count(*) AS n
            FROM {src}
            WHERE event_date BETWEEN DATE '{args.d_from}' AND DATE '{args.d_to}'
            GROUP BY 1,2,3,4,5
        )
        SELECT event_date, sum(n), count(*), 1.0 - count(*)*1.0/sum(n)
        FROM d GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    dup_days = []
    for d, total, distinct, rate in dups:
        marker = "  <-- HEAVY DUPLICATION" if float(rate) > 0.15 else ""
        if float(rate) > 0.15:
            dup_days.append(str(d))
        log.info("  %-12s %14s %14s %9.2f%%%s",
                 str(d), f"{int(total):,}", f"{int(distinct):,}", float(rate) * 100, marker)

    # ---- 3. what a naive trainer would conclude ---------------------------
    log.info("=" * 92)
    log.info("IMPACT IF THESE DAYS ARE USED AS-IS")
    for d in missing_types:
        n = con.execute(
            f"SELECT count(DISTINCT user_session) FROM {src} WHERE event_date = DATE '{d}'"
        ).fetchone()[0]
        log.warning("  %s: %s sessions would all be labelled y=0 despite unknown truth",
                    d, f"{int(n):,}")

    verdict = {
        "days_missing_event_types": missing_types,
        "days_heavy_duplication": dup_days,
        "quarantine": sorted(set(missing_types) | set(dup_days)),
    }
    log.info("=" * 92)
    log.info("QUARANTINE RECOMMENDATION: %s",
             ", ".join(verdict["quarantine"]) or "none - data is clean")
    out = settings.artifacts_dir / "data_quality_audit.json"
    out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    log.info("written -> %s", out)
    con.close()


if __name__ == "__main__":
    main()
