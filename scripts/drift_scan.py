"""Find where distribution shift actually lives in this dataset.

Motivation
----------
The project's continuous-retraining story was designed around Black Friday
being a dramatic regime change. Measured against the real data it is a ~13%
relative lift in conversion with no price discount signature - far milder than
assumed. Rather than build a demo around an event that is not there, this
scans the whole archive for shift that genuinely exists.

Outputs a daily feature series plus PSI between an early and a late window.
PSI convention (industry standard):
    < 0.10  no meaningful shift
    0.10-0.25  moderate shift, worth watching
    > 0.25  major shift, retraining usually warranted

This doubles as the foundation for the production drift monitor.
"""

from __future__ import annotations

import argparse
import json

import duckdb

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)


def psi(con, src: str, expr: str, win_a: tuple[str, str], win_b: tuple[str, str],
        bins: int = 10) -> float:
    """Population Stability Index for `expr` between two date windows.

    Bin edges come from the EARLY window (the reference), which is what a
    production monitor does: you fix the reference when you train, then measure
    how far later traffic has moved from it.
    """
    edges = con.execute(
        f"""
        SELECT quantile_cont({expr}, [{", ".join(str(i / bins) for i in range(1, bins))}])
        FROM {src}
        WHERE event_date BETWEEN DATE '{win_a[0]}' AND DATE '{win_a[1]}' AND {expr} IS NOT NULL
        """
    ).fetchone()[0]
    edges = sorted(set(float(e) for e in edges))
    if not edges:
        return float("nan")

    def bucket_sql(alias: str) -> str:
        cases = " ".join(
            f"WHEN {expr} <= {e} THEN {i}" for i, e in enumerate(edges)
        )
        return f"CASE {cases} ELSE {len(edges)} END AS {alias}"

    rows = con.execute(
        f"""
        WITH a AS (
            SELECT {bucket_sql('b')}, count(*) AS n FROM {src}
            WHERE event_date BETWEEN DATE '{win_a[0]}' AND DATE '{win_a[1]}' AND {expr} IS NOT NULL
            GROUP BY 1
        ),
        b AS (
            SELECT {bucket_sql('b')}, count(*) AS n FROM {src}
            WHERE event_date BETWEEN DATE '{win_b[0]}' AND DATE '{win_b[1]}' AND {expr} IS NOT NULL
            GROUP BY 1
        ),
        ta AS (SELECT sum(n) s FROM a), tb AS (SELECT sum(n) s FROM b)
        SELECT COALESCE(a.b, b.b),
               COALESCE(a.n, 0) * 1.0 / (SELECT s FROM ta),
               COALESCE(b.n, 0) * 1.0 / (SELECT s FROM tb)
        FROM a FULL OUTER JOIN b ON a.b = b.b
        """
    ).fetchall()

    total = 0.0
    for _b, pa, pb in rows:
        pa = max(float(pa or 0), 1e-6)
        pb = max(float(pb or 0), 1e-6)
        total += (pb - pa) * __import__("math").log(pb / pa)
    return total


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--early", default="2019-10-01:2019-10-31")
    ap.add_argument("--late", default="2019-11-15:2019-11-30")
    args = ap.parse_args()
    win_a = tuple(args.early.split(":"))
    win_b = tuple(args.late.split(":"))

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
    src = f"read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')"

    # ---- daily series -----------------------------------------------------
    log.info("computing daily feature series across the archive ...")
    daily = con.execute(
        f"""
        SELECT event_date,
               count(*)                                                AS events,
               count(DISTINCT user_session)                            AS sessions,
               count(*) * 1.0 / count(DISTINCT user_session)           AS events_per_session,
               sum(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END)  AS purchases,
               sum(CASE WHEN event_type='cart' THEN 1 ELSE 0 END)      AS carts,
               avg(price)                                              AS avg_price,
               median(price)                                           AS med_price,
               avg(CASE WHEN brand IS NULL THEN 1.0 ELSE 0.0 END)      AS brand_null,
               avg(CASE WHEN category_code IS NULL THEN 1.0 ELSE 0.0 END) AS cat_null,
               count(DISTINCT user_id)                                 AS users
        FROM {src} GROUP BY 1 ORDER BY 1
        """
    ).fetchall()

    log.info("=" * 104)
    log.info("%-12s %10s %10s %7s %9s %10s %10s %8s %8s",
             "date", "events", "sessions", "ev/ses", "conv/ses", "avg_price", "med_price",
             "brand_n", "cat_n")
    for r in daily:
        (d, ev, ss, eps, pu, ca, apx, mpx, bn, cn, _us) = r
        log.info("%-12s %10s %10s %7.2f %8.3f%% %10.2f %10.2f %7.1f%% %7.1f%%",
                 str(d), f"{ev:,}", f"{ss:,}", float(eps), 100.0 * pu / ss,
                 float(apx), float(mpx), float(bn) * 100, float(cn) * 100)
    log.info("=" * 104)

    # ---- PSI between windows ---------------------------------------------
    log.info("PSI: %s..%s (reference)  vs  %s..%s (current)", *win_a, *win_b)
    feats = {
        "price": "price",
        "log_price": "ln(price + 1)",
    }
    psis = {}
    for name, expr in feats.items():
        v = psi(con, src, expr, win_a, win_b)
        psis[name] = v
        verdict = "no shift" if v < 0.10 else ("MODERATE" if v < 0.25 else "MAJOR")
        log.info("  %-12s PSI = %.4f   %s", name, v, verdict)

    # categorical shift: share by top-level category
    cat = con.execute(
        f"""
        WITH a AS (
            SELECT split_part(category_code,'.',1) c, count(*) n FROM {src}
            WHERE event_date BETWEEN DATE '{win_a[0]}' AND DATE '{win_a[1]}'
              AND category_code IS NOT NULL GROUP BY 1
        ),
        b AS (
            SELECT split_part(category_code,'.',1) c, count(*) n FROM {src}
            WHERE event_date BETWEEN DATE '{win_b[0]}' AND DATE '{win_b[1]}'
              AND category_code IS NOT NULL GROUP BY 1
        )
        SELECT COALESCE(a.c,b.c),
               COALESCE(a.n,0)*1.0/(SELECT sum(n) FROM a),
               COALESCE(b.n,0)*1.0/(SELECT sum(n) FROM b)
        FROM a FULL OUTER JOIN b ON a.c=b.c ORDER BY 2 DESC
        """
    ).fetchall()
    import math
    cat_psi = 0.0
    log.info("  category mix (top-level):")
    log.info("    %-26s %9s %9s %9s", "category", "early", "late", "delta")
    for c, pa, pb in cat[:12]:
        pa_, pb_ = max(float(pa or 0), 1e-6), max(float(pb or 0), 1e-6)
        cat_psi += (pb_ - pa_) * math.log(pb_ / pa_)
        log.info("    %-26s %8.2f%% %8.2f%% %+8.2f%%", c or "(null)",
                 pa_ * 100, pb_ * 100, (pb_ - pa_) * 100)
    for c, pa, pb in cat[12:]:
        pa_, pb_ = max(float(pa or 0), 1e-6), max(float(pb or 0), 1e-6)
        cat_psi += (pb_ - pa_) * math.log(pb_ / pa_)
    psis["category_mix"] = cat_psi
    verdict = "no shift" if cat_psi < 0.10 else ("MODERATE" if cat_psi < 0.25 else "MAJOR")
    log.info("  %-12s PSI = %.4f   %s", "category_mix", cat_psi, verdict)
    log.info("=" * 104)

    out = settings.artifacts_dir / "drift_scan.json"
    out.write_text(json.dumps({
        "windows": {"early": win_a, "late": win_b},
        "psi": psis,
        "daily": [
            {"date": str(r[0]), "events": int(r[1]), "sessions": int(r[2]),
             "events_per_session": float(r[3]), "purchases": int(r[4]), "carts": int(r[5]),
             "avg_price": float(r[6]), "med_price": float(r[7]),
             "brand_null": float(r[8]), "cat_null": float(r[9]), "users": int(r[10])}
            for r in daily
        ],
    }, indent=2), encoding="utf-8")
    log.info("written -> %s", out)
    con.close()


if __name__ == "__main__":
    main()
