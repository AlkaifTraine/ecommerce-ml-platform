"""Drift detection, built around what this dataset actually does.

The measured finding this module exists to handle: between October and late
November, feature distributions barely move (price PSI 0.0024, category-mix PSI
0.0116 - both well under the 0.10 "no shift" threshold) while conversion per
session falls about 26% relative. A monitor watching only feature drift would
report green throughout while the model quietly decayed.

So three independent signals are tracked, because they fail differently and
demand different responses:

    COVARIATE SHIFT   inputs move (PSI on features)
                      -> often survivable; recalibrate before retraining

    PRIOR / LABEL     outcome rate moves while inputs do not
    SHIFT             -> the model's scores are still ranked correctly but
                         mis-calibrated; recalibration usually beats retraining

    DATA QUALITY      the pipeline broke (volume spike, purchases missing)
                      -> retraining on this is actively harmful; quarantine

Only the first two are drift. The third is an outage, and treating it as drift
is how 3.3M sessions end up silently labelled "did not buy" - which is exactly
what 2019-11-14..17 would have done to this project.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import duckdb

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)

PSI_MODERATE = 0.10
PSI_MAJOR = 0.25
LABEL_SHIFT_WARN = 0.15   # 15% relative move in conversion
LABEL_SHIFT_ALERT = 0.25
VOLUME_ANOMALY = 2.0


@dataclass
class DriftReport:
    as_of: str
    window_days: int
    feature_psi: dict[str, float] = field(default_factory=dict)
    label_rate: float = float("nan")
    label_baseline: float = float("nan")
    label_shift_rel: float = float("nan")
    data_quality: list[str] = field(default_factory=list)
    verdict: str = "OK"
    action: str = "none"
    reasons: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _psi(con, src: str, expr: str, ref: tuple[str, str], cur: tuple[str, str],
         bins: int = 10) -> float:
    """PSI with bin edges fixed on the REFERENCE window.

    Fixing edges on the reference is what a production monitor does: the
    baseline is frozen at training time, and later traffic is measured against
    it. Recomputing edges per window would hide the very shift being measured.
    """
    edges = con.execute(
        f"""
        SELECT quantile_cont({expr}, [{", ".join(str(i / bins) for i in range(1, bins))}])
        FROM {src}
        WHERE event_date BETWEEN DATE '{ref[0]}' AND DATE '{ref[1]}' AND {expr} IS NOT NULL
        """
    ).fetchone()[0]
    edges = sorted({float(e) for e in edges})
    if not edges:
        return float("nan")

    case = " ".join(f"WHEN {expr} <= {e} THEN {i}" for i, e in enumerate(edges))
    bucket = f"CASE {case} ELSE {len(edges)} END"

    rows = con.execute(
        f"""
        WITH a AS (
            SELECT {bucket} AS b, count(*) n FROM {src}
            WHERE event_date BETWEEN DATE '{ref[0]}' AND DATE '{ref[1]}' AND {expr} IS NOT NULL
            GROUP BY 1
        ), b AS (
            SELECT {bucket} AS b, count(*) n FROM {src}
            WHERE event_date BETWEEN DATE '{cur[0]}' AND DATE '{cur[1]}' AND {expr} IS NOT NULL
            GROUP BY 1
        )
        SELECT COALESCE(a.n,0)*1.0/(SELECT sum(n) FROM a),
               COALESCE(b.n,0)*1.0/(SELECT sum(n) FROM b)
        FROM a FULL OUTER JOIN b ON a.b = b.b
        """
    ).fetchall()

    total = 0.0
    for pa, pb in rows:
        pa_, pb_ = max(float(pa or 0), 1e-6), max(float(pb or 0), 1e-6)
        total += (pb_ - pa_) * math.log(pb_ / pa_)
    return total


def check(settings, as_of: str, window_days: int = 7, baseline_days: int = 28) -> DriftReport:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
    src = f"read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')"

    cur_lo = con.execute(
        f"SELECT CAST(DATE '{as_of}' - INTERVAL '{window_days} days' AS DATE)"
    ).fetchone()[0]
    ref_hi = cur_lo
    ref_lo = con.execute(
        f"SELECT CAST(DATE '{ref_hi}' - INTERVAL '{baseline_days} days' AS DATE)"
    ).fetchone()[0]

    rep = DriftReport(as_of=as_of, window_days=window_days)
    ref = (str(ref_lo), str(ref_hi))
    cur = (str(cur_lo), as_of)
    log.info("reference %s..%s   current %s..%s", *ref, *cur)

    # ---- 1. covariate shift -------------------------------------------
    for name, expr in {"price": "price", "log_price": "ln(price + 1)"}.items():
        rep.feature_psi[name] = _psi(con, src, expr, ref, cur)

    # ---- 2. prior / label shift ---------------------------------------
    q = con.execute(
        f"""
        SELECT
          (SELECT sum(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END)*1.0
                  / count(DISTINCT user_session)
             FROM {src} WHERE event_date BETWEEN DATE '{cur[0]}' AND DATE '{cur[1]}'),
          (SELECT sum(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END)*1.0
                  / count(DISTINCT user_session)
             FROM {src} WHERE event_date BETWEEN DATE '{ref[0]}' AND DATE '{ref[1]}')
        """
    ).fetchone()
    rep.label_rate, rep.label_baseline = float(q[0] or 0), float(q[1] or 0)
    if rep.label_baseline > 0:
        rep.label_shift_rel = (rep.label_rate - rep.label_baseline) / rep.label_baseline

    # ---- 3. data quality ----------------------------------------------
    for d, ev, pu, ca in con.execute(
        f"""
        SELECT event_date, count(*),
               sum(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END),
               sum(CASE WHEN event_type='cart' THEN 1 ELSE 0 END)
        FROM {src} WHERE event_date BETWEEN DATE '{cur[0]}' AND DATE '{cur[1]}'
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall():
        if pu == 0:
            rep.data_quality.append(f"{d}: ZERO purchase events recorded")
        if ca == 0:
            rep.data_quality.append(f"{d}: ZERO cart events recorded")

    med = con.execute(
        f"""
        SELECT median(n) FROM (
            SELECT count(*) n FROM {src}
            WHERE event_date BETWEEN DATE '{ref[0]}' AND DATE '{ref[1]}'
            GROUP BY event_date)
        """
    ).fetchone()[0]
    if med:
        for d, ev in con.execute(
            f"""
            SELECT event_date, count(*) FROM {src}
            WHERE event_date BETWEEN DATE '{cur[0]}' AND DATE '{cur[1]}'
            GROUP BY 1 ORDER BY 1
            """
        ).fetchall():
            if ev / float(med) > VOLUME_ANOMALY:
                rep.data_quality.append(
                    f"{d}: volume {ev / float(med):.1f}x the reference median"
                )
    con.close()

    _decide(rep)
    return rep


def _decide(rep: DriftReport) -> None:
    """Map signals to an action. Data quality outranks everything."""
    if rep.data_quality:
        rep.verdict = "DATA_QUALITY_INCIDENT"
        rep.action = "quarantine_and_hold"
        rep.reasons = rep.data_quality + [
            "retraining on days with missing events would teach the model that "
            "the outage predicts 'no purchase' - quarantine first"
        ]
        return

    worst_psi = max((v for v in rep.feature_psi.values() if not math.isnan(v)), default=0.0)
    shift = abs(rep.label_shift_rel) if not math.isnan(rep.label_shift_rel) else 0.0

    if worst_psi > PSI_MAJOR:
        rep.verdict = "COVARIATE_SHIFT"
        rep.action = "retrain"
        rep.reasons.append(f"feature PSI {worst_psi:.4f} exceeds {PSI_MAJOR}")
    elif shift > LABEL_SHIFT_ALERT:
        rep.verdict = "LABEL_SHIFT"
        # Inputs unchanged, outcome rate moved: ranking is probably still fine,
        # calibration is not. Recalibration is cheaper and safer than retraining.
        rep.action = "recalibrate"
        rep.reasons.append(
            f"conversion moved {rep.label_shift_rel:+.1%} vs baseline while "
            f"feature PSI stayed at {worst_psi:.4f} - prior shift, not covariate shift"
        )
    elif shift > LABEL_SHIFT_WARN or worst_psi > PSI_MODERATE:
        rep.verdict = "WATCH"
        rep.action = "none"
        rep.reasons.append(
            f"conversion {rep.label_shift_rel:+.1%}, worst feature PSI {worst_psi:.4f}"
        )
    else:
        rep.verdict = "OK"
        rep.action = "none"
        rep.reasons.append("no material shift detected")


def main() -> None:
    import argparse

    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", required=True, help="e.g. 2019-11-20")
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--baseline-days", type=int, default=28)
    args = ap.parse_args()

    rep = check(settings, args.as_of, args.window_days, args.baseline_days)

    log.info("=" * 70)
    log.info("DRIFT REPORT as of %s", rep.as_of)
    for k, v in rep.feature_psi.items():
        tag = "no shift" if v < PSI_MODERATE else ("MODERATE" if v < PSI_MAJOR else "MAJOR")
        log.info("  PSI %-12s %.4f   %s", k, v, tag)
    log.info("  conversion    %.4f%%  (baseline %.4f%%, %+.1f%%)",
             rep.label_rate * 100, rep.label_baseline * 100, rep.label_shift_rel * 100)
    for f in rep.data_quality:
        log.warning("  DATA QUALITY: %s", f)
    log.info("  VERDICT  %s", rep.verdict)
    log.info("  ACTION   %s", rep.action)
    for r in rep.reasons:
        log.info("    - %s", r)
    log.info("=" * 70)

    out = settings.artifacts_dir / f"drift_{rep.as_of}.json"
    out.write_text(rep.to_json(), encoding="utf-8")
    log.info("written -> %s", out)


if __name__ == "__main__":
    main()
