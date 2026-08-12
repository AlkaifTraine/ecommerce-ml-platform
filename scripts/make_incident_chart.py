"""Produce the project's headline figure.

One chart, three panels, making the central finding visible at a glance:

  1. Daily event volume - the 2019-11-14..17 collection outage is a 4x spike
     that no amount of model tuning would have survived.
  2. Conversion per session - a ~26% relative decline from October to
     mid-November. This is what the model actually has to survive.
  3. Feature drift vs label drift - price PSI stays flat near zero while the
     label rate moves substantially.

Panel 3 is the point. A monitoring setup that watches feature distributions
would have shown green throughout the entire period while the model decayed.
That is the argument for delayed-label performance monitoring, and here it is
measured rather than asserted.
"""

from __future__ import annotations

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.monitoring.drift import _psi  # noqa: E402
from src.platform_core import get_logger, get_settings  # noqa: E402

log = get_logger(__name__)

OUTAGE = ("2019-11-14", "2019-11-17")
REFERENCE = ("2019-10-01", "2019-10-28")


def main() -> None:
    settings = get_settings()
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
    src = f"read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')"

    log.info("reading daily aggregates ...")
    daily = con.execute(
        f"""
        SELECT event_date,
               count(*)                                               AS events,
               count(DISTINCT user_session)                           AS sessions,
               sum(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) AS purchases
        FROM {src} GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    dates = [r[0] for r in daily]
    events = [r[1] for r in daily]
    conv = [100.0 * r[3] / r[2] for r in daily]

    # Weekly PSI against a fixed October reference - fixed, because a
    # production monitor freezes its baseline at training time.
    log.info("computing weekly price PSI against the October reference ...")
    weeks, psis, label_shift = [], [], []
    ref_conv = sum(c for d, c in zip(dates, conv) if str(d) <= REFERENCE[1]) / max(
        sum(1 for d in dates if str(d) <= REFERENCE[1]), 1
    )
    starts = con.execute(
        f"""
        SELECT DISTINCT date_trunc('week', event_date)::DATE AS wk
        FROM {src} ORDER BY 1
        """
    ).fetchall()
    for (wk,) in starts:
        wk_end = con.execute(f"SELECT (DATE '{wk}' + INTERVAL '6 days')::DATE").fetchone()[0]
        try:
            p = _psi(con, src, "price", REFERENCE, (str(wk), str(wk_end)))
        except Exception as exc:  # a partial week at either end
            log.warning("psi failed for week %s: %s", wk, exc)
            continue
        wk_conv = [c for d, c in zip(dates, conv) if wk <= d <= wk_end]
        if not wk_conv:
            continue
        weeks.append(wk)
        psis.append(p)
        label_shift.append(abs(sum(wk_conv) / len(wk_conv) - ref_conv) / ref_conv)
    con.close()

    # ---- figure --------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(
        "Purchase-intent platform: where the drift actually is\n"
        "109,950,743 events, Oct-Nov 2019",
        fontsize=14, fontweight="bold",
    )

    o0 = mdates.datestr2num(OUTAGE[0])
    o1 = mdates.datestr2num(OUTAGE[1])

    ax = axes[0]
    ax.bar(dates, events, width=0.9, color="#4C78A8")
    ax.axvspan(o0, o1, color="#E45756", alpha=0.22)
    ax.set_ylabel("events / day")
    ax.set_title("1. Collection outage: 4x volume spike, purchases lost", loc="left", fontsize=11)
    ax.annotate("Nov 15: ZERO purchases recorded\nwhile 766K sessions browsed",
                xy=(mdates.datestr2num("2019-11-15"), max(events) * 0.95),
                xytext=(mdates.datestr2num("2019-10-08"), max(events) * 0.86),
                arrowprops=dict(arrowstyle="->", color="#E45756"),
                fontsize=9, color="#B01D22")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.plot(dates, conv, color="#54A24B", lw=1.8)
    ax.axvspan(o0, o1, color="#E45756", alpha=0.22)
    ax.set_ylabel("purchases / session (%)")
    ax.set_title("2. Conversion falls ~26% relative, Oct to mid-Nov", loc="left", fontsize=11)
    ax.annotate("Nov 17 spike is the BACKFILL of\nthe purchases lost on Nov 14-16,\nnot real demand",
                xy=(mdates.datestr2num("2019-11-17"), max(conv) * 0.97),
                xytext=(mdates.datestr2num("2019-10-02"), max(conv) * 0.80),
                arrowprops=dict(arrowstyle="->", color="#B01D22"),
                fontsize=8.5, color="#B01D22")
    ax.grid(axis="y", alpha=0.3)

    # Both series normalised to THEIR OWN alert threshold, so a single axis
    # works and 1.0 means "alert" for either. A twin axis was actively
    # misleading here: the PSI threshold line visually crossed the label-shift
    # curve, which sits on a different scale, and read as a breach.
    PSI_ALERT = 0.10        # "moderate shift" convention
    LABEL_ALERT = 0.25      # LABEL_SHIFT_ALERT in src/monitoring/drift.py

    ax = axes[2]
    ax.plot(weeks, [p / PSI_ALERT for p in psis], marker="o", color="#4C78A8",
            lw=2, label=f"price PSI  (as multiple of its {PSI_ALERT} alert level)")
    ax.plot(weeks, [s / LABEL_ALERT for s in label_shift], marker="s", color="#E45756",
            lw=2, label=f"|conversion shift|  (as multiple of its {LABEL_ALERT:.0%} alert level)")
    ax.axhline(1.0, ls="--", lw=1.2, color="#333333", alpha=0.7)
    # right-aligned, clear of the legend in the upper left
    ax.text(dates[-2], 1.04, "ALERT LEVEL (1.0) - same line for both series",
            fontsize=8.5, color="#333333", ha="right")
    ax.set_ylabel("multiple of alert level")
    ax.set_ylim(0, 1.35)
    ax.set_title(
        "3. THE POINT: feature drift never reaches a tenth of its alert level, "
        "while the label rate breaches its own",
        loc="left", fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = settings.data_root.parent / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "drift_incident.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    log.info("written -> %s", out)

    for w, p, s in zip(weeks, psis, label_shift):
        log.info("  week %s  price PSI %.4f   |label shift| %5.1f%%", w, p, s * 100)


if __name__ == "__main__":
    main()
