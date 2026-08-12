"""Queries the warehouse exists to answer - and Postgres structurally cannot.

The OLTP store keeps a rolling 7-day window, so none of these questions can be
asked of it. That is the point of having both.
"""

from __future__ import annotations

import duckdb

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    path = settings.data_root / "warehouse.duckdb"
    if not path.exists():
        raise SystemExit(f"no warehouse at {path} - run src.warehouse.build_warehouse")
    con = duckdb.connect(str(path), read_only=True)

    log.info("=" * 78)
    log.info("Q1  Conversion trend by half-month (the drift the model has to survive)")
    for r in con.execute(
        """
        SELECT CASE WHEN day(date_key) <= 15 THEN strftime(date_key, '%Y-%m') || ' 1st half'
                    ELSE strftime(date_key, '%Y-%m') || ' 2nd half' END AS period,
               sum(sessions)                        AS sessions,
               sum(purchases)                       AS purchases,
               100.0*sum(purchases)/sum(sessions)   AS conv_pct,
               avg(avg_price)                       AS avg_price
        FROM agg_daily WHERE NOT date_key IN (SELECT date_key FROM dim_date WHERE is_quarantined)
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall():
        log.info("    %-20s sessions=%12s purchases=%10s conv=%5.2f%% price=%7.2f",
                 r[0], f"{int(r[1]):,}", f"{int(r[2]):,}", float(r[3]), float(r[4]))

    log.info("=" * 78)
    log.info("Q2  The collection outage, visible in one query")
    for r in con.execute(
        """
        SELECT date_key, events, purchases, round(buy_per_cart*100, 1) AS buy_per_cart_pct,
               is_quarantined
        FROM agg_daily JOIN dim_date USING (date_key)
        WHERE date_key BETWEEN DATE '2019-11-12' AND DATE '2019-11-19'
        ORDER BY date_key
        """
    ).fetchall():
        flag = "  <-- QUARANTINED" if r[4] else ""
        log.info("    %s  events=%10s purchases=%8s buy/cart=%6s%%%s",
                 r[0], f"{int(r[1]):,}", f"{int(r[2]):,}",
                 "n/a" if r[3] is None else f"{float(r[3]):.1f}", flag)

    log.info("=" * 78)
    log.info("Q3  Category mix shift, October vs late November")
    for r in con.execute(
        """
        WITH oct AS (
            SELECT category_l1, sum(events) e FROM agg_category_daily
            WHERE date_key < DATE '2019-11-01' GROUP BY 1
        ),
        nov AS (
            SELECT category_l1, sum(events) e FROM agg_category_daily
            WHERE date_key >= DATE '2019-11-18' GROUP BY 1
        )
        SELECT o.category_l1,
               100.0*o.e/(SELECT sum(e) FROM oct) AS oct_pct,
               100.0*n.e/(SELECT sum(e) FROM nov) AS nov_pct
        FROM oct o JOIN nov n USING (category_l1)
        ORDER BY oct_pct DESC LIMIT 6
        """
    ).fetchall():
        log.info("    %-26s Oct %6.2f%%   late-Nov %6.2f%%   delta %+6.2f%%",
                 r[0], float(r[1]), float(r[2]), float(r[2]) - float(r[1]))

    log.info("=" * 78)
    log.info("Q4  Repeat buyers - needs full user history, impossible in a 7-day window")
    r = con.execute(
        """
        SELECT count(*) FILTER (WHERE n_buying_sessions = 0) AS never_bought,
               count(*) FILTER (WHERE n_buying_sessions = 1) AS bought_once,
               count(*) FILTER (WHERE n_buying_sessions > 1) AS repeat_buyers,
               sum(lifetime_revenue) FILTER (WHERE n_buying_sessions > 1)
                 / NULLIF(sum(lifetime_revenue), 0) * 100 AS repeat_revenue_share
        FROM dim_user
        """
    ).fetchone()
    log.info("    never bought   %12s", f"{int(r[0]):,}")
    log.info("    bought once    %12s", f"{int(r[1]):,}")
    log.info("    repeat buyers  %12s", f"{int(r[2]):,}")
    log.info("    -> repeat buyers generate %.1f%% of all revenue", float(r[3]))
    log.info("=" * 78)
    con.close()


if __name__ == "__main__":
    main()
