"""Build the session index: one row per visit, with point-in-time user history.

The provider ships a `user_session` column, but we do not trust it blindly -
`--rederive` rebuilds sessions from a 30-minute inactivity gap and the two are
compared in the profile report.

Point-in-time correctness
-------------------------
User-history features (prior sessions, prior purchases, recency) are computed
with window functions ordered by `session_start`, using frames that end at
`1 preceding`. A session therefore never sees itself or any later session.
This is the cheap O(n log n) equivalent of an as-of join and is what keeps the
training set honest.
"""

from __future__ import annotations

import argparse

import duckdb

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)


def _con(settings) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def build(settings, rederive: bool = False, until: str | None = None) -> int:
    con = _con(settings)
    src = f"read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')"

    # The replayer clock is enforced here: when `until` is set, no event at or
    # after that instant is visible to anything downstream.
    time_filter = f"WHERE event_time < TIMESTAMP '{until}'" if until else ""
    if until:
        log.info("clock enforced: only events strictly before %s are visible", until)

    if rederive:
        log.info("re-deriving sessions from a %d-minute inactivity gap", settings.session_gap_minutes)
        session_key = f"""
            SELECT *,
                   user_id::VARCHAR || '_' || sum(new_session)
                       OVER (PARTITION BY user_id ORDER BY event_time
                             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)::VARCHAR
                       AS session_key
            FROM (
                SELECT *,
                       CASE WHEN event_time - lag(event_time) OVER (
                                PARTITION BY user_id ORDER BY event_time
                            ) > INTERVAL '{settings.session_gap_minutes} minutes'
                            OR lag(event_time) OVER (
                                PARTITION BY user_id ORDER BY event_time
                            ) IS NULL
                       THEN 1 ELSE 0 END AS new_session
                FROM {src} {time_filter}
            )
        """
    else:
        # 12 events in the archive carry a NULL user_session - all cart events,
        # belonging to 12 unrelated users. GROUP BY collapses them into one
        # phantom session containing twelve strangers. They never reached the
        # training table, but only because `e.user_session = c.session_key` is
        # never true for NULL, which is luck rather than a control. Excluded
        # explicitly here so the session index, the warehouse and dim_user are
        # all clean at the source.
        null_filter = "user_session IS NOT NULL"
        where = f"{time_filter} AND {null_filter}" if time_filter else f"WHERE {null_filter}"
        session_key = f"SELECT *, user_session AS session_key FROM {src} {where}"

    out = settings.features_dir / "sessions.parquet"
    log.info("building session index -> %s", out)

    con.execute(
        f"""
        COPY (
            WITH ev AS ({session_key}),
            sess AS (
                SELECT
                    session_key,
                    -- min(), not any_value(): 939 sessions (0.004%) contain
                    -- events from more than one user_id - one spans three
                    -- users across 43 events. Ownership is genuinely ambiguous
                    -- there, but any_value() resolves it ARBITRARILY, so the
                    -- session-to-user mapping could change between runs and
                    -- take the point-in-time history features with it. min()
                    -- is equally arbitrary but reproducible.
                    min(user_id)                                           AS user_id,
                    min(event_time)                                        AS session_start,
                    max(event_time)                                        AS session_end,
                    count(*)                                               AS n_events,
                    max(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) AS bought,
                    max(CASE WHEN event_type='cart'     THEN 1 ELSE 0 END) AS carted,
                    min(CASE WHEN event_type='purchase' THEN event_time END) AS first_purchase_at,
                    sum(CASE WHEN event_type='purchase' THEN price ELSE 0 END) AS revenue
                FROM ev
                GROUP BY session_key
            )
            SELECT
                session_key,
                user_id,
                session_start,
                session_end,
                CAST(session_start AS DATE)                                  AS session_date,
                n_events,
                bought,
                carted,
                first_purchase_at,
                revenue,
                date_diff('second', session_start, session_end)              AS duration_sec,

                -- ---- point-in-time user history (strictly prior sessions) ----
                row_number() OVER w - 1                                      AS user_prior_sessions,
                COALESCE(sum(bought) OVER w_prior, 0)                        AS user_prior_purchases,
                COALESCE(sum(revenue) OVER w_prior, 0.0)                     AS user_prior_revenue,
                date_diff('hour', lag(session_end) OVER w, session_start)    AS hours_since_last_session
            FROM sess
            WINDOW
                w AS (PARTITION BY user_id ORDER BY session_start),
                w_prior AS (PARTITION BY user_id ORDER BY session_start
                            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        )
        TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    n = con.execute(f"SELECT count(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    stats = con.execute(
        f"""
        SELECT avg(bought), avg(carted), median(n_events),
               quantile_cont(n_events, 0.9), avg(duration_sec)
        FROM read_parquet('{out.as_posix()}')
        """
    ).fetchone()
    log.info("sessions: %s", f"{n:,}")
    log.info("purchase rate: %.3f%%", float(stats[0]) * 100)
    log.info("cart rate:     %.3f%%", float(stats[1]) * 100)
    log.info("events/session: median %.0f, p90 %.0f", float(stats[2]), float(stats[3]))
    log.info("mean duration: %.0f sec", float(stats[4]))
    con.close()
    return int(n)


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rederive", action="store_true", help="rebuild sessions from inactivity gap")
    ap.add_argument("--until", default=None, help="clock cutoff, e.g. 2019-11-20T00:00:00")
    args = ap.parse_args()
    build(settings, rederive=args.rederive, until=args.until)


if __name__ == "__main__":
    main()
