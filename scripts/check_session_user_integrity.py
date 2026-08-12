"""Is user_session unique to one user?

The session index takes any_value(user_id) per session. That is only correct if
a session belongs to exactly one user. dim_user came out 129 users short of the
raw distinct count, which is the signature of sessions shared across users - and
if that is happening, every point-in-time user-history feature is attributing
some behaviour to the wrong person.
"""

from __future__ import annotations

import duckdb

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
    src = f"read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')"

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE su AS
        SELECT user_session, count(DISTINCT user_id) AS n_users, count(*) AS n_events
        FROM {src}
        WHERE user_session IS NOT NULL
        GROUP BY user_session
        HAVING count(DISTINCT user_id) > 1
        """
    )
    n_bad, n_ev = con.execute("SELECT count(*), COALESCE(sum(n_events),0) FROM su").fetchone()
    total = con.execute(
        f"SELECT count(DISTINCT user_session) FROM {src} WHERE user_session IS NOT NULL"
    ).fetchone()[0]

    log.info("=" * 70)
    log.info("sessions spanning more than one user_id: %s of %s (%.5f%%)",
             f"{int(n_bad):,}", f"{int(total):,}", 100.0 * n_bad / max(total, 1))
    log.info("events inside those sessions:            %s", f"{int(n_ev):,}")

    if n_bad:
        log.info("worst offenders:")
        for r in con.execute(
            "SELECT user_session, n_users, n_events FROM su ORDER BY n_users DESC LIMIT 5"
        ).fetchall():
            log.info("   %s  users=%s events=%s", r[0], r[1], r[2])

        # Do any of them reach the training table?
        train = settings.features_dir / "train_k5.parquet"
        if train.exists():
            m = con.execute(
                f"""
                SELECT count(*) FROM read_parquet('{train.as_posix()}') t
                JOIN su ON su.user_session = t.session_key
                """
            ).fetchone()[0]
            log.info("of those, rows present in train_k5:      %s", f"{int(m):,}")
    log.info("=" * 70)
    con.close()


if __name__ == "__main__":
    main()
