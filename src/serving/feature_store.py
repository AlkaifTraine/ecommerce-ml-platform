"""Online feature store for point-in-time user history.

Why this exists
---------------
The serving API previously accepted `prior_sessions`, `prior_purchases` and
`prior_revenue` as request fields defaulting to ZERO, and never looked them up.
`scripts/score_demo.py` did not send them, so every scored session was treated
as a brand-new user with no history - while `user_prior_conv_rate` is the
SECOND most important feature by gain (513,033). Serving was running with its
second-strongest feature pinned at zero.

The training/serving parity test did not catch this: it passes history in
explicitly, so it verified that the COMPUTATION matches, not that production
SUPPLIES the inputs. Those are different guarantees.

Design
------
Two-tier, which is the standard shape:

    offline  warehouse dim_user (full history, all 61 days)
                 |  materialise on a schedule
                 v
    online   Redis hash per user, O(1) lookup at serving time
                 |  fallback
                 v
             Postgres storefront.users (7-day hot window only)

`found` is returned explicitly so the API can distinguish "this user genuinely
has no history" - where zeros are CORRECT - from "the store was unreachable",
where zeros are a silent lie. Conflating those is the bug this module fixes.

Staleness
---------
Online values are as of the last materialisation, so they exclude the in-flight
session and anything since. That is inherent to every online store and is
bounded by the materialisation schedule; it is not the same as being wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.platform_core import get_logger, get_settings
from src.serving.features_online import UserHistory

log = get_logger(__name__)

KEY_PREFIX = "feat:user:"
TTL_SECONDS = 7 * 24 * 3600


@dataclass
class LookupResult:
    history: UserHistory
    found: bool
    source: str  # "redis" | "postgres" | "absent" | "unavailable"


class OnlineFeatureStore:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._redis = None

    # -- connection -----------------------------------------------------
    @property
    def redis(self):
        if self._redis is None:
            import redis

            self._redis = redis.Redis.from_url(
                self.settings.redis_url, socket_timeout=2, socket_connect_timeout=2
            )
        return self._redis

    # -- materialisation -------------------------------------------------
    def materialize_from_warehouse(self, batch: int = 50_000) -> int:
        """Push full-history user aggregates from the warehouse into Redis.

        Sourced from the warehouse rather than Postgres because the OLTP store
        keeps only a 7-day hot window - using it would give serving a different
        (truncated) history than training saw, which is training/serving skew
        by construction.
        """
        import duckdb

        wh = self.settings.data_root / "warehouse.duckdb"
        if not wh.exists():
            raise SystemExit(f"no warehouse at {wh} - run src.warehouse.build_warehouse")

        con = duckdb.connect(str(wh), read_only=True)
        rows = con.execute(
            """
            SELECT user_id, n_sessions, n_buying_sessions, lifetime_revenue, last_seen
            FROM dim_user
            """
        ).fetchall()
        con.close()

        pipe = self.redis.pipeline(transaction=False)
        n = 0
        for user_id, n_sessions, n_buying, revenue, last_seen in rows:
            pipe.hset(
                f"{KEY_PREFIX}{int(user_id)}",
                mapping={
                    "n": int(n_sessions or 0),
                    "b": int(n_buying or 0),
                    "r": float(revenue or 0.0),
                    "t": str(last_seen or ""),
                },
            )
            pipe.expire(f"{KEY_PREFIX}{int(user_id)}", TTL_SECONDS)
            n += 1
            if n % batch == 0:
                pipe.execute()
                pipe = self.redis.pipeline(transaction=False)
                log.info("materialised %s users", f"{n:,}")
        pipe.execute()
        log.info("materialised %s users into the online store", f"{n:,}")
        return n

    # -- lookup ----------------------------------------------------------
    def get(self, user_id: int) -> LookupResult:
        try:
            raw = self.redis.hgetall(f"{KEY_PREFIX}{int(user_id)}")
        except Exception as exc:
            # Store unreachable. Returning zeros here would be indistinguishable
            # from a genuine new user, so say so and let the caller decide.
            log.warning("online store unreachable (%s); falling back to Postgres", exc)
            return self._from_postgres(user_id)

        if not raw:
            return self._from_postgres(user_id)

        g = {k.decode(): v.decode() for k, v in raw.items()}
        return LookupResult(
            history=UserHistory(
                prior_sessions=int(g.get("n", 0)),
                prior_purchases=int(g.get("b", 0)),
                prior_revenue=float(g.get("r", 0.0)),
            ),
            found=True,
            source="redis",
        )

    def _from_postgres(self, user_id: int) -> LookupResult:
        """Fallback. Only sees the 7-day hot window, so it is a degraded answer."""
        try:
            import psycopg2

            conn = psycopg2.connect(self.settings.postgres_dsn, connect_timeout=2)
            cur = conn.cursor()
            cur.execute(
                "SELECT total_sessions, total_orders, lifetime_value "
                "FROM storefront.users WHERE user_id = %s",
                (int(user_id),),
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
        except Exception as exc:
            log.warning("postgres fallback failed for user %s: %s", user_id, exc)
            return LookupResult(UserHistory(), found=False, source="unavailable")

        if row is None:
            # Genuinely unknown user. Zeros are the CORRECT answer here.
            return LookupResult(UserHistory(), found=False, source="absent")

        return LookupResult(
            history=UserHistory(
                prior_sessions=int(row[0] or 0),
                prior_purchases=int(row[1] or 0),
                prior_revenue=float(row[2] or 0.0),
            ),
            found=True,
            source="postgres",
        )

    def close(self) -> None:
        if self._redis is not None:
            self._redis.close()
            self._redis = None


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--probe", type=int, default=None, help="look up one user_id")
    args = ap.parse_args()

    store = OnlineFeatureStore()
    try:
        if args.materialize:
            store.materialize_from_warehouse()
        if args.probe is not None:
            r = store.get(args.probe)
            log.info("user %s -> found=%s source=%s sessions=%s purchases=%s revenue=%.2f",
                     args.probe, r.found, r.source, r.history.prior_sessions,
                     r.history.prior_purchases, r.history.prior_revenue)
    finally:
        store.close()


if __name__ == "__main__":
    main()
