"""Storefront consumer: applies replayed events as OLTP transactions.

This is the service that makes Postgres a genuine system of record rather than
a dumping ground. Each event mutates transactional state the way a real
storefront would - sessions accumulate, carts fill, orders get placed, user
lifetime value moves.

Throughput notes
----------------
Row-at-a-time inserts cap out around 2k/sec, which is far too slow for a
replay running at hundreds of times real speed. Everything here is batched and
uses execute_values, and the per-batch work is ordered (users -> products ->
sessions -> events -> carts/orders) so foreign keys are always satisfiable
without deferred constraints.

Retention
---------
`--retain-days` trims session_events to a rolling window of DATA time. This is
what keeps the OLTP store small and fast; history is not lost, it lives in the
Parquet archive and the warehouse. That split is the reason the project needs
both an OLTP and an OLAP store, and it is worth being able to explain.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import execute_values

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)

STREAM = "events.raw"
GROUP = "storefront"
CONSUMER = "storefront-1"

_running = True


def _stop(signum, frame):  # pragma: no cover - signal path
    global _running
    log.info("shutdown signal received, finishing current batch")
    _running = False


class StorefrontConsumer:
    """Applies replayed events as OLTP transactions.

    NOT IDEMPOTENT ACROSS OVERLAPPING REPLAYS, by design and by necessity.
    `seq` is assigned from a running counter, so an event delivered twice gets
    a fresh sequence number and inserts again - measured at 29,693 duplicate
    (session, time, type, product) groups after a replay was restarted over a
    window it had already emitted.

    Content-based deduplication cannot fix this: the archive itself contains
    ~0.2% genuinely-exact duplicate events, so a natural-key unique constraint
    would reject legitimate rows.

    The correct contract is that replaying means re-running the simulation from
    a clock position, so the hot window must be cleared first. `--truncate`
    does that, and `src/replayer/replay.py --reset` should be paired with it.
    """

    def __init__(self, settings, retain_days: int = 7, truncate: bool = False):
        import redis

        self.settings = settings
        self.r = redis.Redis.from_url(settings.redis_url)
        self.conn = psycopg2.connect(settings.postgres_dsn)
        self.conn.autocommit = False
        self.retain_days = retain_days
        self._seq_cache: dict[str, int] = {}
        self.applied = 0

        if truncate:
            self._truncate()

    def _truncate(self) -> None:
        """Clear the OLTP hot window so a replay starts from a clean slate."""
        cur = self.conn.cursor()
        cur.execute(
            """
            TRUNCATE storefront.session_events, storefront.cart_items,
                     storefront.carts, storefront.order_items, storefront.orders,
                     storefront.sessions, storefront.users, storefront.products
            RESTART IDENTITY CASCADE
            """
        )
        self.conn.commit()
        cur.close()
        self._seq_cache.clear()
        log.warning("OLTP hot window truncated - replay will repopulate it")

        try:
            self.r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
            log.info("created consumer group %r on %r", GROUP, STREAM)
        except Exception:
            log.info("consumer group %r already exists", GROUP)

    # ------------------------------------------------------------------
    def apply_batch(self, events: list[dict]) -> None:
        if not events:
            return
        cur = self.conn.cursor()

        # ---- users -----------------------------------------------------
        users: dict[int, tuple[datetime, datetime]] = {}
        for e in events:
            uid, ts = int(e["user_id"]), _ts(e["event_time"])
            lo, hi = users.get(uid, (ts, ts))
            users[uid] = (min(lo, ts), max(hi, ts))
        execute_values(
            cur,
            """
            INSERT INTO storefront.users (user_id, first_seen_at, last_seen_at)
            VALUES %s
            ON CONFLICT (user_id) DO UPDATE
              SET first_seen_at = LEAST(users.first_seen_at, EXCLUDED.first_seen_at),
                  last_seen_at  = GREATEST(users.last_seen_at, EXCLUDED.last_seen_at)
            """,
            [(u, lo, hi) for u, (lo, hi) in users.items()],
            page_size=1000,
        )

        # ---- products --------------------------------------------------
        products: dict[int, tuple] = {}
        for e in events:
            products[int(e["product_id"])] = (
                int(e["product_id"]),
                _int_or_none(e.get("category_id")),
                e.get("category_code"),
                e.get("brand"),
                e.get("price"),
                _ts(e["event_time"]),
            )
        execute_values(
            cur,
            """
            INSERT INTO storefront.products
                (product_id, category_id, category_code, brand, current_price, updated_at)
            VALUES %s
            ON CONFLICT (product_id) DO UPDATE
              SET current_price = EXCLUDED.current_price,
                  updated_at    = GREATEST(products.updated_at, EXCLUDED.updated_at)
            """,
            list(products.values()),
            page_size=1000,
        )

        # ---- sessions --------------------------------------------------
        sessions: dict[str, dict] = {}
        for e in events:
            key, ts = e["user_session"], _ts(e["event_time"])
            s = sessions.setdefault(
                key, {"user_id": int(e["user_id"]), "lo": ts, "hi": ts, "n": 0}
            )
            s["lo"] = min(s["lo"], ts)
            s["hi"] = max(s["hi"], ts)
            s["n"] += 1
        execute_values(
            cur,
            """
            INSERT INTO storefront.sessions
                (session_key, user_id, started_at, last_event_at, n_events)
            VALUES %s
            ON CONFLICT (session_key) DO UPDATE
              SET last_event_at = GREATEST(sessions.last_event_at, EXCLUDED.last_event_at),
                  n_events      = sessions.n_events + EXCLUDED.n_events
            """,
            [(k, v["user_id"], v["lo"], v["hi"], v["n"]) for k, v in sessions.items()],
            page_size=1000,
        )

        # ---- events ----------------------------------------------------
        # seq must be contiguous per session; read the current high-water mark
        # for sessions we have not seen in this process yet.
        unknown = [k for k in sessions if k not in self._seq_cache]
        if unknown:
            cur.execute(
                """
                SELECT session_key, COALESCE(max(seq), 0)
                FROM storefront.session_events
                WHERE session_key = ANY(%s)
                GROUP BY session_key
                """,
                (unknown,),
            )
            for key, mx in cur.fetchall():
                self._seq_cache[key] = int(mx)
            for key in unknown:
                self._seq_cache.setdefault(key, 0)

        rows = []
        for e in sorted(events, key=lambda x: (x["user_session"], _ts(x["event_time"]))):
            key = e["user_session"]
            self._seq_cache[key] += 1
            rows.append(
                (key, self._seq_cache[key], _ts(e["event_time"]),
                 e["event_type"], int(e["product_id"]), e.get("price"))
            )
        execute_values(
            cur,
            """
            INSERT INTO storefront.session_events
                (session_key, seq, event_time, event_type, product_id, price)
            VALUES %s
            ON CONFLICT (session_key, seq) DO NOTHING
            """,
            rows,
            page_size=2000,
        )

        # ---- carts -----------------------------------------------------
        carts = [e for e in events if e["event_type"] == "cart"]
        if carts:
            execute_values(
                cur,
                """
                INSERT INTO storefront.carts (session_key, user_id, created_at)
                VALUES %s ON CONFLICT (session_key) DO NOTHING
                """,
                [(e["user_session"], int(e["user_id"]), _ts(e["event_time"])) for e in carts],
                page_size=1000,
            )
            execute_values(
                cur,
                """
                INSERT INTO storefront.cart_items (cart_id, product_id, qty, unit_price, added_at)
                SELECT c.cart_id, v.product_id, 1, v.price, v.added_at
                FROM (VALUES %s) AS v(session_key, product_id, price, added_at)
                JOIN storefront.carts c ON c.session_key = v.session_key
                ON CONFLICT (cart_id, product_id) DO NOTHING
                """,
                [(e["user_session"], int(e["product_id"]), e.get("price") or 0, _ts(e["event_time"]))
                 for e in carts],
                page_size=1000,
            )

        # ---- orders ----------------------------------------------------
        purchases = [e for e in events if e["event_type"] == "purchase"]
        if purchases:
            totals: dict[str, dict] = {}
            for e in purchases:
                t = totals.setdefault(
                    e["user_session"],
                    {"user_id": int(e["user_id"]), "at": _ts(e["event_time"]), "amt": 0.0, "items": []},
                )
                t["amt"] += float(e.get("price") or 0)
                t["at"] = max(t["at"], _ts(e["event_time"]))
                t["items"].append((int(e["product_id"]), float(e.get("price") or 0)))

            execute_values(
                cur,
                """
                INSERT INTO storefront.orders (session_key, user_id, ordered_at, total_amount)
                VALUES %s
                """,
                [(k, v["user_id"], v["at"], round(v["amt"], 2)) for k, v in totals.items()],
                page_size=500,
            )
            cur.execute(
                """
                UPDATE storefront.users u
                   SET total_orders   = u.total_orders + o.n,
                       lifetime_value = u.lifetime_value + o.amt
                  FROM (SELECT user_id, count(*) AS n, sum(total_amount) AS amt
                          FROM storefront.orders
                         WHERE session_key = ANY(%s)
                         GROUP BY user_id) o
                 WHERE u.user_id = o.user_id
                """,
                (list(totals.keys()),),
            )
            cur.execute(
                "UPDATE storefront.carts SET status='converted' WHERE session_key = ANY(%s)",
                (list(totals.keys()),),
            )

        self.conn.commit()
        cur.close()
        self.applied += len(events)

    # ------------------------------------------------------------------
    def enforce_retention(self, now_data_time: datetime) -> int:
        """Trim session_events to the rolling hot window."""
        cutoff = now_data_time - timedelta(days=self.retain_days)
        cur = self.conn.cursor()
        cur.execute("DELETE FROM storefront.session_events WHERE event_time < %s", (cutoff,))
        deleted = cur.rowcount
        self.conn.commit()
        cur.close()
        if deleted:
            log.info("retention: removed %s events older than %s", f"{deleted:,}", cutoff.date())
        return deleted

    def current_data_time(self) -> datetime:
        cur = self.conn.cursor()
        cur.execute("SELECT current_data_time FROM storefront.replay_clock WHERE id = 1")
        ts = cur.fetchone()[0]
        cur.close()
        return ts

    # ------------------------------------------------------------------
    def run(self, batch_size: int = 2000, block_ms: int = 2000) -> None:
        log.info("consuming %r as %r (retain %d days)", STREAM, CONSUMER, self.retain_days)
        last_retention = time.time()
        t0 = time.time()

        while _running:
            resp = self.r.xreadgroup(
                GROUP, CONSUMER, {STREAM: ">"}, count=batch_size, block=block_ms
            )
            if not resp:
                continue

            ids, events = [], []
            for _stream, messages in resp:
                for msg_id, fields in messages:
                    ids.append(msg_id)
                    events.append(json.loads(fields[b"payload"]))

            try:
                self.apply_batch(events)
                self.r.xack(STREAM, GROUP, *ids)
            except Exception:
                self.conn.rollback()
                log.exception("batch failed (%d events); messages left unacked", len(events))
                continue

            if time.time() - last_retention > 60:
                try:
                    self.enforce_retention(self.current_data_time())
                except Exception:
                    self.conn.rollback()
                    log.exception("retention pass failed")
                last_retention = time.time()
                rate = self.applied / max(time.time() - t0, 1e-6)
                log.info("applied %s events (%.0f/sec)", f"{self.applied:,}", rate)

        log.info("stopped after applying %s events", f"{self.applied:,}")

    def close(self) -> None:
        self.conn.close()
        self.r.close()


def _ts(v) -> datetime:
    return v if isinstance(v, datetime) else datetime.fromisoformat(str(v))


def _int_or_none(v):
    return None if v is None else int(v)


def main() -> None:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--retain-days", type=int, default=7)
    ap.add_argument("--truncate", action="store_true",
                    help="clear the OLTP hot window first; pair with a fresh replay")
    args = ap.parse_args()

    c = StorefrontConsumer(settings, retain_days=args.retain_days, truncate=args.truncate)
    try:
        c.run(batch_size=args.batch_size)
    finally:
        c.close()


if __name__ == "__main__":
    main()
