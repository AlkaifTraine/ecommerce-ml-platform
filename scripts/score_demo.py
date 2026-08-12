"""Score real sessions through the API and measure latency.

Uses sessions from the committed fixture, including their true outcome, so the
output shows what the model said AND what actually happened - not a synthetic
request that proves only that the HTTP layer works.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request

import duckdb

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)

K = 5
FIXTURE = "tests/fixtures/events"


def post(url: str, payload: dict, timeout: float = 10.0) -> tuple[dict | None, float]:
    body = json.dumps(payload, default=str).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        out = {"error": e.code, "detail": e.read().decode()[:200]}
    return out, (time.perf_counter() - t0) * 1000.0


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8500")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    root = settings.data_root.parent
    con = duckdb.connect()
    src = f"read_parquet('{(root / FIXTURE).as_posix()}/**/*.parquet')"

    # Sessions long enough to score, sampled as a BALANCED mix of outcomes.
    # Sorting by `bought` and taking the top N returns only buyers, which
    # leaves nothing to compare against and makes the separation figure - the
    # only number here that says whether the model is working - undefined.
    half = max(args.n // 2, 1)
    sessions = con.execute(
        f"""
        WITH s AS (
            SELECT user_session,
                   max(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) AS bought,
                   min(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) AS dummy
            FROM {src}
            WHERE user_session IS NOT NULL
            GROUP BY 1
            HAVING count(*) >= {K + 2}
        ),
        ranked AS (
            SELECT *, row_number() OVER (PARTITION BY bought ORDER BY user_session) AS rn
            FROM s
        )
        SELECT user_session, bought FROM ranked WHERE rn <= {half}
        ORDER BY user_session
        """
    ).fetchall()
    if not sessions:
        raise SystemExit("no fixture sessions long enough to score")

    log.info("scoring %d real sessions against %s", len(sessions), args.url)
    latencies: list[float] = []
    shown = 0
    scores_by_outcome: dict[int, list[float]] = {0: [], 1: []}

    for key, bought in sessions:
        rows = con.execute(
            f"""
            SELECT event_time, event_type, product_id, category_id,
                   category_code, brand, price, user_id
            FROM {src} WHERE user_session = ?
            ORDER BY event_time, product_id, event_type
            """,
            [key],
        ).fetchall()

        events = [
            {"event_time": r[0].isoformat(), "event_type": r[1], "product_id": r[2],
             "category_id": r[3], "category_code": r[4], "brand": r[5], "price": r[6]}
            for r in rows[: K + 4]
        ]
        payload = {
            "session_key": key,
            "user_id": int(rows[0][7]),
            "events": events,
        }
        resp, ms = post(f"{args.url}/score", payload)
        if "error" in resp:
            log.warning("%s -> HTTP %s %s", key[:8], resp["error"], resp.get("detail", "")[:80])
            continue

        latencies.append(resp["latency_ms"])
        scores_by_outcome[int(bought)].append(resp["score"])
        if shown < 8:
            log.info("  %s score=%.4f  %-16s actual=%s  (%.1f ms)",
                     key[:8], resp["score"], resp["decision"],
                     "BOUGHT" if bought else "no buy", ms)
            shown += 1

    if not latencies:
        raise SystemExit("no successful scores")

    latencies.sort()
    log.info("=" * 66)
    log.info("LATENCY over %d requests (server-side compute)", len(latencies))
    log.info("  mean %.2f ms   p50 %.2f ms   p95 %.2f ms   p99 %.2f ms   max %.2f ms",
             statistics.mean(latencies),
             latencies[len(latencies) // 2],
             latencies[int(len(latencies) * 0.95)],
             latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)],
             latencies[-1])
    log.info("=" * 66)
    for outcome, label in ((1, "actually BOUGHT"), (0, "did NOT buy")):
        s = scores_by_outcome[outcome]
        if s:
            log.info("  mean score, %-16s : %.4f   (n=%d)", label, statistics.mean(s), len(s))
    if scores_by_outcome[1] and scores_by_outcome[0]:
        sep = statistics.mean(scores_by_outcome[1]) - statistics.mean(scores_by_outcome[0])
        log.info("  separation: %+.4f  %s", sep,
                 "buyers score higher, as expected" if sep > 0 else "WRONG DIRECTION")
    log.info("=" * 66)
    con.close()


if __name__ == "__main__":
    main()
