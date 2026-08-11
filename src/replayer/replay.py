"""Replay engine: turns a static Parquet archive into a live event stream.

The archive holds seven months of history all at once. This process is the only
component permitted to see it, and it releases events strictly in event_time
order at a controlled rate. Everything else in the platform observes only what
has been released, so from their point of view data genuinely arrives over time
and the future does not exist yet.

Two phases, mirroring how a real system is brought up:

  backfill  bulk-load history quickly to bootstrap the stores and train an
            initial model - "we just launched, load what we have"
  live      advance the clock at `speed` x real time so that drift events
            (Black Friday, the January reversion, COVID) play out at a pace
            slow enough for the retraining loop to actually react

Example - the full arc in roughly 30 hours of wall time, one data week per
wall hour:

    python -m src.replayer.replay --reset \
        --backfill-until 2019-11-20 --speed 168 --sink redis
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta

import duckdb

from src.platform_core import get_logger, get_settings
from src.replayer.clock import BaseClock, InMemoryClock, PostgresClock
from src.replayer.sinks import EventSink, build_sink

log = get_logger(__name__)

EMIT_COLUMNS = [
    "event_time", "event_type", "product_id", "category_id",
    "category_code", "brand", "price", "user_id", "user_session",
]


def archive_bounds(settings) -> tuple[datetime, datetime]:
    """First and last event_time in the archive, without holding a connection."""
    con = duckdb.connect()
    try:
        lo, hi = con.execute(
            f"SELECT min(event_time), max(event_time) "
            f"FROM read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')"
        ).fetchone()
    finally:
        con.close()
    if lo is None:
        raise SystemExit(f"no events found under {settings.events_dir}")
    return lo, hi


class ReplayEngine:
    def __init__(
        self,
        settings,
        sink: EventSink,
        clock: BaseClock,
        speed: float,
        backfill_until: datetime | None,
        batch_seconds: float = 2.0,
    ):
        self.settings = settings
        self.sink = sink
        self.clock = clock
        self.speed = speed
        self.backfill_until = backfill_until
        self.batch_seconds = batch_seconds

        self._con = duckdb.connect()
        self._con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
        self._con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
        self._src = f"read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')"

    # -- windowed reads

    def _read_window(self, start: datetime, end: datetime) -> list[dict]:
        cols = ", ".join(EMIT_COLUMNS)
        rows = self._con.execute(
            f"""
            SELECT {cols} FROM {self._src}
            WHERE event_time >= ? AND event_time < ?
            ORDER BY event_time, user_session
            """,
            [start, end],
        ).fetchall()
        return [dict(zip(EMIT_COLUMNS, r)) for r in rows]

    def _emit_range(self, start: datetime, end: datetime, chunk: timedelta) -> int:
        """Emit [start, end) in chunks so memory stays flat regardless of span."""
        total = 0
        cursor = start
        while cursor < end:
            stop = min(cursor + chunk, end)
            batch = self._read_window(cursor, stop)
            if batch:
                total += self.sink.emit(batch)
            self.clock.advance_to(stop, events_emitted=len(batch))
            cursor = stop
        return total

    # -- phases

    def backfill(self, start: datetime, until: datetime) -> int:
        log.info("BACKFILL %s -> %s (as fast as the disk allows)", start, until)
        self.clock.set_phase("backfill")
        t0 = time.time()
        n = self._emit_range(start, until, chunk=timedelta(days=1))
        dt = time.time() - t0
        log.info("backfill emitted %s events in %.1fs (%.0f events/sec)",
                 f"{n:,}", dt, n / max(dt, 1e-6))
        return n

    def live(self, until: datetime, max_wall_seconds: float | None = None) -> int:
        log.info("LIVE replay -> %s at %.0fx real time", until, self.speed)
        log.info("  (1 wall second = %.1f data hours)", self.speed / 3600)
        self.clock.set_phase("live")

        total = 0
        wall_start = time.time()
        last_report = wall_start

        while True:
            current = self.clock.now()
            if current >= until:
                log.info("reached end of replay window")
                break
            if max_wall_seconds and (time.time() - wall_start) > max_wall_seconds:
                log.info("hit wall-clock budget of %.0fs, stopping", max_wall_seconds)
                break

            time.sleep(self.batch_seconds)
            advance = timedelta(seconds=self.batch_seconds * self.speed)
            target = min(current + advance, until)

            batch = self._read_window(current, target)
            if batch:
                total += self.sink.emit(batch)
            self.clock.advance_to(target, events_emitted=len(batch))

            if time.time() - last_report >= 30:
                elapsed = time.time() - wall_start
                log.info(
                    "  data time %s | emitted %s | %.0f events/sec wall",
                    target.strftime("%Y-%m-%d %H:%M"), f"{total:,}", total / max(elapsed, 1e-6),
                )
                last_report = time.time()

        self.clock.set_phase("stopped")
        return total

    def close(self) -> None:
        self._con.close()
        self.sink.close()


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speed", type=float, default=None,
                    help="data seconds per wall second (default from config)")
    ap.add_argument("--backfill-until", default=None,
                    help="bulk-load everything before this timestamp, e.g. 2019-11-20")
    ap.add_argument("--until", default=None, help="stop replay at this data timestamp")
    ap.add_argument("--sink", default="redis", choices=["redis", "kafka", "null"])
    ap.add_argument("--clock", default="postgres", choices=["postgres", "memory"])
    ap.add_argument("--reset", action="store_true", help="restart the clock from the beginning")
    ap.add_argument("--max-wall-seconds", type=float, default=None,
                    help="safety budget for the live phase")
    args = ap.parse_args()

    speed = args.speed if args.speed is not None else settings.replay_speed
    sink = build_sink(args.sink, settings)

    lo, hi = archive_bounds(settings)
    log.info("archive spans %s -> %s", lo, hi)

    if args.clock == "postgres":
        clock = PostgresClock(settings.postgres_dsn)
        if args.reset:
            clock.reset(lo, speed)
            log.info("clock reset to %s", lo)
    else:
        clock = InMemoryClock(lo)

    engine = ReplayEngine(
        settings, sink, clock, speed,
        backfill_until=datetime.fromisoformat(args.backfill_until) if args.backfill_until else None,
    )

    until = datetime.fromisoformat(args.until) if args.until else hi + timedelta(seconds=1)

    try:
        start = clock.now()
        if engine.backfill_until and start < engine.backfill_until:
            engine.backfill(start, min(engine.backfill_until, until))
        engine.live(until, max_wall_seconds=args.max_wall_seconds)
    except KeyboardInterrupt:
        log.info("interrupted; clock left at %s", clock.now())
        clock.set_phase("stopped")
    finally:
        engine.close()

    log.info("replay finished at data time %s", clock.now())


if __name__ == "__main__":
    main()
