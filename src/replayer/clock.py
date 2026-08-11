"""The replay clock - the single source of truth for "what time is it?".

Every component asks the clock, never the filesystem. That indirection is the
entire basis of the no-peeking guarantee: the raw Parquet contains seven months
of the future, but nothing downstream is permitted to read past
`current_data_time`.

The authoritative clock lives in one Postgres row so that separate processes
(replayer, scoring API, Airflow tasks) cannot drift apart. `InMemoryClock`
exists for tests and offline runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from src.platform_core import get_logger

log = get_logger(__name__)


class ClockError(RuntimeError):
    """Raised when something tries to read beyond the current data time."""


class BaseClock(ABC):
    @abstractmethod
    def now(self) -> datetime:
        """Current data time."""

    @abstractmethod
    def advance_to(self, ts: datetime, events_emitted: int = 0) -> None:
        ...

    @abstractmethod
    def set_phase(self, phase: str) -> None:
        ...

    def assert_visible(self, ts: datetime) -> None:
        """Guard used by readers. Anything at or after `now()` is the future."""
        current = self.now()
        if ts >= current:
            raise ClockError(
                f"attempted to read data at {ts} but the clock is at {current}; "
                "this would leak future information"
            )

    def bound(self) -> str:
        """Exclusive upper bound for SQL filters, as an ISO string."""
        return self.now().isoformat(sep=" ")


class InMemoryClock(BaseClock):
    def __init__(self, start: datetime):
        self._t = start
        self._phase = "stopped"
        self.events_emitted = 0

    def now(self) -> datetime:
        return self._t

    def advance_to(self, ts: datetime, events_emitted: int = 0) -> None:
        if ts < self._t:
            raise ClockError(f"clock cannot run backwards: {self._t} -> {ts}")
        self._t = ts
        self.events_emitted += events_emitted

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    @property
    def phase(self) -> str:
        return self._phase


class PostgresClock(BaseClock):
    """Clock persisted in storefront.replay_clock (single row, id = 1)."""

    def __init__(self, dsn: str):
        import psycopg2

        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = True

    def now(self) -> datetime:
        with self._conn.cursor() as cur:
            cur.execute("SELECT current_data_time FROM storefront.replay_clock WHERE id = 1")
            row = cur.fetchone()
        if row is None:
            raise ClockError("replay_clock row is missing; run init.sql")
        return row[0]

    def advance_to(self, ts: datetime, events_emitted: int = 0) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE storefront.replay_clock
                   SET current_data_time = GREATEST(current_data_time, %s),
                       events_emitted    = events_emitted + %s,
                       updated_at        = now()
                 WHERE id = 1
                """,
                (ts, events_emitted),
            )

    def set_phase(self, phase: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE storefront.replay_clock SET phase = %s, updated_at = now() WHERE id = 1",
                (phase,),
            )

    def reset(self, start: datetime, speed: float) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE storefront.replay_clock
                   SET current_data_time = %s, phase = 'stopped',
                       speed_multiplier = %s, events_emitted = 0,
                       started_at = now(), updated_at = now()
                 WHERE id = 1
                """,
                (start, speed),
            )

    @property
    def phase(self) -> str:
        with self._conn.cursor() as cur:
            cur.execute("SELECT phase FROM storefront.replay_clock WHERE id = 1")
            return cur.fetchone()[0]

    def close(self) -> None:
        self._conn.close()


def wall_to_data(elapsed_wall_sec: float, speed: float) -> timedelta:
    """Convert elapsed wall-clock seconds into simulated data time."""
    return timedelta(seconds=elapsed_wall_sec * speed)
