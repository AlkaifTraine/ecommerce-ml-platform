"""Where replayed events go.

The sink is an interface with three implementations so the architecture can be
demonstrated at different weights without rewriting the producer:

  redis   - Redis Streams. Default. A real append-only log with consumer
            groups and offsets; maps to Kinesis on AWS.
  kafka   - Apache Kafka. Enabled via the `kafka` docker-compose profile when
            we want the genuine article for the demo.
  null    - counts events and discards them. Used for benchmarking the reader
            in isolation from the transport.

The producer does not know or care which is active.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Iterable

from src.platform_core import get_logger

log = get_logger(__name__)

STREAM = "events.raw"


class EventSink(ABC):
    @abstractmethod
    def emit(self, events: Iterable[dict[str, Any]]) -> int:
        """Publish a batch; return the number accepted."""

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class NullSink(EventSink):
    def __init__(self) -> None:
        self.count = 0

    def emit(self, events: Iterable[dict[str, Any]]) -> int:
        n = sum(1 for _ in events)
        self.count += n
        return n


class RedisStreamSink(EventSink):
    """Append to a Redis Stream, capped so a long replay cannot exhaust RAM.

    MAXLEN is approximate (`~`) because exact trimming forces Redis to do far
    more work per write for no benefit here.
    """

    def __init__(self, url: str, stream: str = STREAM, maxlen: int = 2_000_000):
        import redis

        self._r = redis.Redis.from_url(url)
        self._stream = stream
        self._maxlen = maxlen

    def emit(self, events: Iterable[dict[str, Any]]) -> int:
        pipe = self._r.pipeline(transaction=False)
        n = 0
        for e in events:
            pipe.xadd(
                self._stream,
                {"payload": json.dumps(e, default=str)},
                maxlen=self._maxlen,
                approximate=True,
            )
            n += 1
        if n:
            pipe.execute()
        return n

    def close(self) -> None:
        self._r.close()


class KafkaSink(EventSink):
    def __init__(self, bootstrap: str, topic: str = STREAM):
        from kafka import KafkaProducer  # type: ignore[import-not-found]

        self._p = KafkaProducer(
            bootstrap_servers=bootstrap,
            value_serializer=lambda v: json.dumps(v, default=str).encode(),
            linger_ms=50,
            acks=1,
        )
        self._topic = topic

    def emit(self, events: Iterable[dict[str, Any]]) -> int:
        n = 0
        for e in events:
            # partition by session so a session's events stay ordered
            self._p.send(self._topic, key=str(e.get("user_session", "")).encode(), value=e)
            n += 1
        return n

    def close(self) -> None:
        self._p.flush()
        self._p.close()


def build_sink(kind: str, settings) -> EventSink:
    kind = kind.lower()
    if kind == "redis":
        return RedisStreamSink(settings.redis_url)
    if kind == "kafka":
        return KafkaSink(getattr(settings, "kafka_bootstrap", "localhost:9092"))
    if kind == "null":
        return NullSink()
    raise SystemExit(f"unknown sink {kind!r} (expected redis, kafka or null)")
