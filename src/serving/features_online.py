"""Online feature computation - the serving-time twin of the training SQL.

The training set is built in SQL over Parquet; scoring a live session has to
happen in Python over a handful of events in memory. Two implementations of
the same definition is the classic setup for training/serving skew: they start
identical, one gets edited, and the model silently receives different inputs in
production than it learned from. Nothing errors. The metrics just quietly stop
matching.

Rather than hope they stay in step, `tests/test_serving_parity.py` computes
features BOTH ways over the same real sessions and asserts every value matches.
That test is the contract; this module is one side of it.

Every quirk of the SQL is reproduced deliberately, including the ones that look
like mistakes:

* events are deduplicated on (event_time, event_type, product_id) and ordered
  by (event_time, product_id, event_type) - the total order that fixed the
  non-determinism bug
* `n_distinct_brands` ignores NULL brands, matching count(DISTINCT brand)
* `price_std` is the POPULATION standard deviation (stddev_pop, ddof=0)
* `events_per_minute` is NULL - not zero, not infinity - when the prefix spans
  zero seconds, which happens when all k events share a timestamp
* `day_of_week` follows DuckDB's dayofweek(): 0 = Sunday, not Python's Monday
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

FEATURE_ORDER = [
    "n_distinct_products", "n_distinct_categories", "n_distinct_brands",
    "n_repeat_views", "max_views_same_product", "mean_views_per_product",
    "repeat_product_ratio",
    "prefix_duration_sec", "mean_gap_sec", "median_gap_sec", "min_gap_sec",
    "max_gap_sec", "events_per_minute",
    "price_mean", "price_min", "price_max", "price_std", "price_range",
    "price_at_cutoff",
    "null_brand_ratio", "null_category_ratio",
    "n_cart_events", "cart_ratio",
    "hour_of_day", "day_of_week", "is_weekend",
    "user_prior_sessions", "user_prior_purchases", "user_prior_conv_rate",
    "user_prior_revenue", "hours_since_last_session", "is_new_user",
]


@dataclass
class UserHistory:
    """Point-in-time history, looked up from the online store at serving time.

    Must reflect only sessions that ENDED before the current one started -
    the same constraint the training-time window functions enforce.
    """

    prior_sessions: int = 0
    prior_purchases: int = 0
    prior_revenue: float = 0.0
    hours_since_last_session: float | None = None


def _duckdb_dayofweek(ts: datetime) -> int:
    """DuckDB dayofweek(): Sunday = 0. Python weekday(): Monday = 0."""
    return (ts.weekday() + 1) % 7


def _median(xs: Sequence[float]) -> float:
    # DuckDB's median averages the two middle values on an even count, which is
    # what statistics.median does.
    return float(statistics.median(xs))


def prepare_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate and apply the canonical total order."""
    seen: dict[tuple, dict] = {}
    for e in events:
        key = (e["event_time"], e["event_type"], e["product_id"])
        # keep the first occurrence; the SQL keeps min(price) over the group,
        # and exact duplicates carry identical payloads by definition
        if key not in seen or (e.get("price") or 0) < (seen[key].get("price") or 0):
            seen[key] = e
    return sorted(
        seen.values(),
        key=lambda e: (e["event_time"], e["product_id"], e["event_type"]),
    )


def compute(
    events: list[dict[str, Any]],
    k: int,
    history: UserHistory | None = None,
    session_start: datetime | None = None,
    hard_mode: bool = False,
) -> dict[str, float | None]:
    """Features from the first k events. Raises if the session is too short."""
    ordered = prepare_events(events)
    if len(ordered) < k:
        raise ValueError(f"session has {len(ordered)} distinct events, need {k}")
    prefix = ordered[:k]

    if any(e["event_type"] == "purchase" for e in prefix):
        raise ValueError("session already purchased within the prefix - nothing to predict")

    history = history or UserHistory()
    session_start = session_start or prefix[0]["event_time"]

    prices = [float(e["price"]) for e in prefix if e.get("price") is not None]
    times = [e["event_time"] for e in prefix]

    per_product: dict[Any, int] = {}
    for e in prefix:
        per_product[e["product_id"]] = per_product.get(e["product_id"], 0) + 1

    n_products = len(per_product)
    categories = {e.get("category_id") for e in prefix}
    brands = {e.get("brand") for e in prefix if e.get("brand") is not None}

    duration = (times[-1] - times[0]).total_seconds()
    gaps = [
        (times[i] - times[i - 1]).total_seconds() for i in range(1, len(times))
    ]

    n_cart = sum(1 for e in prefix if e["event_type"] == "cart")

    f: dict[str, float | None] = {
        "n_distinct_products": n_products,
        "n_distinct_categories": len(categories),
        "n_distinct_brands": len(brands),
        "n_repeat_views": k - n_products,
        "max_views_same_product": max(per_product.values()),
        "mean_views_per_product": sum(per_product.values()) / n_products,
        "repeat_product_ratio": 1.0 - (n_products / k),

        "prefix_duration_sec": duration,
        "mean_gap_sec": (sum(gaps) / len(gaps)) if gaps else None,
        "median_gap_sec": _median(gaps) if gaps else None,
        "min_gap_sec": min(gaps) if gaps else None,
        "max_gap_sec": max(gaps) if gaps else None,
        # NULL rather than 0 or inf when all k events share one timestamp
        "events_per_minute": (k * 60.0 / duration) if duration > 0 else None,

        "price_mean": (sum(prices) / len(prices)) if prices else None,
        "price_min": min(prices) if prices else None,
        "price_max": max(prices) if prices else None,
        "price_std": (statistics.pstdev(prices) if len(prices) > 1 else 0.0),
        "price_range": (max(prices) - min(prices)) if prices else None,
        "price_at_cutoff": float(prefix[-1]["price"]) if prefix[-1].get("price") is not None else None,

        "null_brand_ratio": sum(1 for e in prefix if e.get("brand") is None) / k,
        "null_category_ratio": sum(1 for e in prefix if e.get("category_code") is None) / k,

        "n_cart_events": 0 if hard_mode else n_cart,
        "cart_ratio": 0.0 if hard_mode else (n_cart / k),

        "hour_of_day": session_start.hour,
        "day_of_week": _duckdb_dayofweek(session_start),
        "is_weekend": 1 if _duckdb_dayofweek(session_start) in (0, 6) else 0,

        "user_prior_sessions": history.prior_sessions,
        "user_prior_purchases": history.prior_purchases,
        "user_prior_conv_rate": (
            history.prior_purchases / history.prior_sessions
            if history.prior_sessions > 0 else None
        ),
        "user_prior_revenue": history.prior_revenue,
        "hours_since_last_session": history.hours_since_last_session,
        "is_new_user": 1 if history.prior_sessions == 0 else 0,
    }
    return f


def to_vector(features: dict[str, float | None]) -> list[float]:
    """Feature dict -> model input, in the exact training column order.

    None becomes NaN rather than 0: LightGBM handles missing values natively,
    and substituting zero would put "no previous session" and "converted at 0%"
    in the same bucket.
    """
    return [
        float("nan") if features.get(name) is None else float(features[name])
        for name in FEATURE_ORDER
    ]
