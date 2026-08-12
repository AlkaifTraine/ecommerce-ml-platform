"""Training/serving parity: the SQL and the Python must agree exactly.

Training features are built in SQL over Parquet. Scoring a live session happens
in Python over events in memory. Two implementations of one definition drift
apart the moment somebody edits one of them, and the failure is silent - no
exception, no error, just a model receiving different inputs in production than
it was trained on while the offline metrics keep looking fine.

This test computes features BOTH ways over the same real sessions and asserts
every value matches. It is the reason it is safe to have two implementations at
all.
"""

from __future__ import annotations

import math

import duckdb
import polars as pl
import pytest

from src.serving.features_online import UserHistory, compute

K = 5
SAMPLE = 200

# Columns the online path produces; identifiers and the label are excluded.
COMPARED = [
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
    "user_prior_revenue", "is_new_user",
]


@pytest.fixture(scope="module")
def parity_data(real_pipeline):
    """SQL features, raw events and session history for a sample of sessions."""
    settings = real_pipeline
    sql_feats = pl.read_parquet(settings.features_dir / f"train_k{K}.parquet")
    assert sql_feats.height > 0, "no training rows to compare against"

    keys = sql_feats["session_key"].to_list()[:SAMPLE]
    sql_feats = sql_feats.filter(pl.col("session_key").is_in(keys))

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    events = con.execute(
        f"""
        SELECT user_session AS session_key, event_time, event_type, product_id,
               category_id, category_code, brand, price
        FROM read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')
        WHERE user_session IN ({",".join(repr(k) for k in keys)})
        """
    ).pl()
    sessions = con.execute(
        f"""
        SELECT session_key, session_start, user_prior_sessions,
               user_prior_purchases, user_prior_revenue
        FROM read_parquet('{(settings.features_dir / 'sessions.parquet').as_posix()}')
        WHERE session_key IN ({",".join(repr(k) for k in keys)})
        """
    ).pl()
    con.close()
    return sql_feats, events, sessions


def test_online_features_match_training_sql(parity_data):
    sql_feats, events, sessions = parity_data

    by_session = {k: g for k, g in events.group_by("session_key")}
    sess_meta = {r["session_key"]: r for r in sessions.to_dicts()}

    compared = 0
    mismatches: list[str] = []

    for row in sql_feats.to_dicts():
        key = row["session_key"]
        grp = by_session.get((key,)) if (key,) in by_session else by_session.get(key)
        if grp is None:
            continue
        meta = sess_meta[key]

        online = compute(
            events=grp.to_dicts(),
            k=K,
            history=UserHistory(
                prior_sessions=int(meta["user_prior_sessions"]),
                prior_purchases=int(meta["user_prior_purchases"]),
                prior_revenue=float(meta["user_prior_revenue"]),
            ),
            session_start=meta["session_start"],
        )
        compared += 1

        for col in COMPARED:
            sql_v, on_v = row.get(col), online.get(col)
            if sql_v is None and on_v is None:
                continue
            if sql_v is None or on_v is None:
                mismatches.append(f"{key}.{col}: sql={sql_v} online={on_v}")
                continue
            if not math.isclose(float(sql_v), float(on_v), rel_tol=1e-9, abs_tol=1e-9):
                mismatches.append(f"{key}.{col}: sql={sql_v} online={on_v}")

    assert compared >= 50, f"only compared {compared} sessions - fixture too small"
    assert not mismatches, (
        f"training/serving skew in {len(mismatches)} value(s):\n  "
        + "\n  ".join(mismatches[:25])
    )


def test_online_rejects_short_sessions(real_pipeline):
    from datetime import datetime

    events = [
        {"event_time": datetime(2019, 11, 5, 10, 0, i), "event_type": "view",
         "product_id": i, "category_id": 1, "category_code": "a.b",
         "brand": "x", "price": 10.0}
        for i in range(K - 1)
    ]
    with pytest.raises(ValueError, match="need"):
        compute(events, k=K)


def test_online_rejects_already_purchased(real_pipeline):
    from datetime import datetime

    events = [
        {"event_time": datetime(2019, 11, 5, 10, 0, i), "event_type": "view",
         "product_id": i, "category_id": 1, "category_code": "a.b",
         "brand": "x", "price": 10.0}
        for i in range(K)
    ]
    events[2]["event_type"] = "purchase"
    with pytest.raises(ValueError, match="already purchased"):
        compute(events, k=K)


def test_vector_order_matches_training(real_pipeline):
    """The model consumes a positional vector; column order is load-bearing."""
    from src.models.train import ID_COLS, LABEL, feature_columns
    from src.serving.features_online import FEATURE_ORDER

    df = pl.read_parquet(real_pipeline.features_dir / f"train_k{K}.parquet")
    training_cols = feature_columns(df)

    assert set(FEATURE_ORDER) == set(training_cols), (
        "online feature set differs from training:\n"
        f"  only online:   {sorted(set(FEATURE_ORDER) - set(training_cols))}\n"
        f"  only training: {sorted(set(training_cols) - set(FEATURE_ORDER))}"
    )
    assert FEATURE_ORDER == training_cols, (
        "feature ORDER differs - the model would receive values in the wrong "
        f"positions.\n  training: {training_cols}\n  online:   {FEATURE_ORDER}"
    )
    for banned in ID_COLS + [LABEL]:
        assert banned not in FEATURE_ORDER
