"""Verify the feature SQL computes what the docstring claims.

Every expected value here was worked out by hand from the fixture in
conftest.py. If the SQL is rewritten for performance later, these numbers are
what stop the rewrite from silently changing the definition of the task.
"""

from __future__ import annotations

import polars as pl
import pytest

K = 5


@pytest.fixture(scope="module")
def train(handmade_pipeline):
    path = handmade_pipeline.features_dir / f"train_k{K}.parquet"
    assert path.exists(), f"feature build produced no output at {path}"
    return pl.read_parquet(path)


def _row(df: pl.DataFrame, key: str) -> dict:
    sub = df.filter(pl.col("session_key") == key)
    assert sub.height == 1, f"expected exactly one row for session {key}, got {sub.height}"
    return sub.to_dicts()[0]


def test_only_eligible_sessions_survive(train):
    """C is too short; D purchased inside the prefix. Both must be dropped."""
    keys = set(train["session_key"].to_list())
    assert keys == {"A", "B"}, f"unexpected eligible set: {sorted(keys)}"


def test_session_a_hand_computed_features(train):
    a = _row(train, "A")
    # prefix is rn1..rn5: P1,P2,P1,P3,P1
    assert a["n_distinct_products"] == 3
    assert a["max_views_same_product"] == 3      # P1 seen three times
    assert a["n_repeat_views"] == 2              # k - distinct = 5 - 3
    assert a["repeat_product_ratio"] == pytest.approx(1 - 3 / K)
    assert a["prefix_duration_sec"] == 120       # t+120 minus t+0
    assert a["mean_gap_sec"] == pytest.approx(30.0)
    assert a["events_per_minute"] == pytest.approx(K * 60 / 120)
    # prices 100,200,100,300,100
    assert a["price_mean"] == pytest.approx(160.0)
    assert a["price_min"] == pytest.approx(100.0)
    assert a["price_max"] == pytest.approx(300.0)
    assert a["price_range"] == pytest.approx(200.0)
    assert a["price_at_cutoff"] == pytest.approx(100.0)
    # the purchase at rn7 is after the cutoff, so the label is positive
    assert a["y"] == 1


def test_post_cutoff_cart_is_invisible(train):
    """Session A carts at rn6 - AFTER the k=5 cutoff.

    If that cart reached the features the model would be reading the future,
    and since cart is nearly the label it would look brilliant and be useless.
    """
    a = _row(train, "A")
    assert a["n_cart_events"] == 0, "a post-cutoff cart event leaked into the features"


def test_session_b_hand_computed_features(train):
    b = _row(train, "B")
    assert b["n_distinct_products"] == 5
    assert b["max_views_same_product"] == 1
    assert b["n_repeat_views"] == 0
    assert b["prefix_duration_sec"] == 40
    assert b["mean_gap_sec"] == pytest.approx(10.0)
    assert b["price_mean"] == pytest.approx(70.0)
    assert b["price_range"] == pytest.approx(40.0)
    assert b["y"] == 0


def test_first_time_users_flagged(train):
    for key in ("A", "B"):
        row = _row(train, key)
        assert row["user_prior_sessions"] == 0
        assert row["is_new_user"] == 1
        assert row["user_prior_purchases"] == 0


def test_no_nulls_in_core_features(train):
    """Nulls here would silently become LightGBM's default direction."""
    core = [
        "n_distinct_products", "n_distinct_categories", "max_views_same_product",
        "prefix_duration_sec", "price_mean", "price_min", "price_max", "y",
    ]
    nulls = {c: train[c].null_count() for c in core if train[c].null_count() > 0}
    assert not nulls, f"unexpected nulls: {nulls}"


def test_feature_build_is_deterministic(real_pipeline):
    """Rebuilding must produce byte-identical labels and features.

    Regression test for a real bug: the event ordering key was
    (event_time, product_id), which is NOT unique - timestamps are
    second-granular and ~0.2% of rows are exact duplicates. row_number() over a
    tied key is arbitrary, so successive runs ranked boundary events
    differently and the training set silently changed between builds. The
    leakage audit caught it as one no_prepurchase violation in 5,153,372
    sessions.
    """
    from src.features import build_features

    path = real_pipeline.features_dir / f"train_k{K}.parquet"
    first = pl.read_parquet(path).sort("session_key")

    build_features.build(real_pipeline, k=K)
    second = pl.read_parquet(path).sort("session_key")

    assert first.height == second.height, (
        f"row count changed between builds: {first.height} -> {second.height}"
    )
    assert first["session_key"].to_list() == second["session_key"].to_list(), (
        "the set of eligible sessions changed between identical builds"
    )
    assert first["y"].to_list() == second["y"].to_list(), (
        "labels changed between identical builds - event ordering is not total"
    )
    for col in ("n_distinct_products", "max_views_same_product", "price_at_cutoff",
                "cutoff_time", "n_cart_events"):
        assert first[col].to_list() == second[col].to_list(), (
            f"feature {col} changed between identical builds"
        )


def test_real_slice_positive_rate_is_plausible(real_pipeline):
    """The real fixture must land near the archive-wide base rate.

    Measured on the full archive: 6.10% of sessions purchase, and ~7.0% of
    k=5-eligible sessions do. A fixture far outside that band means the slice
    is unrepresentative and any test built on it is misleading.
    """
    path = real_pipeline.features_dir / f"train_k{K}.parquet"
    df = pl.read_parquet(path)
    assert df.height > 500, f"too few eligible sessions to test with: {df.height}"
    rate = float(df["y"].mean())
    assert 0.02 < rate < 0.20, f"positive rate {rate:.4f} is far from the archive's ~7%"
