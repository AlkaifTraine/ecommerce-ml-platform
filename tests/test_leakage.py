"""Leakage guarantees, plus proof that the audit would actually catch a breach.

A test suite that only ever sees clean data proves nothing. `test_audit_has_teeth`
deliberately corrupts the training table and asserts the audit fails, so we know
the passing results above are meaningful.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.features.leakage_audit import audit, audit_splits

K = 5


def test_audit_passes_on_handmade(handmade_pipeline):
    res = audit(handmade_pipeline, k=K)
    assert res.passed, f"leakage audit failed:\n{res.report()}"


def test_audit_passes_on_real_slice(real_pipeline):
    res = audit(real_pipeline, k=K)
    assert res.passed, f"leakage audit failed:\n{res.report()}"


def test_every_check_actually_ran(handmade_pipeline):
    """Guard against a check silently disappearing from the audit."""
    res = audit(handmade_pipeline, k=K)
    expected = {"prefix_size", "no_prepurchase", "label_provenance", "no_forbidden_cols"}
    assert expected.issubset(res.checks.keys()), f"missing checks: {expected - set(res.checks)}"


def test_audit_has_teeth(real_pipeline, tmp_path):
    """Flip a single label and confirm the audit notices.

    This is the control experiment for the whole suite.
    """
    path = real_pipeline.features_dir / f"train_k{K}.parquet"
    original = pl.read_parquet(path)
    backup = tmp_path / "backup.parquet"
    original.write_parquet(backup)

    try:
        corrupted = original.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(1 - pl.col("y"))
            .otherwise(pl.col("y"))
            .alias("y")
        )
        corrupted.write_parquet(path)

        res = audit(real_pipeline, k=K)
        assert not res.passed, "audit passed on deliberately corrupted labels - it has no teeth"
        assert res.checks["label_provenance"] > 0, (
            f"expected label_provenance to fire, got: {res.checks}"
        )
    finally:
        pl.read_parquet(backup).write_parquet(path)

    # and it must go back to passing once restored
    assert audit(real_pipeline, k=K).passed


def test_clock_bound_is_enforced(real_pipeline):
    """Rebuild with a clock cutoff; nothing at or after it may appear."""
    from src.features import build_features, sessionize

    # must fall inside the fixture's own date range, else the build is empty
    cutoff = "2019-11-07T00:00:00"
    sessionize.build(real_pipeline, until=cutoff)
    build_features.build(real_pipeline, k=K, until=cutoff)

    res = audit(real_pipeline, k=K, until=cutoff)
    assert res.checks.get("no_future_events") == 0, f"clock breach:\n{res.report()}"

    df = pl.read_parquet(real_pipeline.features_dir / f"train_k{K}.parquet")
    assert df.height > 0, "clock-bounded build produced nothing to test"
    latest = df["cutoff_time"].max()
    assert str(latest) < cutoff.replace("T", " "), f"row at {latest} is beyond the clock"

    # restore the unbounded build for any later test
    sessionize.build(real_pipeline)
    build_features.build(real_pipeline, k=K)


def test_split_windows_do_not_overlap(real_pipeline):
    from src.models.train import chronological_split

    df = pl.read_parquet(real_pipeline.features_dir / f"train_k{K}.parquet")
    splits = chronological_split(df)
    res = audit_splits(splits)
    assert res.passed, f"split windows overlap:\n{res.report()}"

    order = ["train", "valid", "test", "drift"]
    present = [s for s in order if splits[s].height > 0]
    for a, b in zip(present, present[1:]):
        assert splits[a]["session_date"].max() < splits[b]["session_date"].min()


def test_identifiers_never_reach_the_model(real_pipeline):
    from src.models.train import ID_COLS, LABEL, feature_columns

    df = pl.read_parquet(real_pipeline.features_dir / f"train_k{K}.parquet")
    feats = feature_columns(df)
    for banned in ID_COLS + [LABEL]:
        assert banned not in feats, f"{banned} reached the feature matrix"
    assert len(feats) > 10, f"suspiciously few features: {feats}"
