"""Test fixtures.

There is no synthetic data in this project. Two fixtures, both real in the
sense that matters:

* `handmade_pipeline` - 21 events written out by hand in this file, with every
  expected feature value computed by hand in test_features.py. This is a unit
  test of the arithmetic, not a stand-in for the dataset.

* `real_pipeline` - an actual slice of the REES46 archive, carved by
  scripts/make_test_fixture.py and committed under tests/fixtures/. Whole
  sessions only, selected by a stable hash so the sample is reproducible.

An earlier version generated a few thousand fake sessions instead. It was
removed: generated data encodes the generator author's assumptions, which is
precisely what tests are supposed to catch. That generator gave browsers and
buyers barely-overlapping click gaps, so one feature separated them perfectly
and the fixture could no longer detect a bug anywhere else.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest

K = 5
FIXTURE_EVENTS = Path(__file__).resolve().parent / "fixtures" / "events"


def _fresh_settings(root: Path):
    """Build a Settings pointed at `root`, bypassing the module-level cache."""
    os.environ["DATA_ROOT"] = str(root).replace("\\", "/")
    os.environ["DUCKDB_TEMP_DIR"] = str(root / ".duckdb_tmp").replace("\\", "/")
    os.environ["DUCKDB_MEMORY_LIMIT"] = "2GB"

    from src.platform_core import config as config_mod

    config_mod.get_settings.cache_clear()
    return config_mod.get_settings()


def _build(settings, k: int = K) -> None:
    from src.features import build_features, sessionize

    sessionize.build(settings)
    build_features.build(settings, k=k)


@pytest.fixture(scope="session")
def handmade_pipeline(tmp_path_factory):
    """Hand-written events whose correct feature values are known by hand.

    Session A (eligible, y=1)
        rn1 view P1 t+0      rn2 view P2 t+30    rn3 view P1 t+60
        rn4 view P3 t+90     rn5 view P1 t+120   rn6 cart P1 t+150
        rn7 purchase P1 t+200
        -> prefix = rn1..rn5: 3 distinct products, P1 seen 3x,
           2 repeat views, 120s duration, purchase after cutoff => y=1

    Session B (eligible, y=0): 5 distinct views, nothing else
    Session C (excluded): only 3 events, never reaches k=5
    Session D (excluded): purchases at rn3, i.e. inside the prefix
    """
    import polars as pl

    root = tmp_path_factory.mktemp("handmade")
    settings = _fresh_settings(root)
    t0 = datetime(2019, 10, 1, 9, 0, 0)

    def ev(sess, user, offset, etype, pid, price):
        return dict(
            event_time=t0 + timedelta(seconds=offset),
            event_type=etype,
            product_id=pid,
            category_id=100 + (pid % 3),
            category_code="electronics.smartphone",
            brand="samsung",
            price=price,
            user_id=user,
            user_session=sess,
        )

    rows = [
        ev("A", 1, 0, "view", 1, 100.0),
        ev("A", 1, 30, "view", 2, 200.0),
        ev("A", 1, 60, "view", 1, 100.0),
        ev("A", 1, 90, "view", 3, 300.0),
        ev("A", 1, 120, "view", 1, 100.0),
        ev("A", 1, 150, "cart", 1, 100.0),
        ev("A", 1, 200, "purchase", 1, 100.0),
        ev("B", 2, 0, "view", 10, 50.0),
        ev("B", 2, 10, "view", 11, 60.0),
        ev("B", 2, 20, "view", 12, 70.0),
        ev("B", 2, 30, "view", 13, 80.0),
        ev("B", 2, 40, "view", 14, 90.0),
        ev("C", 3, 0, "view", 20, 20.0),
        ev("C", 3, 5, "view", 21, 25.0),
        ev("C", 3, 9, "view", 22, 30.0),
        ev("D", 4, 0, "view", 30, 15.0),
        ev("D", 4, 5, "view", 31, 18.0),
        ev("D", 4, 10, "purchase", 31, 18.0),
        ev("D", 4, 20, "view", 32, 22.0),
        ev("D", 4, 30, "view", 33, 26.0),
        ev("D", 4, 40, "view", 34, 28.0),
    ]

    df = pl.DataFrame(rows).with_columns(pl.col("event_time").dt.date().alias("event_date"))
    out = settings.events_dir
    out.mkdir(parents=True, exist_ok=True)
    for (d,), part in df.group_by(["event_date"]):
        pdir = out / f"event_date={d}"
        pdir.mkdir(parents=True, exist_ok=True)
        part.drop("event_date").write_parquet(pdir / "part_0.parquet")

    _build(settings, k=K)
    return settings


@pytest.fixture(scope="session")
def real_pipeline(tmp_path_factory):
    """A committed slice of the real archive, run through the full pipeline."""
    if not FIXTURE_EVENTS.exists() or not any(FIXTURE_EVENTS.rglob("*.parquet")):
        pytest.skip(
            f"no real fixture at {FIXTURE_EVENTS}. Build it once with:\n"
            "    python -m scripts.make_test_fixture"
        )

    root = tmp_path_factory.mktemp("real")
    settings = _fresh_settings(root)
    settings.events_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FIXTURE_EVENTS, settings.events_dir)
    _build(settings, k=K)
    return settings
