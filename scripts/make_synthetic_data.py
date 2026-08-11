"""Generate a small synthetic clickstream with the REAL dataset's schema.

Purpose
-------
This exists ONLY to exercise and test the pipeline (feature SQL, leakage
guarantees, replayer clock, training loop) without waiting on the 15GB Kaggle
download. Results produced from this data are never reported as project
results - it has known, planted signal, so a good AUC here proves the code
works, not that the problem is solvable.

Generation model
----------------
Each session is drawn as one of two archetypes:

  * "intent"  - focuses on 1-3 products, revisits them, dwells longer,
                often carts, and usually purchases
  * "browse"  - sprays across many categories, short gaps, rarely purchases

A planted Black Friday window shifts price and conversion distributions so the
drift-detection code has something real to fire on.

Usage:
    set DATA_ROOT=D:/ecommerce-ml-platform/data_synth
    python -m scripts.make_synthetic_data --sessions 40000
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)

CATEGORIES = [
    ("electronics.smartphone", 2053013555631882655),
    ("electronics.audio.headphone", 2053013553031414015),
    ("computers.notebook", 2053013554658804075),
    ("appliances.kitchen.refrigerators", 2053013554776244595),
    ("furniture.living_room.sofa", 2053013553559896355),
    ("apparel.shoes", 2053013556168753601),
]
BRANDS = ["samsung", "apple", "xiaomi", "lg", "sony", "bosch", "ikea", "nike", None]


def generate(
    n_sessions: int,
    start: datetime,
    days: int,
    n_users: int,
    n_products: int,
    black_friday_day: int | None,
    seed: int = 7,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)

    # Stable product catalogue: each product has a fixed category/brand/price.
    prod_ids = rng.choice(np.arange(1_000_000, 1_000_000 + n_products * 4), n_products, replace=False)
    prod_cat = rng.integers(0, len(CATEGORIES), n_products)
    prod_brand = rng.integers(0, len(BRANDS), n_products)
    prod_price = np.round(np.exp(rng.normal(4.9, 1.05, n_products)), 2)

    rows: list[dict] = []
    users = rng.integers(500_000_000, 500_000_000 + n_users, n_sessions)

    for s in range(n_sessions):
        day = int(rng.integers(0, days))
        is_bf = black_friday_day is not None and day == black_friday_day

        # Black Friday: more traffic converts, and cheaper items dominate.
        p_intent = 0.55 if is_bf else 0.30
        intent = rng.random() < p_intent

        t = start + timedelta(
            days=day,
            hours=int(rng.integers(0, 24)),
            minutes=int(rng.integers(0, 60)),
            seconds=int(rng.integers(0, 60)),
        )
        session_id = f"{s:08x}-synt-{day:03d}"
        user_id = int(users[s])

        # Click-gap distributions overlap heavily and are jittered per session.
        # An earlier version gave intent sessions 15-120s and browsers 2-25s,
        # which barely overlapped - `max_gap_sec` then separated the archetypes
        # almost perfectly, LightGBM converged in 5 trees, and the fixture could
        # no longer reveal a bug in any other feature. Signal must be spread
        # across several weak features for this data to be a useful test.
        base_gap = float(rng.uniform(4, 40))
        if intent:
            n_focus = int(rng.integers(1, 4))
            focus = rng.choice(n_products, n_focus, replace=False)
            n_events = int(rng.integers(6, 30))
            pool = np.concatenate(
                [np.repeat(focus, 4), rng.choice(n_products, max(n_events, 1), replace=True)]
            )
            gap_lo, gap_hi = max(2, base_gap * 0.5), base_gap * 2.2
        else:
            n_events = int(rng.integers(3, 22))
            pool = rng.choice(n_products, max(n_events * 2, 2), replace=True)
            gap_lo, gap_hi = max(1, base_gap * 0.4), base_gap * 1.6
        gap_lo, gap_hi = int(gap_lo), max(int(gap_hi), int(gap_lo) + 1)

        picks = rng.choice(pool, n_events, replace=True)
        carted = False
        for i, p in enumerate(picks):
            p = int(p)
            price = float(prod_price[p])
            if is_bf:
                price = round(price * float(rng.uniform(0.55, 0.85)), 2)
            cat_code, cat_id = CATEGORIES[int(prod_cat[p])]
            rows.append(
                dict(
                    event_time=t,
                    event_type="view",
                    product_id=int(prod_ids[p]),
                    category_id=int(cat_id),
                    category_code=cat_code,
                    brand=BRANDS[int(prod_brand[p])],
                    price=price,
                    user_id=user_id,
                    user_session=session_id,
                )
            )
            t += timedelta(seconds=int(rng.integers(gap_lo, gap_hi)))

            # Intent sessions cart late, then usually convert.
            if intent and not carted and i >= max(2, n_events - 4) and rng.random() < 0.6:
                carted = True
                last = dict(rows[-1])
                last.update(event_time=t, event_type="cart")
                rows.append(last)
                t += timedelta(seconds=int(rng.integers(20, 180)))

        if carted and rng.random() < (0.75 if is_bf else 0.55):
            last = dict(rows[-1])
            last.update(event_time=t, event_type="purchase")
            rows.append(last)

    df = pl.DataFrame(rows).with_columns(
        pl.col("event_time").dt.date().alias("event_date")
    )
    return df.select(
        "event_time", "event_date", "event_type", "product_id", "category_id",
        "category_code", "brand", "price", "user_id", "user_session",
    ).sort("event_time")


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", type=int, default=40_000)
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--users", type=int, default=12_000)
    ap.add_argument("--products", type=int, default=3_000)
    ap.add_argument("--start", default="2019-10-01")
    ap.add_argument(
        "--black-friday-day", type=int, default=40,
        help="day offset that gets the planted regime shift; -1 to disable",
    )
    args = ap.parse_args()

    bf = None if args.black_friday_day < 0 else args.black_friday_day
    df = generate(
        n_sessions=args.sessions,
        start=datetime.fromisoformat(args.start),
        days=args.days,
        n_users=args.users,
        n_products=args.products,
        black_friday_day=bf,
    )

    out = settings.events_dir
    out.mkdir(parents=True, exist_ok=True)
    for (d,), part in df.group_by(["event_date"]):
        pdir = out / f"event_date={d}"
        pdir.mkdir(parents=True, exist_ok=True)
        part.drop("event_date").write_parquet(pdir / "synthetic_0.parquet", compression="zstd")

    sess = df.group_by("user_session").agg(
        bought=pl.col("event_type").eq("purchase").max(),
        n=pl.len(),
    )
    log.info("wrote %s events across %d days -> %s", f"{df.height:,}", df["event_date"].n_unique(), out)
    log.info("sessions: %s", f"{sess.height:,}")
    log.info("session purchase rate: %.2f%%", float(sess["bought"].mean()) * 100)
    log.info("median events/session: %.0f", float(sess["n"].median()))
    log.info("SYNTHETIC DATA - for pipeline testing only, never for reported results")


if __name__ == "__main__":
    main()
