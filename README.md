# Real-Time Purchase-Intent Platform

A continuously-retraining ML platform built on a real e-commerce clickstream.
It stands inside a live shopping session, predicts whether that session will end
in a purchase, and keeps itself accurate as shopper behaviour changes.

> **Status: in progress.** Sections marked _(pending)_ have no measured results
> yet. No number appears in this README until it has actually been produced by a
> run against the real dataset.

---

## The problem

An online store with a large catalogue makes two decisions thousands of times a
second, and does both badly by default:

1. **What to show.** Ranking by "current bestsellers" ignores what the visitor
   in front of you is actually doing.
2. **Who to spend money on.** Retargeting ads, discount codes and trigger emails
   have a fixed budget, but firing them at every abandoned cart wastes most of
   it — on people who were never going to buy, and on people who would have
   bought anyway.

This platform replaces the second decision with a learned, real-time one, and
keeps it correct through genuine changes in shopper behaviour.

## The prediction task

Stand inside a session immediately after its **k-th event**. Using only those k
events plus history that predates the session, predict whether a purchase
happens **later in the same session**.

Three eligibility rules make the task honest:

| Rule | Why it exists |
|---|---|
| Session must have at least k events | You cannot make a k-event prediction on a session that never reached k events |
| No purchase inside the first k events | If they already bought, there is nothing left to predict |
| Label reads only events k+1..end | Nothing at or after the cutoff touches the features; nothing at or before it touches the label |

The most common way a project like this produces impressive-but-worthless
numbers is by letting `cart` leak in from after the cutoff — cart is very nearly
the label, so the model scores ~0.97 AUC and is useless in production, because by
the time someone carts it is too late to act. `tests/test_features.py::
test_post_cutoff_cart_is_invisible` exists specifically to prevent that.

Cart events *before* the cutoff are legitimately observable and are kept.
`--hard-mode` strips them anyway, to show the model still works without its
strongest single feature.

---

## Architecture

```
  Parquet archive (7 months, all of it, on disk)
        │   ONLY the replayer may read this
        ▼
  ┌───────────────┐   holds current_data_time; nothing may read past it
  │   REPLAYER    │──────────────────────────────────────────────┐
  └───────┬───────┘                                              │
          ▼                                                      │
   Redis Streams  ──────────────►  storefront consumer           │
   (Kafka optional)                       │                      │
                                          ▼                      │
                                 ┌──────────────────┐            │
                                 │  OLTP: Postgres  │            │
                                 │  users, sessions │            │
                                 │  carts, orders   │            │
                                 │  7-day hot window│            │
                                 └────────┬─────────┘            │
                                          │ CDC extract          │
                                          ▼                      │
                                 ┌──────────────────┐            │
                                 │  OLAP warehouse  │◄───────────┘
                                 │  + dbt models    │
                                 └────────┬─────────┘
                                          ▼
                            features → training → MLflow registry
                                          ▼
                              serving API  ←→  drift monitor
                                          └── triggers retrain
```

### Why both an OLTP and an OLAP store

Not decoration. Postgres holds a **rolling 7-day hot window** of events —
normalised, constrained, indexed for row-level access, and deliberately small.
History is trimmed out of it and lives in Parquet and the warehouse.

That means "how did conversion shift across last quarter?" *cannot* be answered
from Postgres. The constraint is real, and it is the reason the warehouse exists.

---

## How it is continuous when the data is static

The dataset is seven months of history sitting in files. The replayer is the
only component allowed to read it, and it releases events strictly in
`event_time` order against a clock it owns. Every other component reads through
that clock and therefore cannot see the future.

The clock runs faster than real time, so seven months of shopping — including
its regime changes — plays out in about 30 wall-clock hours.

**What is simulated:** the clock, and only the clock.
**What is real:** the events, their ordering, the drift, the retraining
pipeline, the model comparisons, the promotion and rollback logic.

The no-peeking guarantee is enforced in code, not by good intentions:

- the clock is a single row in Postgres, shared by every process
- time-bounded reads take an explicit `--until`, applied as a SQL predicate
- `src/features/leakage_audit.py` recomputes the truncation logic **from the
  definition rather than from the implementation** and checks the published
  training table against it
- `tests/test_leakage.py::test_audit_has_teeth` deliberately corrupts a label
  and asserts the audit fails — a suite that only ever sees clean data proves
  nothing

On a CV this is described as **event-time replay of a historical archive**,
never as live production traffic.

---

## Repository layout

```
src/
  platform_core/     config and logging; every path resolves through here
  ingest/            Kaggle download, CSV→Parquet, data profiling
  features/          sessionisation, truncation-point features, leakage audit
  models/            training, business-facing metrics
  replayer/          the clock, event sinks, replay engine, OLTP consumer
  serving/           scoring API                                    (pending)
  monitoring/        drift detection and auto-promotion             (pending)
  warehouse/         CDC extract and warehouse loading              (pending)
dags/                Airflow DAGs                                   (pending)
dbt/                 warehouse models                               (pending)
docker/              compose stack + OLTP schema
tests/               feature correctness and leakage guarantees
scripts/             synthetic data, environment helpers
```

---

## Setup

### 1. Environment

```bash
python -m venv --system-site-packages .venv
.venv\Scripts\pip install -r requirements.txt
cp .env.example .env
```

> The venv inherits system site-packages deliberately: the dev machine has
> ~100 KB/s to PyPI, and numpy/pandas/pyarrow/scikit-learn/LightGBM were already
> installed system-wide. `scripts/fetch_wheels.ps1` downloads the remaining
> large wheels with `curl -C -` because pip cannot resume an interrupted
> transfer, which matters a great deal on a slow link.

### 2. Data

The dataset is REES46's multi-category store clickstream.

> **Note:** REES46's own mirror (`data.rees46.com`) serves an **expired TLS
> certificate**. Certificate verification is not disabled to work around this;
> Kaggle is used as the trusted distribution channel instead.

Kaggle needs an API token, which only you can create:

1. Go to <https://www.kaggle.com/settings> → **API** → **Create New Token**
2. Save the downloaded `kaggle.json` to `%USERPROFILE%\.kaggle\kaggle.json`

```bash
python -m src.ingest.download --list          # see files and sizes first
python -m src.ingest.download
python -m src.ingest.csv_to_parquet --delete-source
python -m src.ingest.validate
```

`--delete-source` removes each CSV once its conversion is verified, keeping peak
disk usage bounded.

### 3. Build the pipeline

```bash
python -m src.features.sessionize
python -m src.features.build_features --k 5
python -m src.models.train --k 5
```

Before modelling, two checks run against the raw archive:

```bash
python -m scripts.gate1_signal_check     # is the task viable at all?
python -m scripts.data_quality_audit     # which days did the source pipeline break?
```

The audit is not optional. This dataset loses purchase events on
**2019-11-14..17** — 2019-11-15 records none at all while 766K sessions browse
normally. Those days are quarantined in `config.py`; sessions touching them are
dropped, because their labels are unknowable rather than negative.

> **No synthetic data.** Tests run on a committed slice of the real archive
> (`tests/fixtures/`, built by `scripts/make_test_fixture.py`) plus a 21-event
> hand-written unit fixture. An earlier synthetic generator was removed: it
> gave browsers and buyers barely-overlapping click gaps, so one feature
> separated them perfectly and the fixture stopped being able to detect bugs
> anywhere else.

### 4. Services

```bash
cd docker && docker compose up -d
```

Postgres on **5433**, Redis on **6379**. MLflow, Kafka and Adminer sit behind
profiles so they are not pulled unless wanted.

### 5. Tests

```bash
python -m pytest -v
```

---

## Results

Measured on the real archive: **109,950,743 events**, Oct–Nov 2019, 23,016,650
sessions, 5,316,649 users, 206,876 products. Purchase rate is **6.10% of
sessions**. Sessions are short — median 2 events, p90 11 — so `k=5` is the
operating point: it covers 28% of traffic and ~32% of all purchasing sessions,
where `k=10` reaches only 11.7% and 13%.

Training set: 5,148,713 eligible sessions, 6.86% positive. Split strictly by
time — train Oct 1–Nov 9, validate Nov 10–20, test Nov 21–25, drift Nov 26–30.

| | ROC-AUC | PR-AUC | recall@10% | lift@10% |
|---|---|---|---|---|
| Random | 0.5003 | 0.079 | 10.0% | 1.00x |
| Heuristic: dwell time | 0.4358 | 0.068 | 8.1% | 0.81x |
| Heuristic: revisit count | 0.6722 | 0.150 | 27.9% | 2.79x |
| Heuristic: cart count | 0.7147 | 0.232 | 42.5% | 4.25x |
| **LightGBM — test** | **0.8406** | **0.4140** | **49.9%** | **4.99x** |
| LightGBM — drift window | 0.8369 | 0.4042 | 49.8% | 4.98x |

**Contact 10% of traffic, reach 49.9% of the buyers** — 5x better than random,
from the first five events of a session.

### Robustness: the model is not a cart-counter

`n_cart_events` carries more than double the gain of any other feature, and
cart recording is known to be inconsistent between October and November. So the
same model was retrained with every cart signal stripped (`--hard-mode`):

| | ROC-AUC | PR-AUC | recall@10% | lift@10% |
|---|---|---|---|---|
| Full model | 0.8406 | 0.4140 | 49.9% | 4.99x |
| Hard mode (no cart) | 0.8092 | 0.3394 | 42.8% | 4.28x |

Losing its strongest feature costs 0.031 AUC. The stripped model still reaches
**4.28x lift — marginally better than the 4.25x that cart-count ranking
achieves while still having cart data**. The signal is largely reconstructable
from revisit patterns, price behaviour and prior-session history, so the
instrumentation inconsistency degrades the model gracefully rather than
breaking it.

(Sanity check: the `heuristic_cart` baseline drops to exactly AUC 0.5000 under
hard mode, confirming the signals were genuinely removed.)

### Reading these honestly

* **Compare on PR-AUC, not lift.** Ranking by cart count alone already achieves
  4.25x lift, so the model's 4.99x is a 17% gain on that metric. The real
  separation is PR-AUC: 0.232 → 0.414, a 79% improvement, which is the metric
  that matters at a 6.9% base rate.
* **Dwell time is anti-predictive** (AUC 0.4358, below random). Slow browsing
  converts *less*; decisive sessions buy. The intuition that longer dwell means
  higher intent is simply wrong on this dataset.
* **The drift window barely degrades** (0.8406 → 0.8369). Consistent with the
  PSI finding below: the features are not moving, the label rate is.

### Feature importance (gain)

`n_cart_events` dominates at 1,189,857 — more than double the next feature.
Ranks 2, 3, 5 and 6 are all point-in-time user history
(`user_prior_conv_rate`, `user_prior_revenue`, `user_prior_sessions`,
`user_prior_purchases`), which confirms the as-of-session-start windows are
contributing real signal rather than leaking current-session outcomes.

### Where the drift actually is

| Signal | PSI (Oct vs late Nov) | Verdict |
|---|---|---|
| price | 0.0024 | no shift |
| log price | 0.0025 | no shift |
| category mix | 0.0116 | no shift |

Feature distributions are essentially static, while conversion per session
slides from ~8% in early October to ~5.5% by mid-November — a 30% relative
decline. **A feature-drift dashboard would show all green while the model
quietly degrades.** That is the argument for delayed-label performance
monitoring rather than feature monitoring, and it is demonstrable here on real
data.

Black Friday, for the record, is a damp squib in this dataset: +13% relative
conversion and no price discount signature at all. The continuous-retraining
story is built on the collection outage and the label-rate decline instead.

### Data-quality finding

The source pipeline **lost purchase events on 2019-11-14..17**:

| Date | Views | Carts | Purchases | buy/cart |
|---|---|---|---|---|
| Nov 11–13 (normal) | ~1.9M | ~72K | ~23K | ~32% |
| Nov 14 | 2.88M | 170K | 22K | 12.98% |
| Nov 15 | 5.74M | 483K | **0** | 0% |
| Nov 16 | 6.03M | 407K | 68K | 16.78% |
| Nov 17 | 5.78M | 427K | 185K | 43.38% |
| Nov 18+ | ~1.9M | ~83K | ~28K | ~34% |

Nov 15 records no purchases at all while 766K sessions browse normally, and the
backlog lands on Nov 17. Trained on as-is, **3.3M sessions would be silently
labelled "did not buy"** when the truth is unknowable. Those days are
quarantined in `config.py` and any session touching them is dropped.

Separately, cart events are badly under-recorded in October (buy/cart ~69% vs
~34% in November), so cart-derived features are not comparable across the two
months — which is what `--hard-mode` exists to test.

---

## Warehouse and dbt

The base star schema is built by `src/warehouse/build_warehouse.py` (55s, 2.3GB
DuckDB file) rather than by dbt, because scanning 110M events inside a 4GB
memory budget needs tuning that dbt does not express well. dbt owns the layer
above it — derived marts and, more importantly, the tests.

| Table | Rows |
|---|---|
| `dim_date` | 61 |
| `dim_product` | 206,876 |
| `dim_user` | 5,316,115 |
| `fct_session` | 23,016,650 |
| `agg_daily` | 61 |
| `agg_category_daily` | 854 |

```bash
python -m src.warehouse.build_warehouse
cd dbt && dbt build --profiles-dir .        # 2 models, 24 tests
python -m scripts.warehouse_demo            # queries Postgres cannot answer
```

Questions the warehouse answers that the OLTP store structurally cannot, since
it retains only seven days:

* conversion by half-month — **8.08% → 7.93% → 5.97% → 6.21%**, a 26% relative
  decline, while average price barely moves
* the 2019-11-14..17 outage, in a single query, with quarantine flags joined
  from `dim_date`
* **256,958 repeat buyers — 37% of all buyers — generate 73.8% of revenue**

### Determinism

Three separate places resolved ties *arbitrarily*, which meant rebuilding the
pipeline could produce a different answer:

| Location | Arbitrary choice | Fix |
|---|---|---|
| `row_number()` over `(event_time, product_id)` | which event is k-th | dedupe, then order by `(event_time, product_id, event_type)` |
| `any_value(user_id)` per session | who owns a multi-user session | `min(user_id)` |
| `any_value(category_code)` per product | a product's category | `min(...)` |

Individually these touch 1 row, 939 sessions and a handful of products — none
would move a metric. Collectively they meant "rebuild" and "get the same
answer" were different things, which invalidates any A/B comparison between
model versions. On a platform whose premise is continuous retraining, that is
the difference between measuring improvement and measuring build noise.
`tests/test_features.py::test_feature_build_is_deterministic` guards the first.

## Known limitations

Stated up front because they will come up in any serious conversation about
this project:

- **939 sessions (0.004%) contain events from more than one `user_id`**, one of
  them spanning three users across 43 events. Ownership is genuinely ambiguous;
  they are attributed to `min(user_id)` — equally arbitrary but reproducible.
  109 of them reach the training table. A consequence: 534 users appear *only*
  as the non-minimum party in such sessions and therefore own no session, which
  is why `dim_user` holds 5,316,115 rows against 5,316,649 distinct user ids in
  the raw archive.
- **12 events carry a NULL `user_session`** — all cart events, from 12
  unrelated users. `GROUP BY` merged them into one phantom session containing
  twelve strangers. They never reached training, but only because `NULL = NULL`
  is never true in a join, which is luck rather than a control. Now excluded
  explicitly in `sessionize.py`.

- **No treatment data.** The dataset records what shoppers did, not what the
  store did to them. That supports purchase *propensity* but not *uplift*
  ("would buy only if nudged"), which needs randomised treatment assignment.
  The OLTP schema includes an `interventions.is_holdout` flag for the untreated
  control group such a design would require.
- **Offline ranking metrics are biased** by whatever ranker produced the logged
  behaviour. Counterfactual estimation (IPS) is the honest fix and is not yet
  implemented.
- **The clock is simulated.** See above; stated plainly rather than glossed.
