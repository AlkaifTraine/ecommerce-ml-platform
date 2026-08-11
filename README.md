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

### 3. Pipeline test without the download

The full pipeline runs on generated data with the same schema:

```bash
set DATA_ROOT=D:/ecommerce-ml-platform/data_synth
python -m scripts.make_synthetic_data --sessions 40000
python -m src.features.sessionize
python -m src.features.build_features --k 10
python -m src.models.train --k 10
```

This validates that the **code** is correct. It says nothing about whether the
problem is solvable — the generator has planted signal. Only the real dataset
produces reportable results.

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

_(pending — this section stays empty until there are measured numbers from the
real dataset. Targets going in are AUC ≥ 0.82 and lift@10% ≥ 4x, with the
project reassessed rather than quietly reported if they are not met.)_

---

## Known limitations

Stated up front because they will come up in any serious conversation about
this project:

- **No treatment data.** The dataset records what shoppers did, not what the
  store did to them. That supports purchase *propensity* but not *uplift*
  ("would buy only if nudged"), which needs randomised treatment assignment.
  The OLTP schema includes an `interventions.is_holdout` flag for the untreated
  control group such a design would require.
- **Offline ranking metrics are biased** by whatever ranker produced the logged
  behaviour. Counterfactual estimation (IPS) is the honest fix and is not yet
  implemented.
- **The clock is simulated.** See above; stated plainly rather than glossed.
