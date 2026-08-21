# Real-Time Purchase-Intent Platform — Complete Project Reference

**Purpose of this document.** A self-contained description of the entire
project: the problem, the data, the modelling contract, the architecture, every
measured result, every bug found and why it mattered, and the known limitations.
It is written so it can be pasted into an AI assistant as context for answering
questions about the project.

**Repository:** https://github.com/AlkaifTraine/ecommerce-ml-platform
**Local path:** `D:\ecommerce-ml-platform`
**Status:** complete; all components executed and verified against real data.
**Not deployed to cloud** — AWS is designed and costed only (see §12).

> **Resuming work in a new session? Read §16 first.** It records exactly what
> exists on disk, what is running, the traps that will bite again, and the open
> items in rough order of value.

Every number in this document was measured by running the code. Nothing is
estimated. Where something is unverified or uncertain, it says so.

---

## 1. THE PROBLEM

An online store with a large catalogue makes one decision thousands of times a
second: **who is worth spending money on?**

Retargeting ads, discount codes and trigger emails have a fixed budget. Firing
them at everyone who abandons a cart wastes most of it — on people who were
never going to buy, and on people who would have bought anyway.

This platform scores every live session for purchase intent so the budget goes
to the sessions where an intervention can actually change the outcome.

### Why this domain rather than finance

In markets the outcome is caused by information you cannot observe, and any
edge is arbitraged away — you fight for a 1–2% edge. Here the outcome is
*caused by* directly observable behaviour: someone who views the same product
four times and returns the next day is demonstrating intent. There is no
adversary erasing the signal. The achievable edge is roughly 5×, not 2%.

### The business decision the score drives

Implemented in `src/serving/api.py::_decide`:

| Score | Decision | Reasoning |
|---|---|---|
| ≥ 0.60 | no intervention | likely to convert unaided — do not spend |
| 0.15 – 0.60 | **intervene** | persuadable band — discount or free shipping |
| < 0.15 | no intervention | unlikely to convert — not worth the spend |

Spending on the highest scores is wasteful; those sessions largely convert
anyway. The money belongs in the middle band.

---

## 2. THE DATA

### Source

REES46 "eCommerce behavior data from multi-category store", obtained from
Kaggle: `mkechinov/ecommerce-behavior-data-from-multi-category-store`.

**Note on provenance:** REES46's own mirror (`data.rees46.com`) serves an
**expired TLS certificate** (verified with curl, error 35 `SEC_E_CERT_EXPIRED`).
Certificate verification was not disabled to work around this; Kaggle was used
as the trusted distribution channel.

### Scale (measured)

| Metric | Value |
|---|---|
| Events | **109,950,743** |
| Date range | 2019-10-01 00:00:00 → 2019-11-30 23:59:59 (61 days) |
| Sessions | 23,016,650 |
| Users | 5,316,649 |
| Products | 206,876 |
| Categories | 691 |
| Brands | 4,303 |
| Raw CSV | 13.67 GB (Nov 8.39 GB + Oct 5.28 GB) |
| Parquet | 1.9 GB, 61 date partitions (**7.2× compression**) |

### Schema

Columns: `event_time`, `event_type`, `product_id`, `category_id`,
`category_code`, `brand`, `price`, `user_id`, `user_session`.

`event_time` in the raw CSV carries a trailing `" UTC"` (e.g.
`2019-10-01 00:00:00 UTC`) which is not a parseable timestamp literal. The
converter sniffs the first row and strips it.

### Event type distribution

| Type | Count | Share |
|---|---|---|
| view | 104,335,509 | 94.893% |
| cart | 3,955,446 | 3.598% |
| purchase | 1,659,788 | 1.510% |

**There is no `remove_from_cart` event type in this data.** An
`n_remove_events` feature was written and then removed because it was always
zero.

### Null rates

| Column | Null % |
|---|---|
| category_code | 32.209% |
| brand | 13.944% |
| all others | 0.000% |

`user_session` shows 0.000% but is **not** truly zero — see §8.6.

### Base rates

| Metric | Value |
|---|---|
| Sessions that purchase | **6.095%** |
| Sessions that cart | 10.064% |
| Cart → purchase | 40.59% |

### Session length distribution (critical)

| Statistic | Events |
|---|---|
| mean | 4.78 |
| p25 | 1 |
| **median** | **2** |
| p75 | 5 |
| p90 | 11 |
| p99 | 35 |
| max | 4,128 |

Sessions are very short. This drove the choice of k=5 (see §3).

---

## 3. THE PREDICTION TASK

### Definition

Stand inside a session immediately after its **k-th event**. Using only those k
events plus history that predates the session, predict whether a purchase
happens **later in the same session**.

`k = 5` is the operating point.

### Eligibility rules

1. **Session must have ≥ k events.** You cannot make a k-event prediction on a
   session that never reached k events.
2. **No purchase within the first k events.** If the customer already bought,
   there is nothing left to predict, and including those rows hands the model
   the answer.
3. **No event on a quarantined day** (see §7.1). A session touching the outage
   has an unknowable label.
4. **The label reads only events strictly after the cutoff TIMESTAMP.**

### Why the label is defined by time, not by rank

Timestamps have one-second granularity, so events within a session frequently
tie. An event ranked k+1 can share the cutoff second with the event ranked k.
Labelling such an event "after the cutoff" is optimistic — at the instant you
score, it may already have happened.

So the label window is `event_time > cutoff_time`, and events sharing the
cutoff second are excluded from **both** features and label. This costs a few
positives and removes the ambiguity entirely.

### Why k=5 and not k=10

| k | Eligible sessions | % of all traffic | Positive rate | Share of all purchases reached |
|---|---|---|---|---|
| **5** | **6,437,299** | **28.0%** | 7.008% | ~32% |
| 10 | 2,692,064 | 11.7% | 6.727% | ~13% |
| 20 | 800,406 | 3.5% | 6.984% | — |

k=10 covers only 11.7% of traffic and reaches only ~13% of purchasing sessions.
k=5 is the right operating point.

### The leakage trap, and the nuance

The naive version lets `cart` leak in from after the cutoff. Cart is very nearly
the label, so the model scores ~0.97 AUC and is useless in production — by the
time someone carts, it is too late to act.

**But cart events *before* the cutoff are legitimate** and are kept: they are
genuinely observable at prediction time. The rule is about the cutoff, not about
cart. A `--hard-mode` variant strips all cart signals to prove the model does
not depend on them (§6.3).

`tests/test_features.py::test_post_cutoff_cart_is_invisible` exists specifically
to catch a post-cutoff cart reaching the features.

### The 32 features

**Breadth vs depth of browsing:** `n_distinct_products`,
`n_distinct_categories`, `n_distinct_brands`, `n_repeat_views`,
`max_views_same_product`, `mean_views_per_product`, `repeat_product_ratio`

**Pace:** `prefix_duration_sec`, `mean_gap_sec`, `median_gap_sec`,
`min_gap_sec`, `max_gap_sec`, `events_per_minute`

**Price context:** `price_mean`, `price_min`, `price_max`, `price_std`
(population stddev), `price_range`, `price_at_cutoff`

**Data-quality signals:** `null_brand_ratio`, `null_category_ratio` (browsers
hit sparse listings more often)

**Intent:** `n_cart_events`, `cart_ratio`

**Calendar:** `hour_of_day`, `day_of_week` (DuckDB convention, Sunday = 0),
`is_weekend`

**Point-in-time user history:** `user_prior_sessions`, `user_prior_purchases`,
`user_prior_conv_rate`, `user_prior_revenue`, `hours_since_last_session`,
`is_new_user`

### Point-in-time correctness

User-history features are computed with window functions ordered by
`session_start`, using frames ending at `1 PRECEDING`. A session therefore never
sees itself or any later session. This is the cheap O(n log n) equivalent of an
as-of join.

### Splitting

Strictly chronological, never random. Four windows as fractions of the observed
date range: train → 70%, valid → 82%, test → 92%, drift → end. The drift window
is reported separately and expected to be worse; that gap is evidence that
retraining is needed, so it is a result rather than a failure.

---

## 4. ARCHITECTURE

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
                                 │  DuckDB + dbt    │
                                 └────────┬─────────┘
                                          ▼
                            features → training → MLflow registry
                                          ▼
                              serving API  ←→  drift monitor
                                          └── triggers retrain
```

### 4.1 The replayer — how a static archive becomes "continuous"

The dataset is seven months of history in files. The replayer is the **only**
component permitted to read it, and it releases events strictly in `event_time`
order against a clock it owns. Everything else reads through that clock and
therefore cannot see the future.

Two phases, mirroring how a real system is brought up:
- **backfill** — bulk-load history fast to bootstrap the stores
- **live** — advance the clock at `speed` × real time so drift events play out
  slowly enough for the retraining loop to react

**What is simulated:** the clock, and only the clock.
**What is real:** the events, their ordering, the drift, the retraining
pipeline, the model comparisons, promotion and rollback.

On a CV this is described as **"event-time replay of a historical archive"**,
never as live production traffic.

**Why replay beats a genuinely live feed:** with replay you get seven months of
real business including two anomalies in about a day, and you can re-run it
while debugging. With a live feed you would wait seven months and might get
nothing interesting.

### 4.2 Enforcement of the no-peeking guarantee

- The clock is a single row in `storefront.replay_clock`, shared by every
  process, so separate services cannot drift apart.
- Time-bounded reads take an explicit `--until`, applied as a SQL predicate.
- `src/features/leakage_audit.py` recomputes the truncation logic **from the
  definition rather than from the implementation**, and checks the published
  training table against it. A bug copied into both would defeat the point.
- `tests/test_leakage.py::test_audit_has_teeth` deliberately corrupts a label
  and asserts the audit fails. A suite that only ever sees clean data proves
  nothing.

### 4.3 OLTP — Postgres

Normalised, constrained, indexed for row-level access. 13 tables:
`replay_clock`, `users`, `products`, `sessions`, `session_events`, `carts`,
`cart_items`, `orders`, `order_items`, `predictions`, `prediction_outcomes`,
`interventions`, `cdc_watermark`.

**Retention:** `session_events` is a rolling **7-day hot window of DATA time**.
Older rows are archived to the lake and deleted.

**This constraint is the point.** "How did conversion move across two months?"
*cannot* be answered from Postgres. That is why the warehouse exists.

Tuning: `synchronous_commit=off` (trades crash durability for throughput —
correct here because the archive is the source of truth and a replay can be
re-run), `shared_buffers=384MB`, `wal_compression=on`.

### 4.4 Event bus — Redis Streams

An append-only log with consumer groups and offsets; maps to Kinesis on AWS.
Redis was chosen locally because it needs no KRaft/ZooKeeper and the image is
small, and because there is exactly **one** consumer — so Kafka's real
advantages (multiple independent consumer groups, replay from arbitrary
offsets, partitioned horizontal scale, replication) do not apply.

**Honesty note.** A `KafkaSink` exists in `src/replayer/sinks.py` behind the
same `EventSink` interface, and a `kafka` docker-compose profile exists — but
**`kafka-python` is not installed and that code path has never executed once.**
Redis Streams is what actually ran. Do not claim Kafka experience from this
project; claim the trade-off reasoning instead, which is the better answer
anyway.

**When Kafka becomes correct here:** a second consumer needs the same stream, or
you need to replay from an offset rather than from the archive, or one broker
stops being enough.

### 4.5 OLAP — DuckDB star schema + dbt

Built by `src/warehouse/build_warehouse.py` in ~55s:

| Table | Rows |
|---|---|
| `dim_date` | 61 |
| `dim_product` | 206,876 |
| `dim_user` | 5,316,115 |
| `fct_session` | 23,016,650 |
| `agg_daily` | 61 |
| `agg_category_daily` | 854 |

dbt (`dbt-core 1.12.0`, `dbt-duckdb 1.11.0`) owns the derived marts and the
tests: `mart_daily_conversion` (conversion vs a trailing 7-day baseline that
excludes the current day, so a bad day cannot flatter its own baseline) and
`mart_user_segments` (RFM). **`dbt build` passes 26/26.**

A singular test, `assert_outage_days_are_flagged.sql`, asserts the outage days
remain flagged so a future threshold change cannot silently disable the detector.

**Why the base tables are not dbt models:** scanning 110M events inside a 4GB
memory budget needs tuning dbt does not express well.

### 4.6 CDC extraction

`src/warehouse/cdc_extract.py` drains the OLTP hot window into the Parquet lake
before retention deletes it.

**The monotonic-id trap:** the obvious implementation is
`SELECT ... WHERE id > watermark`. That is subtly wrong — `BIGSERIAL` values are
assigned when a statement executes, not when its transaction commits, so a row
with id 100 can become visible *after* row 101 was already extracted.

Two defences: a **safety lag** (never extract closer than `--lag-seconds` to the
data clock) and a watermark advanced only inside the same transaction as the
read. The genuinely correct fix is logical replication (Debezium reading the
WAL), which observes commit order; that is noted as the cloud path.

A third defence was added after a real data-loss bug — see §8.12.

### 4.7 Orchestration — Airflow

**Topology:** ONE container running `LocalExecutor`, with Airflow's metadata in
a separate database inside the existing Postgres. The conventional
webserver + scheduler + triggerer + dedicated Postgres wants ~2.5GB; the
development machine has ~4GB free. Base image pinned to `apache/airflow:2.10.0`
because that tag was already local (any other tag is a ~2GB pull at ~100 KB/s).

**Three DAGs:**

**`continuous_training`** — `read_clock → build_features → leakage_audit →
train → register → promote`.
- The leakage audit runs **before** training and fails the DAG. A pipeline that
  trains first and checks later will register a leaking model and only mention
  it afterwards.
- Every task is bounded by the replay clock, not wall time.

**`drift_monitor`** — hourly; branches four ways on the verdict:
- `OK` / `WATCH` → `no_action`
- `COVARIATE_SHIFT` → `trigger_retrain`
- `LABEL_SHIFT` → `recalibrate`
- `DATA_QUALITY_INCIDENT` → `quarantine_and_hold` (fails deliberately)

**`warehouse_refresh`** — `cdc_extract → build_warehouse → dbt_build`, every 6
hours. Ordered so extraction happens before OLTP retention deletes the window.

### 4.8 Model registry — MLflow

SQLite-backed tracking. **A `file://` store supports runs but not model versions
or aliases**, and the promotion flow needs both.

Three aliases: `champion` (what serving uses), `challenger` (new candidate),
`shadow` (scored alongside, output discarded).

**Promotion requires beating the incumbent by 0.002 AUC**, not merely tying.
Without a margin, daily retraining on a metric that moves ±0.003 churns the
production model constantly — it looks like progress and is not.

### 4.9 Serving — FastAPI

- Serves whichever version holds `champion`; `/reload` picks up promotion or
  rollback without a redeploy.
- **Looks user history up from the online feature store** (§4.11) rather than
  expecting the caller to supply it. Returns `history_source` on every response
  so the provenance of a score is visible.
- Falls back to the local artifact when the registry is unreachable — refusing
  to serve because a tracking server is down would make monitoring a hard
  dependency of the storefront.
- Logs predictions with their feature snapshot **before** responding, because
  "why did it score 0.9?" is unanswerable without the inputs, and delayed-label
  monitoring has nothing to join against.
- Logging failures never fail a request.
- Returns **422, not 500**, for sessions that already purchased inside the
  prefix — the training eligibility rule enforced at serving time.
- `scored_at_data_time` is the replay clock, not wall time, so monitoring joins
  line up with the simulated calendar.

### 4.11 Online feature store

`src/serving/feature_store.py`. Two-tier, the standard shape:

```
offline   warehouse dim_user (full 61-day history)
              |  materialise on a schedule
              v
online    Redis hash per user (feat:user:{id}), O(1) at serving time
              |  fallback on miss
              v
          Postgres storefront.users (7-day hot window only — DEGRADED)
```

**5,316,115 users materialised**, occupying ~732MB of Redis's 768MB cap.

**Materialisation reads the warehouse, not Postgres**, because the OLTP store
keeps only a 7-day hot window — using it would hand serving a truncated history
that training never saw, which is training/serving skew by construction.

**`found` and `source` are returned explicitly.** This is the whole point: the
API must distinguish "this user genuinely has no history", where zeros are
CORRECT, from "the store was unreachable", where zeros are a silent lie.
Conflating those caused the bug in §8.19. When the store is unavailable the API
returns **503** rather than quietly serving a worse prediction.

**Known limitation — eviction.** 5.3M hashes nearly fill the cap, so
`allkeys-lru` will evict cold users. That is acceptable for a cache with a
fallback, but the Postgres fallback is degraded (7 days only), so an evicted
user gets different history than training saw. In production you size the store
to the working set, or use DynamoDB, which does not evict.

**Staleness.** Online values are as of the last materialisation, so they exclude
the in-flight session. That is inherent to every online store and bounded by the
materialisation schedule; it is not the same as being wrong.

### 4.10 Training/serving parity

Training features are built in SQL over Parquet; scoring a live session happens
in Python over events in memory. Two implementations of one definition is the
classic setup for training/serving skew — they start identical, someone edits
one, and the model silently receives different inputs in production while
offline metrics keep looking fine.

`tests/test_serving_parity.py` computes features **both ways** over the same
real sessions and asserts every value matches to 1e-9, across 31 features and
50+ sessions. It also asserts the feature **order** matches, since the model
consumes a positional vector.

Reproducing the SQL exactly required matching several behaviours that look like
mistakes and are not:
- `stddev_pop` (population), not sample standard deviation
- DuckDB's `dayofweek()` starts at Sunday, Python's `weekday()` at Monday
- `count(DISTINCT brand)` ignores NULLs, so the Python side must too
- `events_per_minute` is **NULL**, not 0 or infinity, when all k events share a
  timestamp
- the `(event_time, product_id, event_type)` total order from the determinism fix

---

## 5. RESULTS — headline model

Training set: **5,148,713 eligible sessions**, 6.858% positive, 32 features,
1:14 class imbalance.

Splits (chronological): train Oct 1 – Nov 9, valid Nov 10 – 20,
test Nov 21 – 25, drift Nov 26 – 30.

| | ROC-AUC | PR-AUC | recall@10% | lift@10% |
|---|---|---|---|---|
| Random | 0.5003 | 0.079 | 10.0% | 1.00× |
| Heuristic: dwell time | 0.4358 | 0.068 | 8.1% | 0.81× |
| Heuristic: revisit count | 0.6722 | 0.150 | 27.9% | 2.79× |
| Heuristic: cart count | 0.7147 | 0.232 | 42.5% | 4.25× |
| **LightGBM — valid** | 0.8433 | 0.3913 | 53.3% | 5.33× |
| **LightGBM — test** | **0.8406** | **0.4140** | **49.9%** | **4.99×** |
| LightGBM — drift window | 0.8369 | 0.4042 | 49.8% | 4.98× |

**Headline:** contact 10% of traffic, reach **49.9%** of the buyers — 5× better
than random, from the first five events of a session.

### 5.1 Three things to state before being asked

**Compare on PR-AUC, not lift.** Cart-count ranking alone already achieves 4.25×
lift, so the model's 4.99× is only a 17% gain on that metric. The real
separation is PR-AUC: **0.232 → 0.414, a 79% improvement**, which is the metric
that matters at a 6.9% base rate.

**Dwell time is anti-predictive** — AUC 0.4358, *below random*. Slow browsing
converts less; decisive sessions buy. The intuition that longer dwell means
higher intent is simply wrong on this dataset.

**The drift window barely degrades** (0.8406 → 0.8369), consistent with the PSI
finding: the features are not moving, the label rate is.

### 5.2 Feature importance (gain)

| Rank | Feature | Gain |
|---|---|---|
| 1 | `n_cart_events` | 1,189,857 |
| 2 | `user_prior_conv_rate` | 513,033 |
| 3 | `user_prior_revenue` | 149,004 |
| 4 | `price_range` | 142,721 |
| 5 | `user_prior_sessions` | 124,838 |
| 6 | `user_prior_purchases` | 111,280 |
| 7 | `price_std` | 100,891 |
| 8 | `hours_since_last_session` | 76,753 |
| 9 | `n_distinct_categories` | 71,596 |
| 10 | `hour_of_day` | 70,269 |
| 11 | `price_at_cutoff` | 46,718 |
| 12 | `max_views_same_product` | 43,952 |
| 13 | `price_min` | 43,174 |
| 14 | `price_mean` | 34,971 |
| 15 | `max_gap_sec` | 34,832 |

Ranks 2, 3, 5 and 6 are all **point-in-time user history**, confirming the
as-of-session-start windows contribute real signal rather than leaking
current-session outcomes.

### 5.3 Robustness — the model is not a cart-counter

`n_cart_events` carries more than double the gain of any other feature, and cart
recording is inconsistent between months (§7.3). So the same model was retrained
with **every cart signal stripped** (`--hard-mode`):

| | ROC-AUC | PR-AUC | recall@10% | lift@10% |
|---|---|---|---|---|
| Full model | 0.8406 | 0.4140 | 49.9% | 4.99× |
| Hard mode (no cart) | 0.8092 | 0.3394 | 42.8% | 4.28× |

Losing its strongest feature costs 0.031 AUC. The stripped model still reaches
**4.28× lift — marginally better than the 4.25× that cart-count ranking achieves
while still having cart data.** The signal is largely reconstructable from
revisit patterns, price behaviour and prior-session history.

Sanity check: the `heuristic_cart` baseline drops to exactly **AUC 0.5000** under
hard mode, confirming the signals were genuinely removed.

---

## 6. DRIFT — the central finding

### 6.1 Feature drift is essentially zero

PSI between October and late November (bin edges fixed on the October
reference, as a production monitor would):

| Signal | PSI | Convention |
|---|---|---|
| price | 0.0024 | < 0.10 = no shift |
| log price | 0.0025 | 0.10–0.25 = moderate |
| category mix | 0.0116 | > 0.25 = major |

Weekly PSI against a fixed October reference **peaks at 0.0100** — a tenth of
the "moderate shift" threshold.

### 6.2 Label drift is large

Conversion by half-month (from the warehouse):

| Period | Sessions | Conversion | Avg price |
|---|---|---|---|
| Oct 1st half | 4,403,449 | **8.08%** | 291.99 |
| Oct 2nd half | 4,881,378 | 7.93% | 289.22 |
| Nov 1st half | 5,188,213 | **5.97%** | 292.47 |
| Nov 2nd half | 5,339,913 | 6.21% | 279.65 |

A **~26% relative decline**, while average price barely moves. Weekly label
shift peaks at **26.4%** against a 25% alert threshold.

### 6.3 Why this matters

**Feature distributions barely move while the outcome rate moves 26%.** A
monitoring setup watching feature drift would have shown green for two months
while the model decayed. That is the argument for delayed-label performance
monitoring rather than feature monitoring — demonstrated on real data rather
than asserted.

Category mix over the same period is stable, independently confirming it:

| Category | October | Late Nov | Delta |
|---|---|---|---|
| electronics | 55.77% | 51.97% | −3.80% |
| appliances | 17.17% | 19.60% | +2.43% |
| computers | 8.03% | 9.09% | +1.06% |
| apparel | 5.33% | 6.44% | +1.11% |
| furniture | 4.31% | 4.44% | +0.13% |
| auto | 3.50% | 2.69% | −0.81% |

### 6.4 Three signals, three responses

`src/monitoring/drift.py` tracks them separately because they fail differently:

| Signal | Meaning | Response |
|---|---|---|
| **Covariate shift** | inputs move (PSI on features) | often survivable; retrain |
| **Prior / label shift** | outcome rate moves while inputs do not | ranking is probably still sound, calibration is not — **recalibrate**, cheaper and safer than retraining |
| **Data quality** | the pipeline broke | **quarantine and HOLD** — retraining here is actively harmful |

Only the first two are drift. The third is an outage, and treating it as drift
is how 3.3M sessions end up labelled "did not buy".

---

## 7. DATA QUALITY FINDINGS

### 7.1 The collection outage — 2019-11-14 to 11-17

| Date | Views | Carts | Purchases | buy/cart |
|---|---|---|---|---|
| Nov 11–13 (normal) | ~1.9M | ~72K | ~23K | **~32%** |
| Nov 14 | 2,877,130 | 170,472 | 22,124 | **12.98%** |
| **Nov 15** | 5,737,111 | 483,305 | **0** | **0%** |
| Nov 16 | 6,027,932 | 406,778 | 68,247 | **16.78%** |
| Nov 17 | 5,783,241 | 426,941 | 185,195 | **43.38%** |
| Nov 18+ | ~1.9M | ~83K | ~28K | ~34% |

Daily event volume on those days: Nov 15 **6,220,416**, Nov 16 **6,502,957**,
Nov 17 **6,395,377** — against a typical 1.6–2.0M. That is **4.2–4.5×** normal.

**2019-11-15 records zero purchase events all day while 766,025 sessions browse
normally.** The backlog lands on Nov 17, whose buy/cart of 43.38% is inflated.

**Impact if used as-is: 3.3M sessions silently labelled "did not buy" when the
truth is unknowable.** That is label corruption, not class imbalance, and no
amount of model tuning detects it.

Duplicate-row rate on those days is a normal ~0.2%, so this is **not**
duplication — it is missing and misdated purchases.

**Quarantined:** 2019-11-14, 11-15, 11-16, 11-17 (in
`src/platform_core/config.py`). Any session with *any* event on those days is
dropped, because a session straddling the outage has an unknowable label.

**Why volume is the detection rule and ratios are not.** Two ratio-based rules
were tried and both failed:
- A **global** buy/cart median flags nothing, because October (~69%) and
  November (~34%) are structurally different — there is no stable baseline.
- A **local ±7-day** median flags every weekend, because buy/cart dips Fri–Sun
  and a 7-day window mixes weekday and weekend days.

Daily event volume is stable within ±25% across the whole archive, so a 2×
excursion is unambiguous. That is the rule that works.

### 7.2 Black Friday is a damp squib

| Date | Events | Sessions | Purchases | conv/session | Avg price |
|---|---|---|---|---|---|
| Nov 22–28 | ~1.6M | ~380K | ~24K | 5.96 – 6.47% | 275–285 |
| **Nov 29 (Black Friday)** | 1,854,426 | 450,853 | 32,107 | **7.121%** | **285.44** |
| Nov 30 | 1,754,878 | 391,840 | 28,178 | 7.191% | 280.13 |
| Nov 1–20 average | — | — | — | 6.743% | 296.88 |

Conversion is **+13% relative**, not the 3–5× commonly assumed, and the average
price **rose** — there is no discount signature at all. Nov 30 is actually
higher than Black Friday itself.

**The project's drift narrative was originally built around Black Friday and was
rebuilt around the outage and the label-rate decline after the data refuted the
assumption.**

### 7.3 Cart instrumentation differs by month

buy/cart median: **October ~69%, November ~34%.** Cart events are badly
under-recorded in October, so `n_cart_events` means something different in each
month. This is an instrumentation change masquerading as behaviour, and it is
why `--hard-mode` exists.

### 7.4 Sessions spanning multiple users

**939 sessions (0.00408%)** contain events from more than one `user_id` — one
spans three users across 43 events. 2,962 events total; **109 reach the training
table**.

Ownership is genuinely ambiguous. They are attributed to `min(user_id)` —
equally arbitrary but **reproducible** (see §8.5).

Consequence: **534 users appear only as the non-minimum party in such sessions**
and therefore own no session, which is why `dim_user` holds 5,316,115 rows
against 5,316,649 distinct user ids in the raw archive.

### 7.5 NULL user_session

**12 events carry a NULL `user_session`** — all cart events, from 12 unrelated
users. `GROUP BY user_session` merged them into a single phantom session
containing twelve strangers.

They never reached the training table, but **only because `NULL = NULL` is never
true in a join** — luck, not a control. Now excluded explicitly in
`sessionize.py`.

---

## 8. EVERY BUG FOUND, AND WHY EACH MATTERED

These are the most valuable interview material. Every one was **silent** — the
tooling reported success and the breakage appeared somewhere unrelated.

### 8.1 Ambiguous SQL join
`USING (session_key)` broke once two prior tables both exposed that column.
Failed immediately on first run. Fixed with explicit `ON` conditions.

### 8.2 The synthetic fixture was too easy (and was then deleted)
An early synthetic generator gave browsers 2–25s click gaps and buyers 15–120s
— barely overlapping. `max_gap_sec` scored gain 28,231 with everything else
under 60, and LightGBM converged in **5 trees**. One feature was solving the
whole task, so the fixture could no longer reveal a bug in any other feature.

Widening the distributions fixed it (66 trees, model beat the best heuristic
honestly), and the generator was later **removed entirely** in favour of a
committed real slice — see §9.

### 8.3 `count(DISTINCT)` OOM
`count(DISTINCT x) GROUP BY session_key` over 5.15M groups builds one hash set
per group, which DuckDB **cannot spill**. It died at 7.4 GiB.

Fixed with two-stage aggregation (`DISTINCT`, then `COUNT`) — an ordinary hash
aggregate that spills fine — plus dropping large temp tables before the
aggregation stage. It now completes at a **2GB** cap inside the container.

### 8.4 Non-deterministic ranking (the important one)
Events were ordered by `(event_time, product_id)`. That key is **not unique**:
timestamps are second-granular and ~0.2% of rows are exact duplicates. A view
and a purchase of the same product can share a second.

When the sort key ties completely, `row_number()` assigns ranks arbitrarily —
**and not consistently between runs**. So "the first 5 events" meant something
slightly different each time the build ran. The leakage audit caught it as
**one `no_prepurchase` violation in 5,153,372 sessions**.

**Why it matters far beyond one row:** the training set was **not reproducible**.
Rebuild it and you get a subtly different one, so any experiment comparing two
models could be measuring build noise instead of model quality. On a platform
whose premise is continuous retraining, that is corrosive.

**Fix:** collapse exact duplicate events, then order by
`(event_time, product_id, event_type)`, which after dedup is unique within a
session. Guarded by `test_feature_build_is_deterministic`.

**Why the audit caught it:** it was written from the task definition rather than
by reusing the feature builder's SQL. Shared code would have made both sides
agree on the same bug.

### 8.5 Two more arbitrary-tie sources
- `any_value(user_id)` per session — 939 multi-user sessions could change owner
  between runs. Now `min(user_id)`.
- `any_value(category_code)` / `brand` per product in `dim_product`. Now `min()`.

Individually these touch 939 sessions and a handful of products and move no
metric. **Together with §8.4 they meant "rebuild" and "get the same answer" were
different things.**

### 8.6 NULL user_session phantom session
See §7.5.

### 8.7 The logging cycle that broke every Airflow task
`logging_setup.py` attached a `StreamHandler` bound to `sys.stdout`. **Under
Airflow, `sys.stdout` is itself a logger** (`StreamLogWriter`) that forwards
what it receives back into the logging system:

```
log.info() → our handler → writes to sys.stdout
           → sys.stdout IS a logger → logs it
           → our handler → writes to sys.stdout → ...
```

Unbounded cycle → `RecursionError` → **the stack was consumed before any
traceback could be written**, so the task died with an empty log.

**Why it took five runs to find:** the task body ran flawlessly standalone in the
same container (33s, correct verdict) because a terminal's stdout is a real
stream; `airflow tasks test` passed because it does not install the redirect;
and the visible traceback showed only Airflow's `secrets_masker` with none of
the project's code in it. The original frame was at the **top** of the
traceback, not the tail.

**Fix:** a library must not seize the root logger. Attach nothing if the root
logger already has handlers (Airflow, pytest, uvicorn own it); otherwise bind to
`sys.__stdout__`, the original stream captured at interpreter start, which no
framework can replace.

**This broke every task in every DAG**, since all of them import modules that
call `get_logger`.

### 8.8 Missing database credentials in the container
`POSTGRES_USER` / `PASSWORD` / `DB` were not passed to the Airflow service, and
`.env` is not mounted into containers, so `config.py` fell back to its
placeholder defaults. First DAG run died with `password authentication failed`.

### 8.9 dbt cannot live in the Airflow image
Not a pin that can be tightened — a hard contradiction:

| Package | Requires |
|---|---|
| `dbt-core` | `protobuf >= 5` |
| `opentelemetry-proto 1.26.0` (Airflow ships it) | `protobuf < 5` |

Installing dbt pulled protobuf 6.33.6 and broke OpenTelemetry plus ~20
`google-cloud-*` packages.

**Second, independent reason:** DuckDB permits only **one writing process**, so
the container and the host cannot both hold `warehouse.duckdb` open read-write.

**Resolution:** dbt is not in the image; its DAG task **SKIPS with an
explanation** rather than failing, so a green DAG never implies dbt ran when it
did not. In cloud, dbt gets its own image via `KubernetesPodOperator` /
`ECSOperator` — the standard pattern regardless.

### 8.10 Missing `libgomp1`
LightGBM links against the OpenMP runtime, which the slim Airflow image lacks.
`pip install` succeeded and reported success; **`import lightgbm` failed at
runtime** with `OSError: libgomp.so.1: cannot open shared object file` — inside
every training task.

Caught by a build-time smoke test that was written for a *different* problem.
The smoke test does not trust version strings: it builds a DataFrame from a
numpy array and trains a one-round LightGBM model, exercising the actual binary
interop, and **fails the build** if the combination is broken.

Related: pip reports dependency conflicts as **warnings and exits 0**, so a
broken image builds "successfully". `numpy<2` had to be pinned *together with*
`scipy<1.14` in one command, because pinning numpy in an earlier layer did
nothing — pip resolves each `RUN` independently and scipy dragged numpy 2 back
in.

### 8.11 The Redis sink stalled at real volume
`RedisStreamSink.emit` built **one pipeline containing every event in the
window**. At high replay speed a window held ~295,000 events, so redis-py
accumulated 295K commands in memory and the round trip never completed. The
producer looked frozen — clock stuck, `events_emitted = 0` — while some events
had already reached Redis from a partial flush.

**The `events_emitted = 0` was the tell:** it proved execution never got past
`sink.emit()` to `advance_to()`, which ruled out the reader and the clock.

**Fix:** flush every 5,000 commands.

**Why it hid:** the earlier synthetic dataset had ~12K events per window; real
data has ~1.6M events *per day*, so the same code path hit windows 25× larger.
The synthetic fixture could not have caught this.

### 8.12 CDC silently lost 734 events (the most serious bug)
`cdc_extract` filtered on `id > watermark AND event_time <= ceiling`, then
advanced the watermark to the highest id it saw. That is only safe if ids and
timestamps are ordered together — **they are not**. The consumer inserts batched
by session, not chronologically, so a low id can carry a late timestamp. Those
rows are excluded by the ceiling, and the watermark then moves past them
permanently.

**734 of 421,627 events vanished. No error, no warning, both DAG tasks green.**

**Why it matters:** the lake is the durable record — Postgres keeps only 7 days.
Anything the extractor skips is gone for good once retention runs. A 0.17% loss
rate compounding on every extract, invisible.

**Caught by** `scripts/verify_lake.py`, written *after* the run specifically to
compare lake rows against source rows at the watermark rather than trusting
"success".

**Fix:** the watermark may only advance to **one below the first id whose
timestamp exceeds the ceiling**, so nothing beneath it can be unextracted. More
conservative; catches up as the clock advances.

**Self-critique worth stating:** the module docstring already documented the
"monotonic-id trap" and then only half-solved it — a safety lag was added for
late commits, but the assumption that id order implies time order remained.
Writing the caveat down is not the same as fixing it.

### 8.13 CDC could not detect a source reset
After `TRUNCATE ... RESTART IDENTITY`, the stored watermark points past the end
of a table that now begins at 1, so **every subsequent extract silently returns
nothing, forever**.

Fixed: the extractor compares `max(pk)` against the watermark and refuses to run
if the source has gone backwards, with instructions to resolve it.

### 8.14 MLflow registration could never have worked
`mlflow.register_model()` requires a **logged model** created by a flavour's
`log_model()`. This project stores the raw LightGBM booster with
`log_artifact()`, because serving loads the booster file directly rather than
through pyfunc. Registration failed with
`Unable to find a logged_model with artifact_path model_k5.txt` — **after seven
minutes of successful training**.

Fixed with `create_registered_model` + `create_model_version`, which registers a
pointer to the artifact — all the champion/challenger aliases actually need.

### 8.15 Cross-environment artifact paths (green pipeline, unloadable model)
MLflow records an **absolute** artifact location, but the host sees
`D:\...\data` and the container sees `/opt/project/data` for the same directory.
No single `file://` URI is valid in both.

The container inherited an experiment created on the host, wrote the model into
a **literal directory named `D:`** inside the container, and registered a
champion pointing at a path that existed nowhere — **while the DAG reported
success.**

**Fix:** the registry stores metadata plus a filename; the file is resolved
through each environment's own `DATA_ROOT`. Training already writes the booster
to `settings.artifacts_dir`, which both environments resolve correctly.

### 8.16 `version` type mismatch broke every score
MLflow 3 returns `ModelVersion.version` as an **int**; `ScoreResponse.model_version`
is typed `str`. `/model` returns an untyped dict and looked perfectly healthy,
while **every `/score` request failed pydantic validation with a 500**.

A health check would have passed while the service was completely broken — a
concrete argument for separating "is it up?" from "does it work?".

### 8.17 Two guards produced false positives on real data
Both were fixed rather than ignored, because **a guard that flags correct
behaviour is worse than no guard — it trains you to ignore it**.

- The buy/cart data-quality rule flagged every **weekend** (see §7.1).
- `reconcile_window.py` asserted **zero** duplicate events and failed. But the
  event count matched the archive **exactly at 428,668**, and you cannot both
  double-write events and have the total match. Those 367 groups are the
  archive's *own* exact duplicates, faithfully reproduced. The check now
  compares against the archive's duplicate count over the same window.

### 8.18 Replay is not idempotent across overlapping windows
Re-consuming an overlapping replay window produced **29,693 duplicate
(session, time, type, product) groups**. `seq` comes from a running counter, so a
re-delivered event gets a fresh sequence number and inserts again.

**Content-based dedup cannot fix it** — the archive genuinely contains ~0.2%
exact-duplicate events, so a natural-key unique constraint would reject
legitimate rows.

**The correct semantic:** a replay is a re-run of the simulation, so the hot
window must be cleared first. Now a `--truncate` flag with the reasoning in the
class docstring.

### 8.19 Serving scored every user as brand new
`ScoreRequest` declared `prior_sessions` / `prior_purchases` / `prior_revenue`
with **default 0**, and the API never looked them up. `scripts/score_demo.py`
did not send them, so **every session ever scored was treated as a brand-new
user with no history** — while `user_prior_conv_rate` is the **second most
important feature by gain (513,033)**.

**Why the parity test missed it.** `test_serving_parity.py` passes history in
explicitly, so it verified that the **computation** matches training — not that
production **supplies** the inputs. Two different guarantees; only one was
being checked. A parity test proves your two implementations agree; it says
nothing about whether the caller populates them.

**Measured impact**, identical 85 sessions:

| | Before | After |
|---|---|---|
| mean score, actually bought | 0.1271 | **0.2739** |
| mean score, did not buy | 0.0637 | 0.0691 |
| **separation** | +0.0634 | **+0.2048** |

**3.2× better separation.** The model had been running with its second-strongest
feature pinned at zero.

**Fix:** the online feature store (§4.11), plus request fields defaulting to
`None` so "caller did not supply" is distinguishable from "caller supplied
zero", plus `history_source` on every response.

**How it was found:** not by a test. By someone asking *"where are we physically
storing our features?"* — a question about architecture that turned out to be a
question about correctness.

---

## 9. TESTING — 19 tests, and the reasoning

**There is no synthetic data in this project.** Tests run on:
- a **21-event hand-written fixture** in `tests/conftest.py` whose every expected
  feature value was worked out by hand
- **`tests/fixtures/events/`** — a committed 0.7 MB slice of the real archive
  (34,844 events, 7,858 sessions, 5.205% purchase rate, 2,256 sessions eligible
  at k=5), carved by `scripts/make_test_fixture.py`

The fixture samples **by session, never by event** — splitting a session mid-way
would corrupt every feature that depends on within-session ordering. Selection is
a deterministic hash of `user_session`, so the same sessions are chosen on every
machine.

**Why the synthetic generator was deleted:** generated data encodes the
generator author's assumptions, which is exactly what tests are supposed to
catch. See §8.2.

### Key tests

| Test | What it protects |
|---|---|
| `test_post_cutoff_cart_is_invisible` | a post-cutoff cart reaching the features |
| `test_audit_has_teeth` | corrupts a label, asserts the audit fails |
| `test_feature_build_is_deterministic` | builds twice, asserts identical output |
| `test_online_features_match_training_sql` | 31 features × 50+ sessions to 1e-9 |
| `test_vector_order_matches_training` | model consumes a positional vector |
| `test_clock_bound_is_enforced` | no row beyond the replay clock |
| `test_split_windows_do_not_overlap` | chronological splits stay disjoint |

### The eight guards

Every one exists because something looked fine and was not. **None of these
failures announce themselves.**

| Guard | Silent failure it catches |
|---|---|
| `features/leakage_audit.py` | model trained on data it could not have seen |
| `test_feature_build_is_deterministic` | training set changes between builds |
| `test_serving_parity.py` | serving features drift from training features |
| `docker/airflow/smoke_test.py` | pip reports success, import fails at runtime |
| `scripts/data_quality_audit.py` | source pipeline breaks, labels become fiction |
| `quarantine_and_hold` branch | drift monitor retrains on corrupted data |
| `scripts/verify_lake.py` | CDC skips rows and advances past them anyway |
| `scripts/reconcile_window.py` | replay writes a different dataset than the source |

---

## 10. VERIFIED EXECUTION EVIDENCE

Every DAG has been run against the real archive. None of this is claimed on the
strength of the code alone.

### `continuous_training` — all six tasks green

```
read_clock       success
build_features   success   ~7 min, 4,495,843 eligible sessions, clock-bounded
leakage_audit    success   6/6 checks, 0 violations
train            success   test ROC-AUC 0.8423
register         success   purchase_intent version 1
promote          success   "no incumbent champion; promoting on test_roc_auc=0.8423"
```

Splits in that run (clock at 2019-11-24 08:00): train Oct 1 – Nov 5 (2,994,740
rows, 6.833% pos), valid Nov 6 – 11 (665,077, 5.936%), test Nov 12 – 20
(540,888, 6.582%), drift Nov 21 – 24 (295,138, 7.932%).

Build stage timings inside the container: candidates 3.1s → rank events 219.8s →
pre-cutoff purchase rule 8.2s → split prefix/label 42.4s → release ranked 0.2s →
distinct counts 33.4s → aggregate 25.4s → assemble and write.

Leakage audit output:
```
prefix_size          PASS
cutoff_is_last_seen  PASS
no_prepurchase       PASS
label_provenance     PASS
no_future_events     PASS      ← only runs when a clock bound is set
no_forbidden_cols    PASS
```

### `warehouse_refresh`

`cdc_extract` (7s) → `build_warehouse` (85s) → `dbt_build` **skipped** with its
explanation. Lake verified afterwards:

```
session_events   lake=418,637  distinct=418,637  ids=1..418637  watermark=418637  OK
orders           lake=  6,741  distinct=  6,741  ids=1..6741    watermark=6741    OK
predictions      lake=    104  distinct=    104  ids=1..105     watermark=105     OK
```

Contiguous ids with matching counts: every row below the watermark present
exactly once.

### Replay → OLTP, reconciled

Full-day replay: **1,591,765 events** emitted (exactly matching the archive's
count for 2019-11-24) at ~19,600 events/sec. A clean 8-hour replay of 428,668
events at 13,088 events/sec, then reconciled:

| Metric | Archive | OLTP | Delta |
|---|---|---|---|
| events | 428,668 | 428,668 | **0** |
| sessions | 101,379 | 101,379 | **0** |
| users | 75,997 | 75,997 | **0** |
| products | 46,420 | 46,420 | **0** |
| sessions that purchased | 6,114 | 6,114 | **0** |
| sessions that carted | 12,240 | 12,240 | **0** |
| duplicate groups | 367 | 367 | **0** |

### Serving

```
/model → {"version":"1","source":"mlflow:champion","num_trees":296,"num_features":32,"k":5}
```

**296 trees confirms this is the DAG-trained model, not the earlier local
artifact (309 trees)** — proof that promotion reaches serving.

Scoring 85 real sessions: **p50 0.56 ms, p95 0.61 ms, p99 3.71 ms.**
An earlier run over 104 requests: p50 0.61 ms, p95 0.75 ms, **p99 0.90 ms**.

Separation on real sessions with known outcomes, **after** the online feature
store was added (§4.11 / §8.19):
- mean score, actually bought: **0.2739** (n=35)
- mean score, did not buy: **0.0691** (n=50)
- separation **+0.2048** — buyers score ~4× higher

Before the fix, with user history silently defaulted to zero, the same 85
sessions gave 0.1271 / 0.0637 and a separation of only +0.0634. Latency p50
moved 0.56ms → 1.45ms to pay for the Redis lookup.

### `drift_monitor` — proven in BOTH directions

On the outage window (clock 2019-11-17):
```
run_check            success   31.5s over 109,950,743 events
route_on_verdict     success   "Branch into quarantine_and_hold"
trigger_retrain      skipped
recalibrate          skipped
no_action            skipped
quarantine_and_hold  failed    ← intended
```

The monitor's own output in that run:
```
as_of=2019-11-17 verdict=DATA_QUALITY_INCIDENT action=quarantine_and_hold
label_rate=0.07255 baseline=0.07381 shift=-0.017     ← looks HEALTHY
psi[price]=0.0060                                     ← NO feature drift
reason: 2019-11-15: ZERO purchase events recorded
reason: 2019-11-14: volume 2.1x the reference median
reason: 2019-11-15: volume 4.3x the reference median
reason: 2019-11-16: volume 4.5x the reference median
reason: 2019-11-17: volume 4.4x the reference median
```

**Read those three signals together.** Label drift says −1.7% (fine). Feature
PSI says 0.0060 (no shift). A monitor watching either — or both — would have
declared the platform healthy and retrained straight into four days where
purchases were never recorded. **Only the data-quality signal caught it**,
because Nov 17's backfill masks the missing purchases.

On a clean window (clock 2019-11-28) the same DAG routes to `no_action` and
finishes green.

**Note on the 7-day lookback:** an incident keeps firing for a week after the
source recovers, because the window still contains the bad days. That is correct
— you should not retrain on a window containing corrupted labels — but it is
worth knowing before someone asks why it is still red.

---

## 11. REPOSITORY LAYOUT

```
src/
  platform_core/     config and logging; every path resolves through here
    config.py        Settings (pydantic); quarantine_dates; derived paths
    logging_setup.py the sys.__stdout__ fix (§8.7)
  ingest/
    download.py      Kaggle download, one file at a time to bound disk
    csv_to_parquet.py streaming convert; sniffs the " UTC" suffix; verifies
                     per-file row counts before deleting the source
    validate.py      data profile → data_profile.json
  features/
    sessionize.py    session index + point-in-time user history
    build_features.py truncation-point features; staged temp tables
    leakage_audit.py independent audit, written from the definition
  models/
    train.py         chronological split, baselines, LightGBM, metrics
    metrics.py       recall@k / lift@k — business-facing, not just AUC
    registry.py      MLflow tracking + champion/challenger promotion
  replayer/
    clock.py         PostgresClock / InMemoryClock; assert_visible guard
    sinks.py         EventSink: redis / kafka / null; chunked flush (§8.11)
    replay.py        backfill + live phases
    storefront_consumer.py  applies events as OLTP transactions; --truncate
  warehouse/
    cdc_extract.py   incremental drain with safety lag + reset detection
    build_warehouse.py star schema builder
  monitoring/
    drift.py         covariate / label / data-quality signals
  serving/
    features_online.py Python twin of the training SQL
    feature_store.py online store: warehouse -> Redis -> Postgres fallback
    api.py           FastAPI scoring service

dags/
  dag_continuous_training.py   leakage audit as a hard gate
  dag_drift_monitor.py         hourly, four-way branch
  dag_warehouse_refresh.py     CDC → warehouse → dbt (skips)

dbt/
  models/marts/mart_daily_conversion.sql   trailing baseline, DQ flag
  models/marts/mart_user_segments.sql      RFM
  tests/assert_outage_days_are_flagged.sql singular regression test

docker/
  docker-compose.yml           ecomml-* naming; profiles for airflow/tools/kafka
  postgres/init.sql            13-table OLTP schema
  airflow/Dockerfile           one package per layer; numpy<2 pin; libgomp1
  airflow/smoke_test.py        build-time interop check

scripts/
  gate1_signal_check.py        is the task viable at all
  data_quality_audit.py        finds broken days
  drift_scan.py                PSI + daily series
  make_incident_chart.py       the headline figure
  make_test_fixture.py         carves the committed real slice
  reconcile_oltp.py            full-archive reconciliation
  reconcile_window.py          windowed reconciliation
  verify_lake.py               lake vs source at the watermark
  registry_status.py           what is in MLflow right now
  warehouse_demo.py            queries Postgres cannot answer
  score_demo.py                scores real sessions, measures latency
  md_to_pdf.py                 renders this document to PDF
  fetch_wheels.ps1             resumable wheel downloads

tests/                         19 tests
infra/AWS_DEPLOYMENT.md        designed and costed, NOT deployed
reports/figures/drift_incident.png
```

---

## 12. AWS DESIGN (NOT DEPLOYED)

| Local | AWS | Why |
|---|---|---|
| Parquet archive | S3 (Standard-IA after 30d) | immutable source of truth |
| `ecomml-postgres` | RDS Postgres `db.t4g.micro` | Graviton ~20% cheaper |
| Redis Streams | Kinesis Data Streams | same append-log semantics |
| DuckDB warehouse | Athena over S3 + Iceberg | 2.3 GB does not justify Redshift |
| dbt-duckdb | dbt-athena | same models, different adapter |
| `ecomml-airflow` | ECS Fargate Spot | MWAA is ~$350/mo minimum |
| MLflow | Fargate + RDS backend | registry needs a real DB |
| FastAPI | ECS Fargate behind an ALB | p99 <1ms — one small task carries it |

**~$61/month as specified.** Reducible to **~$24**: drop the ALB (−$16.20, a
Fargate task with a public IP and API-key auth is adequate for a demo), SQS
instead of Kinesis (−$10.50, ordering per session is what matters and there is
one consumer), schedule Airflow off outside demo hours (−$3.50).

**Where each cheap choice breaks:** Athena past ~1 TB scanned/month;
single-container Airflow the moment more than one person depends on it (no HA,
loses queued task state); SQS as soon as a second consumer needs the stream or
replay from an offset; single-AZ RDS the moment anything real depends on it.

**Why it is not deployed:** it costs real money on a personal account, and it
would not prove any technical claim — the model results, leakage audit, drift
detection, parity guarantee and sub-millisecond serving are all demonstrated
locally against the full dataset. Running the same containers on Fargate would
demonstrate that Fargate runs containers. A deploy adds a live URL, which has
genuine portfolio value but is not the same as a technical claim.

---

## 13. KNOWN LIMITATIONS

State these before being asked.

1. **No treatment data.** The dataset records what shoppers did, not what the
   store did to them. That supports purchase **propensity** but not **uplift**
   ("would buy only if nudged"), which needs randomised treatment assignment.
   The OLTP schema includes `interventions.is_holdout` for the untreated control
   group such a design requires.

2. **Offline ranking metrics are biased** by whatever ranker produced the logged
   behaviour. Counterfactual estimation (IPS) is the honest fix and is not
   implemented.

3. **The clock is simulated.** Event-time replay of a historical archive, never
   live traffic.

4. **939 sessions span multiple users** (§7.4), attributed to `min(user_id)`;
   534 users consequently own no session.

5. **dbt does not run inside the Airflow container** (§8.9); its task skips with
   an explanation rather than failing.

6. **The k=5 operating point covers 28% of traffic.** Sessions shorter than 5
   events cannot be scored at all — that is inherent to the task definition, not
   a defect, but it bounds the addressable population.

7. **Cart features are not comparable across months** (§7.3); `--hard-mode`
   quantifies the dependence.

8. **The online feature store can evict.** 5.3M user hashes nearly fill Redis's
   768MB cap, so `allkeys-lru` drops cold users; the Postgres fallback only sees
   7 days, so an evicted user is scored with a shorter history than training
   saw. Production sizes the store to the working set or uses DynamoDB.

9. **Kafka is unverified.** `KafkaSink` exists but `kafka-python` is not
   installed and the path has never run (§4.4). Redis Streams is what executed.

---

## 14. ENVIRONMENT CONSTRAINTS (context for odd decisions)

- **PyPI throughput ~100 KB/s** on the development machine. pip cannot resume,
  so large wheels restart from zero. The venv uses `--system-site-packages` on
  Python 3.13 to inherit already-installed heavy wheels, and
  `scripts/fetch_wheels.ps1` uses `curl -C -` for the rest. The Airflow image
  installs one package per `RUN` layer so a timeout only retries the failing one.
- **`data.rees46.com` serves an expired TLS certificate.** Verification was not
  disabled; Kaggle used instead.
- **Disk:** C: ~12 GB free, D: ~41 GB. All data on D:. Docker images live on C:.
- **RAM:** 15.3 GB total, but only ~4–6 GB free in practice. DuckDB memory limit
  set to 4 GB on host / 2 GB in container, relying on disk spilling (peaks
  ~8.5 GB of spill during the feature build).
- **CPU:** 16 logical cores.

**Key versions:** Python 3.13 (host) / 3.12 (container), DuckDB 1.5.5,
Polars 1.43.2, LightGBM 4.6.0, MLflow 3.15.1, Airflow 2.10.0, dbt-core 1.12.0,
dbt-duckdb 1.11.0, numpy 1.26.4 and pandas 2.1.4 in the container (numpy pinned
<2 — see §8.10).

---

## 15. HOW TO RUN IT

```bash
# services — everything this project owns is prefixed ecomml-
cd docker && docker compose up -d                    # postgres + redis
cd docker && docker compose --profile airflow up -d  # + airflow on :8080
docker ps --filter "name=ecomml-"

# data
python -m src.ingest.download --list
python -m src.ingest.download
python -m src.ingest.csv_to_parquet --delete-source
python -m src.ingest.validate

# gates — run before trusting anything
python -m scripts.gate1_signal_check
python -m scripts.data_quality_audit

# pipeline
python -m src.features.sessionize
python -m src.features.build_features --k 5
python -m src.features.leakage_audit --k 5      # hard gate
python -m src.models.train --k 5

# warehouse
python -m src.warehouse.build_warehouse
cd dbt && dbt build --profiles-dir .
python -m scripts.warehouse_demo

# replay (clear the hot window first — replay re-runs the simulation)
python -m src.replayer.storefront_consumer --truncate    # terminal 1
python -m src.replayer.replay --sink redis --clock postgres \
    --speed 1500 --until 2019-11-25T00:00:00              # terminal 2
python -m scripts.reconcile_window --from "2019-11-24 00:00:00" \
                                   --to   "2019-11-25 00:00:00"

# online feature store — MUST be materialised before serving, or every
# request is scored as a brand-new user (see 8.19)
python -m src.serving.feature_store --materialize
python -m src.serving.feature_store --probe 520088904

# serving (port 8500; 8000 may be taken locally)
python -m uvicorn src.serving.api:app --port 8500
python -m scripts.score_demo

# monitoring
python -m src.monitoring.drift --as-of 2019-11-17
python -m scripts.make_incident_chart

python -m pytest -q
```

**Ports:** Airflow 8080, serving API 8500, Postgres 5433, Redis 6379.

---

## 16. CURRENT STATE — READ THIS FIRST IN A NEW SESSION

Everything below was true at the last commit. Verify rather than assume: the
platform runs on a laptop, so containers may be stopped and the machine may have
slept.

### What exists and is verified

| Component | State |
|---|---|
| Archive | `data/parquet/events/` — 109,950,743 events, 61 date partitions, 1.9 GB |
| Session index | `data/features/sessions.parquet` — 869 MB |
| Training table | `data/features/train_k5.parquet` — 328 MB (also `train_k5_hard.parquet`, 367 MB) |
| Warehouse | `data/warehouse.duckdb` — star schema, ~2.3 GB |
| CDC lake | `data/parquet/lake/` — session_events, orders, predictions |
| MLflow | `data/mlflow.db` — `purchase_intent` **version 1**, aliases `champion` and `challenger` both → v1 |
| Model artifact | `data/artifacts/model_k5.txt` — 296 trees, 32 features |
| Online store | Redis `feat:user:*` — 5,316,115 users materialised |
| Replay clock | last set to **2019-11-28** (a clean window) |
| OLTP | populated from a clean 8-hour replay of 2019-11-24 (428,668 events) |
| Tests | **19 passing** |
| dbt | **26/26** |

### Containers and processes

```bash
docker ps --filter "name=ecomml-"      # ecomml-postgres :5433, ecomml-redis :6379, ecomml-airflow :8080
```

The serving API runs on the **host**, not in a container, on port **8500**
(8000 is occupied by an unrelated `business-intelligence` service on this
machine). It is started manually and does not survive a reboot:

```bash
python -m uvicorn src.serving.api:app --port 8500
```

**Airflow DAG state:** `drift_monitor` is unpaused and runs **hourly**, scanning
110M events each time (~30s). `continuous_training` and `warehouse_refresh` are
unpaused but scheduled infrequently. Pause them if the machine is needed for
anything else:

```bash
docker exec ecomml-airflow airflow dags pause drift_monitor
```

### To bring everything back up

```bash
cd docker && docker compose --profile airflow up -d
cd .. && python -m uvicorn src.serving.api:app --port 8500 &
python -m src.serving.feature_store --materialize     # if Redis was wiped
python -m pytest -q                                    # expect 19 passed
```

Redis has `--save ""` and `--appendonly no`, so **the online store and the event
stream do not survive a container restart** — re-materialise after any restart.
That is deliberate (both are derived data) but it will surprise you once.

### Traps that will bite again

1. **Unpausing a cron DAG fires it immediately** for the most recent missed
   interval. This once raced a truncate and contaminated the CDC lake.
2. **Replay is not idempotent.** Always run the consumer with `--truncate`
   before a replay, or you get duplicate events (§8.18).
3. **The drift monitor's 7-day lookback** keeps an incident firing for a week
   after the source recovers. At clock = 2019-11-24 the window still contains
   the 11-17 outage, so it correctly quarantines. Use clock ≥ 2019-11-25 for a
   clean window.
4. **DuckDB permits one writer.** Nothing else may hold `warehouse.duckdb` open
   read-write while `build_warehouse` or dbt runs.
5. **~100 KB/s to PyPI.** Any new dependency is slow; check whether Python 3.13
   system site-packages already has it before installing.
6. **The machine sleeping kills background jobs.** A 9-hour gap once ate a
   feature build.

### Open items, roughly by value

1. **Re-run `continuous_training` end to end** now that the online store exists,
   so the registered model and the serving path are verified together in one
   run. The current champion predates the feature-store fix; nothing is wrong
   with it, but the two have not been exercised in a single pass.
2. **Add a test that catches §8.19's class of bug** — assert that a `/score`
   request *without* history fields returns `history_source != "absent"` for a
   user known to have history. The parity test cannot catch this by design.
3. **Verify or delete the Kafka sink.** It has never run. Either install
   `kafka-python`, bring up the `kafka` profile and test it, or remove the code
   so the repo does not imply capability it lacks.
4. **Feast or an equivalent feature registry**, if you want the feature store to
   be more than a hand-rolled cache.
5. **Counterfactual evaluation (IPS)** for the ranking side — the honest fix for
   the bias noted in §13.
6. **AWS deploy** (§12) if a live URL is wanted. ~$24/month optimised. Adds no
   technical claim, only a demo link.

### Things NOT to do

- Do **not** claim AWS deployment, Kafka experience, or a live URL.
- Do **not** re-add synthetic data. Tests run on a committed real slice plus a
  21-event hand-written fixture, deliberately (§9).
- Do **not** quote the old serving separation of +0.0634; it is +0.2048 since
  the feature-store fix.

---

## 17. THE THROUGH-LINE

If there is one thing to take from this project, it is that **the failures that
matter are silent.**

Every bug in §8 reported success at the point of failure:
- pip said the install worked; `import lightgbm` died at runtime
- both DAG tasks said success while 734 events were permanently lost
- the DAG registered a champion whose artifact existed nowhere
- `/model` returned healthy while every `/score` returned 500
- the training set was irreproducible and every metric looked fine

That is why the project has eight independent guards rather than a coverage
percentage, and why two of them were fixed after producing **false positives** on
real data — a guard that flags correct behaviour is worse than no guard, because
it teaches you to ignore it.

It is also why the distinction between *"the code is written"* and *"it runs"*
was tracked separately throughout. Six of the bugs above were only findable by
executing the thing.
