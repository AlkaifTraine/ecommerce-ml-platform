"""Scoring API.

Serves whichever model version currently holds the `champion` alias in the
MLflow registry, so promotion and rollback take effect without redeploying.

Design points worth defending:

* **Features are computed by `features_online`, never re-implemented here.**
  That module is held to the training SQL by `tests/test_serving_parity.py`.
* **Predictions are logged before the response is returned**, including the
  feature snapshot. Without the inputs recorded, a later "why did it score
  0.9?" is unanswerable, and delayed-label monitoring has nothing to join to.
* **Logging failures never fail a request.** A monitoring outage must not take
  the storefront down with it.
* **`scored_at_data_time` is the replay clock**, not wall time, so monitoring
  joins line up with the simulated calendar.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.platform_core import get_logger, get_settings
from src.serving.features_online import UserHistory, compute, to_vector

log = get_logger(__name__)

K = 5
STATE: dict[str, Any] = {"model": None, "version": "unloaded", "source": "none"}


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------
class Event(BaseModel):
    event_time: datetime
    event_type: str = Field(pattern="^(view|cart|remove_from_cart|purchase)$")
    product_id: int
    category_id: int | None = None
    category_code: str | None = None
    brand: str | None = None
    price: float | None = None


class ScoreRequest(BaseModel):
    session_key: str
    user_id: int
    events: list[Event]
    prior_sessions: int = 0
    prior_purchases: int = 0
    prior_revenue: float = 0.0
    hours_since_last_session: float | None = None


class ScoreResponse(BaseModel):
    session_key: str
    score: float
    model_version: str
    model_source: str
    latency_ms: float
    decision: str
    reason: str


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------
def _load_model() -> None:
    """Champion from the registry; fall back to the artifact on disk.

    The fallback exists so the API still starts when the registry is
    unreachable - refusing to serve because a tracking server is down would
    make monitoring infrastructure a hard dependency of the storefront.
    """
    import os

    import lightgbm as lgb

    settings = get_settings()

    # Bound the registry lookup. MLflow's default is 7 retries with exponential
    # backoff, so an unreachable tracking URI stalls startup for ~4.5 minutes
    # before the fallback below is even reached. A model server must fail over
    # in seconds, not minutes.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")

    try:
        from src.models.registry import CHAMPION, get_by_alias

        ref = get_by_alias(CHAMPION, settings)
        if ref is not None:
            # Resolve the model file through OUR OWN artifacts_dir rather than
            # downloading the run's artifact URI. MLflow stores an absolute
            # location, and the host and container mount the same directory at
            # different absolute paths - downloading by URI gets a path that
            # exists in the other environment, or nowhere. The registry supplies
            # the version and the filename; the bytes come from local disk.
            path = settings.artifacts_dir / ref.model_filename
            if not path.exists():
                raise FileNotFoundError(
                    f"champion version {ref.version} names {ref.model_filename}, "
                    f"which is not present at {path}"
                )
            # str(): MLflow 3 returns ModelVersion.version as an int, while
            # ScoreResponse.model_version is typed str. Left uncoerced, /model
            # (an untyped dict) works fine and every /score request fails
            # pydantic validation with a 500 - so the API looks healthy right
            # up until it is asked to do its job.
            STATE.update(model=lgb.Booster(model_file=str(path)),
                         version=str(ref.version), source="mlflow:champion")
            log.info("loaded champion version %s from %s", ref.version, path)
            return
    except Exception as exc:
        log.warning("registry unavailable (%s); falling back to local artifact", exc)

    local = settings.artifacts_dir / f"model_k{K}.txt"
    if not local.exists():
        raise RuntimeError(f"no champion in registry and no local model at {local}")
    STATE.update(model=lgb.Booster(model_file=str(local)),
                 version="local", source=f"file:{local.name}")
    log.info("loaded local model %s", local)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    STATE["model"] = None


app = FastAPI(title="Purchase Intent API", version="1.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------
# prediction logging
# --------------------------------------------------------------------------
def _log_prediction(req: ScoreRequest, features: dict, score: float, latency_ms: float) -> None:
    """Best effort. A monitoring outage must not break scoring."""
    import json

    import psycopg2

    settings = get_settings()
    try:
        conn = psycopg2.connect(settings.postgres_dsn, connect_timeout=2)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT current_data_time FROM storefront.replay_clock WHERE id = 1")
        row = cur.fetchone()
        data_time = row[0] if row else datetime.utcnow()
        cur.execute(
            """
            INSERT INTO storefront.predictions
                (session_key, model_name, model_version, model_alias, cutoff_seq,
                 scored_at_data_time, score, latency_ms, feature_snapshot)
            VALUES (%s, %s, %s, 'champion', %s, %s, %s, %s, %s)
            ON CONFLICT (session_key, model_alias, cutoff_seq) DO NOTHING
            """,
            (req.session_key, settings.mlflow_model_name, STATE["version"], K,
             data_time, float(score), latency_ms, json.dumps(features, default=str)),
        )
        cur.close()
        conn.close()
    except Exception as exc:
        log.warning("prediction logging failed (scoring unaffected): %s", exc)


def _decide(score: float) -> tuple[str, str]:
    """Turn a probability into the action the business actually takes.

    Measured on the archive: the base rate is 6.9% and the model reaches 49.9%
    of buyers in the top decile. Spending on the highest scores is wasteful -
    those sessions largely convert anyway - so the budget goes to the middle
    band, where an intervention can still change the outcome.
    """
    if score >= 0.60:
        return "no_intervention", "likely to convert unaided - do not spend"
    if score >= 0.15:
        return "intervene", "persuadable band - discount or free shipping"
    return "no_intervention", "unlikely to convert - not worth the spend"


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if STATE["model"] is not None else "no_model",
        "model_version": STATE["version"],
        "model_source": STATE["source"],
    }


@app.get("/model")
def model_info() -> dict:
    if STATE["model"] is None:
        raise HTTPException(503, "no model loaded")
    b = STATE["model"]
    return {
        "version": STATE["version"],
        "source": STATE["source"],
        "num_trees": b.num_trees(),
        "num_features": b.num_feature(),
        "k": K,
    }


@app.post("/reload")
def reload_model() -> dict:
    """Pick up a promotion or rollback without a redeploy."""
    _load_model()
    return {"reloaded": True, "model_version": STATE["version"], "source": STATE["source"]}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    if STATE["model"] is None:
        raise HTTPException(503, "no model loaded")

    t0 = time.perf_counter()
    try:
        feats = compute(
            events=[e.model_dump() for e in req.events],
            k=K,
            history=UserHistory(
                prior_sessions=req.prior_sessions,
                prior_purchases=req.prior_purchases,
                prior_revenue=req.prior_revenue,
                hours_since_last_session=req.hours_since_last_session,
            ),
        )
    except ValueError as exc:
        # Not an error: the session is simply not scoreable yet.
        raise HTTPException(422, str(exc)) from exc

    p = float(STATE["model"].predict([to_vector(feats)])[0])
    latency_ms = (time.perf_counter() - t0) * 1000.0
    decision, reason = _decide(p)

    _log_prediction(req, feats, p, latency_ms)

    return ScoreResponse(
        session_key=req.session_key,
        score=p,
        model_version=STATE["version"],
        model_source=STATE["source"],
        latency_ms=round(latency_ms, 3),
        decision=decision,
        reason=reason,
    )
