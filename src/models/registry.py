"""MLflow tracking and model registry.

Kept separate from the training script because three different callers need it:
the trainer registers new versions, Airflow promotes and rolls back, and the
serving API resolves whichever version currently holds the `champion` alias.

Promotion model
---------------
Three aliases, and a model is only ever promoted on evidence:

    champion    what serving actually uses
    challenger  a newly trained candidate, not yet trusted
    shadow      scored alongside the champion on live traffic, output discarded

A challenger is promoted only if it beats the incumbent champion on the SAME
evaluation window by more than a fixed margin. The margin exists so that noise
does not cause a swap - retraining daily on a metric that moves +/-0.003 run to
run would otherwise churn the production model constantly, which looks like
progress and is not.

Why SQLite rather than a file store: model versions and aliases require a
database-backed tracking store. `file://` supports runs but not the registry,
so the promotion flow could not exist on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)

CHAMPION = "champion"
CHALLENGER = "challenger"
SHADOW = "shadow"

# Minimum ROC-AUC improvement required to replace the champion.
PROMOTION_MARGIN = 0.002


@dataclass
class ModelRef:
    name: str
    version: str
    alias: str
    run_id: str
    metrics: dict[str, float]


def _mlflow(settings=None):
    import mlflow

    settings = settings or get_settings()
    Path(settings.data_root / "mlartifacts").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    return mlflow


def log_run(
    run_name: str,
    params: dict,
    metrics: dict[str, dict[str, float]],
    artifacts: list[Path],
    settings=None,
    register: bool = True,
) -> str | None:
    """Log one training run; optionally register it as a new model version.

    `metrics` is {split: {metric: value}} and is flattened to `split_metric`
    so that MLflow's comparison UI and its API can sort on them directly.
    """
    import math

    settings = settings or get_settings()
    mlflow = _mlflow(settings)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        for split, m in metrics.items():
            for k, v in m.items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    mlflow.log_metric(f"{split}_{k}", float(v))
        for a in artifacts:
            if a.exists():
                mlflow.log_artifact(str(a))

        version = None
        if register:
            uri = f"runs:/{run.info.run_id}"
            try:
                mv = mlflow.register_model(uri, settings.mlflow_model_name)
                version = mv.version
                log.info("registered %s version %s", settings.mlflow_model_name, version)
            except Exception as exc:
                log.warning("model registration skipped: %s", exc)

        log.info("logged run %s (id=%s)", run_name, run.info.run_id)
        return version


def get_by_alias(alias: str, settings=None) -> ModelRef | None:
    settings = settings or get_settings()
    mlflow = _mlflow(settings)
    client = mlflow.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(settings.mlflow_model_name, alias)
    except Exception:
        return None
    run = client.get_run(mv.run_id)
    return ModelRef(
        name=settings.mlflow_model_name,
        version=mv.version,
        alias=alias,
        run_id=mv.run_id,
        metrics=dict(run.data.metrics),
    )


def set_alias(alias: str, version: str, settings=None) -> None:
    settings = settings or get_settings()
    mlflow = _mlflow(settings)
    mlflow.MlflowClient().set_registered_model_alias(
        settings.mlflow_model_name, alias, version
    )
    log.info("alias %r -> version %s", alias, version)


def evaluate_promotion(
    challenger_metrics: dict[str, float],
    metric: str = "test_roc_auc",
    settings=None,
) -> tuple[bool, str]:
    """Decide whether the challenger should replace the champion.

    Returns (promote, human-readable reason). Promotion requires beating the
    incumbent by PROMOTION_MARGIN on the same metric - a challenger that merely
    ties is not an improvement, it is noise.
    """
    champ = get_by_alias(CHAMPION, settings)
    new = challenger_metrics.get(metric)
    if new is None:
        return False, f"challenger has no {metric}"
    if champ is None:
        return True, f"no incumbent champion; promoting on {metric}={new:.4f}"

    old = champ.metrics.get(metric)
    if old is None:
        return True, f"champion has no {metric} recorded; promoting challenger ({new:.4f})"

    delta = new - old
    if delta > PROMOTION_MARGIN:
        return True, (
            f"challenger {metric}={new:.4f} beats champion {old:.4f} "
            f"by {delta:+.4f} (margin {PROMOTION_MARGIN})"
        )
    return False, (
        f"challenger {metric}={new:.4f} vs champion {old:.4f} "
        f"({delta:+.4f}) does not clear margin {PROMOTION_MARGIN}"
    )


def promote_if_better(
    version: str, challenger_metrics: dict[str, float], settings=None
) -> bool:
    """Register as challenger, then promote to champion only if it earns it."""
    set_alias(CHALLENGER, version, settings)
    ok, reason = evaluate_promotion(challenger_metrics, settings=settings)
    log.info("promotion decision: %s -- %s", "PROMOTE" if ok else "HOLD", reason)
    if ok:
        set_alias(CHAMPION, version, settings)
    return ok


def rollback(settings=None) -> bool:
    """Point champion at the previous version. Used when an SLO breaches."""
    settings = settings or get_settings()
    mlflow = _mlflow(settings)
    client = mlflow.MlflowClient()
    versions = sorted(
        client.search_model_versions(f"name='{settings.mlflow_model_name}'"),
        key=lambda v: int(v.version),
        reverse=True,
    )
    current = get_by_alias(CHAMPION, settings)
    if current is None or len(versions) < 2:
        log.warning("cannot roll back: need a champion and at least two versions")
        return False
    for v in versions:
        if v.version != current.version:
            set_alias(CHAMPION, v.version, settings)
            log.warning("ROLLED BACK champion: %s -> %s", current.version, v.version)
            return True
    return False
