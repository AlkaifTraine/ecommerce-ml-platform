"""What is actually in the MLflow registry right now."""

from __future__ import annotations

import mlflow

from src.models.registry import CHALLENGER, CHAMPION, get_by_alias
from src.platform_core import get_logger, get_settings

log = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_uri)
    client = mlflow.MlflowClient()

    log.info("tracking uri: %s", settings.mlflow_uri)

    try:
        runs = mlflow.search_runs(search_all_experiments=True)
        log.info("logged runs: %d", len(runs))
    except Exception as exc:
        log.warning("could not list runs: %s", exc)

    try:
        versions = client.search_model_versions(f"name='{settings.mlflow_model_name}'")
        log.info("registered versions of %r: %d", settings.mlflow_model_name, len(versions))
        for v in versions[:8]:
            log.info("   version %s  run=%s", v.version, v.run_id[:12])
    except Exception as exc:
        log.warning("no registered model %r (%s)", settings.mlflow_model_name, type(exc).__name__)

    for alias in (CHAMPION, CHALLENGER):
        ref = get_by_alias(alias, settings)
        log.info("alias %-11s -> %s", alias, "NOT SET" if ref is None else f"version {ref.version}")


if __name__ == "__main__":
    main()
