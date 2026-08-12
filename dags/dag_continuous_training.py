"""Continuous training loop.

    read clock -> build features -> LEAKAGE AUDIT -> train -> register -> promote

Two things make this more than a chain of scripts:

1. **The leakage audit is a hard gate.** It runs BEFORE training and the DAG
   fails there if it trips. A pipeline that trains first and checks later will
   happily register a leaking model and only tell you afterwards. The audit
   already caught a real non-determinism bug in this project - one violation in
   5,153,372 sessions - so it is not decoration.

2. **Everything is bounded by the replay clock, not wall time.** Each run reads
   `storefront.replay_clock` and passes that instant to the feature build as
   `--until`. Tasks therefore cannot see data that "has not happened yet",
   regardless of what is sitting in the archive on disk.

Promotion is never automatic on completion: a new model is registered as
`challenger` and only takes the `champion` alias if it beats the incumbent by
more than the configured margin.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

DEFAULT_ARGS = {
    "owner": "ml-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}

K = 5


@dag(
    dag_id="continuous_training",
    description="Feature build -> leakage audit -> train -> registry promotion",
    default_args=DEFAULT_ARGS,
    schedule=None,  # triggered by drift_monitor, or manually
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,  # the feature build needs the whole memory budget
    tags=["ml", "training"],
)
def continuous_training():

    @task
    def read_clock() -> str:
        """The data instant this run is allowed to see."""
        import psycopg2

        from src.platform_core import get_settings

        settings = get_settings()
        conn = psycopg2.connect(settings.postgres_dsn)
        try:
            cur = conn.cursor()
            cur.execute("SELECT current_data_time FROM storefront.replay_clock WHERE id = 1")
            ts = cur.fetchone()[0]
            cur.close()
        finally:
            conn.close()
        print(f"replay clock at {ts}; nothing at or after this is visible")
        return ts.isoformat()

    @task
    def build_features(until: str) -> int:
        from src.features import build_features as bf
        from src.platform_core import get_settings

        n = bf.build(get_settings(), k=K, until=until)
        if n == 0:
            raise AirflowFailException("feature build produced no rows")
        return n

    @task
    def leakage_audit(until: str, n_rows: int) -> dict:
        """HARD GATE. Nothing downstream runs if this fails."""
        from src.features.leakage_audit import audit
        from src.platform_core import get_settings

        res = audit(get_settings(), k=K, until=until)
        print(res.report())
        if not res.passed:
            raise AirflowFailException(
                f"LEAKAGE AUDIT FAILED on {n_rows} rows - refusing to train.\n{res.report()}"
            )
        return res.checks

    @task
    def train(_gate: dict) -> dict:
        from src.models.train import run
        from src.platform_core import get_settings

        results = run(K, hard_mode=False, settings=get_settings(), log_mlflow=False)
        test = results["model"].get("test", {})
        print(f"test AUC={test.get('roc_auc'):.4f} lift@10%={test.get('lift_at_10pct'):.2f}x")
        return {f"test_{k}": v for k, v in test.items()}

    @task
    def register(metrics: dict) -> str | None:
        from src.models.registry import log_run
        from src.platform_core import get_settings

        settings = get_settings()
        model_path = settings.artifacts_dir / f"model_k{K}.txt"
        version = log_run(
            run_name=f"airflow_k{K}_{datetime.utcnow():%Y%m%d_%H%M%S}",
            params={"k": K, "hard_mode": False, "trigger": "airflow"},
            metrics={"test": {k.replace("test_", ""): v for k, v in metrics.items()}},
            artifacts=[model_path],
            settings=settings,
        )
        if version is None:
            raise AirflowFailException("model registration returned no version")
        return version

    @task
    def promote(version: str, metrics: dict) -> bool:
        """Challenger first; champion only on evidence."""
        from src.models.registry import promote_if_better

        promoted = promote_if_better(version, metrics)
        print("PROMOTED to champion" if promoted else "held as challenger - did not clear margin")
        return promoted

    until = read_clock()
    rows = build_features(until)
    gate = leakage_audit(until, rows)
    metrics = train(gate)
    version = register(metrics)
    promote(version, metrics)


continuous_training()
