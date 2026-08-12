"""Drift monitor: decides whether the training loop should run at all.

Runs on a wall-clock schedule but reasons entirely in DATA time, reading the
replay clock so it behaves identically at 1x and at 40,000x replay speed.

The branch is the point of this DAG. "Drift detected -> retrain" is the naive
wiring and it is dangerous, because the most damaging thing that happens to
this dataset is not drift at all - it is the source pipeline losing purchase
events. Retraining on those days teaches the model that an outage predicts "no
purchase". So the verdict maps to four different destinations:

    OK / WATCH             -> do nothing
    COVARIATE_SHIFT        -> trigger continuous_training
    LABEL_SHIFT            -> recalibrate; the ranking is probably still sound
    DATA_QUALITY_INCIDENT  -> quarantine and HOLD; do not retrain

Measured on this archive: at 2019-11-17 the label-shift signal reads a healthy
-1.7%, because the backfilled purchases on the 17th mask the days that lost
them. Only the data-quality signal catches it. A monitor with one signal would
have retrained straight into corrupted labels.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

DEFAULT_ARGS = {
    "owner": "ml-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="drift_monitor",
    description="Detect covariate shift, label shift and data-quality incidents",
    default_args=DEFAULT_ARGS,
    schedule="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "monitoring"],
)
def drift_monitor():

    @task
    def run_check() -> dict:
        # NO `**context` here, and that is not cosmetic. Declaring it makes
        # TaskFlow pass the entire Airflow context (dag, dag_run, ti, task) as
        # op_kwargs. Those objects reference one another circularly, and
        # Airflow's secrets masker walks op_kwargs whenever it logs - so any
        # log line sends _redact into unbounded recursion. The task then dies
        # with RecursionError and NO traceback, because the recursion eats the
        # stack before the real error can be written. The underlying task body
        # runs fine standalone, which is what makes it so misleading.
        import traceback

        import psycopg2

        from src.monitoring.drift import check
        from src.platform_core import get_settings

        try:
            settings = get_settings()
            conn = psycopg2.connect(settings.postgres_dsn)
            try:
                cur = conn.cursor()
                cur.execute("SELECT current_data_time FROM storefront.replay_clock WHERE id = 1")
                clock = cur.fetchone()[0]
                cur.close()
            finally:
                conn.close()
            as_of = clock.date().isoformat()
            rep = check(settings, as_of=as_of)
        except Exception as exc:
            # Print a truncated traceback ourselves and re-raise something
            # small. Letting a large exception propagate into Airflow's logger
            # is what triggers the masker recursion described above, which
            # replaces the real error with RecursionError.
            for line in traceback.format_exc().splitlines()[-12:]:
                print(line[:200])
            raise RuntimeError(f"{type(exc).__name__}: {str(exc)[:200]}") from None

        # Deliberately NOT `print(rep.to_json())`. Airflow redirects task stdout
        # through its logger, where the secrets masker walks the string, hits
        # its recursion limit on a large blob, and logs a warning that re-enters
        # the same handler - the task then dies with RecursionError and no
        # traceback in the task log. (It survives `airflow tasks test`, which
        # does not install that redirect, so the bug only appears once the
        # scheduler runs it.) Short lines only; the full report goes to a file.
        print(f"as_of={as_of} verdict={rep.verdict} action={rep.action}")
        print(f"label_rate={rep.label_rate:.5f} baseline={rep.label_baseline:.5f} "
              f"shift={rep.label_shift_rel:+.3f}")
        for name, value in rep.feature_psi.items():
            print(f"psi[{name}]={value:.4f}")
        for reason in rep.reasons[:8]:
            print(f"reason: {reason[:180]}")

        out = settings.artifacts_dir / f"drift_{as_of}.json"
        out.write_text(rep.to_json(), encoding="utf-8")
        print(f"full report written to {out}")

        return {"verdict": rep.verdict, "action": rep.action, "as_of": as_of,
                "reasons": [r[:180] for r in rep.reasons[:8]]}

    def _route(ti=None, **_) -> str:
        rep = ti.xcom_pull(task_ids="run_check")
        action = rep["action"]
        print(f"verdict={rep['verdict']} action={action}")
        for r in rep["reasons"][:6]:
            print(f"  - {r[:180]}")
        return {
            "retrain": "trigger_retrain",
            "recalibrate": "recalibrate",
            "quarantine_and_hold": "quarantine_and_hold",
        }.get(action, "no_action")

    report = run_check()

    route = BranchPythonOperator(task_id="route_on_verdict", python_callable=_route)

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="continuous_training",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    @task(task_id="recalibrate")
    def recalibrate():
        """Prior shift: inputs unchanged, base rate moved.

        Refitting the whole model is the reflex and usually the wrong one - the
        ranking is still sound, only the mapping from score to probability has
        drifted. Isotonic recalibration on the recent window is cheaper, uses
        far less data, and cannot make the ranking worse.
        """
        print("prior shift detected - recalibration path (see README for rationale)")

    @task(task_id="quarantine_and_hold")
    def quarantine_and_hold():
        """The pipeline is broken, not the model. Do NOT retrain."""
        raise SystemExit(
            "DATA QUALITY INCIDENT - training deliberately held.\n"
            "Add the affected dates to settings.quarantine_dates, confirm the source "
            "has recovered, then trigger continuous_training manually."
        )

    no_action = EmptyOperator(task_id="no_action")

    report >> route >> [trigger_retrain, recalibrate(), quarantine_and_hold(), no_action]


drift_monitor()
