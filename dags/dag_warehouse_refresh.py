"""Warehouse refresh: drain the OLTP hot window, rebuild marts, run dbt tests.

Ordering matters and is not arbitrary:

    cdc_extract -> build_warehouse -> dbt build (models + tests)

`cdc_extract` must run before the OLTP retention job trims its rolling 7-day
window, or history is lost permanently - Postgres is not the system of record
for anything older than a week. That is the whole reason this DAG exists on a
schedule rather than being run by hand.

dbt runs LAST and its tests are not advisory. They caught a real defect on
their first execution against this data: 12 events carrying a NULL
`user_session`, which `GROUP BY` had merged into a single phantom session
containing twelve unrelated users.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task

DEFAULT_ARGS = {
    "owner": "data-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}


@dag(
    dag_id="warehouse_refresh",
    description="CDC drain -> star schema rebuild -> dbt models and tests",
    default_args=DEFAULT_ARGS,
    schedule="0 */6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["data", "warehouse"],
)
def warehouse_refresh():

    @task
    def cdc_extract() -> dict:
        """Drain OLTP into the lake before retention deletes it."""
        from src.platform_core import get_settings
        from src.warehouse.cdc_extract import SOURCES, extract

        settings = get_settings()
        moved = {}
        for table in SOURCES:
            total = 0
            # loop until drained; each pass respects the safety lag
            while True:
                n = extract(settings, table, batch_size=500_000, lag_seconds=300)
                total += n
                if n == 0:
                    break
            moved[table] = total
        print(f"extracted: {moved}")
        return moved

    @task
    def build_warehouse(_extracted: dict) -> None:
        from src.platform_core import get_settings
        from src.warehouse.build_warehouse import build

        build(get_settings())

    @task
    def dbt_build() -> str:
        """Run `dbt build` if dbt is present; skip loudly if it is not.

        dbt is deliberately absent from this image. Two independent reasons:

        1. dbt-core requires protobuf>=5 while the opentelemetry-proto that
           Airflow 2.10 ships requires protobuf<5. No pin satisfies both;
           installing dbt anyway broke OpenTelemetry and ~20 google-cloud
           packages.
        2. DuckDB permits a single writing process. If this container held
           warehouse.duckdb open read-write, nothing on the host could write
           to it, and vice versa.

        The production answer to both is the same and is what a cloud deploy
        would do: run dbt in its own image via KubernetesPodOperator or
        ECSOperator. Locally it runs on the host. This task therefore SKIPS
        rather than fails, so a green DAG never implies dbt ran when it did
        not.
        """
        import shutil
        import subprocess

        from airflow.exceptions import AirflowSkipException

        if shutil.which("dbt") is None:
            raise AirflowSkipException(
                "dbt is not installed in this image by design (protobuf conflict "
                "with Airflow's opentelemetry, and DuckDB's single-writer limit). "
                "Run it on the host:  cd dbt && dbt build --profiles-dir ."
            )

        proc = subprocess.run(
            ["dbt", "build", "--profiles-dir", ".", "--no-partial-parse"],
            cwd="/opt/project/dbt", capture_output=True, text=True,
        )
        print(proc.stdout[-4000:])
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
            raise RuntimeError(f"dbt build failed with exit {proc.returncode}")
        return "dbt build passed"

    build_warehouse(cdc_extract()) >> dbt_build()


warehouse_refresh()
