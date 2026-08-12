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
from airflow.operators.bash import BashOperator

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

    dbt_build = BashOperator(
        task_id="dbt_build",
        # `dbt build` runs models and their tests together and stops a
        # downstream model from being built on a failed upstream test.
        bash_command=(
            "cd /opt/project/dbt && "
            "dbt build --profiles-dir . --no-partial-parse"
        ),
    )

    build_warehouse(cdc_extract()) >> dbt_build


warehouse_refresh()
