# Airflow image with this project's runtime dependencies.
#
# The stock apache/airflow image cannot run our pipeline - it has pandas and
# numpy but not duckdb, polars, lightgbm or scikit-learn. Rather than shell out
# to the host (impossible from a container) the tasks run in-process here, so
# the image needs the same libraries the pipeline uses.
#
# Base tag is pinned to 2.10.0 because that image is ALREADY present on this
# machine. Pulling a different tag would mean ~2GB over a ~100 KB/s link.
#
# The extra wheels are ~150MB and are the only download this build needs.

FROM apache/airflow:2.10.0

USER airflow

# Airflow pins its own dependency tree tightly; installing without care breaks
# it. None of these are Airflow dependencies, so they can be added directly -
# but they are installed in one layer so a retry does not redo the whole set.
RUN pip install --no-cache-dir \
        "duckdb>=1.1" \
        "polars>=1.17" \
        "lightgbm>=4.5" \
        "scikit-learn>=1.5" \
        "pyarrow>=18" \
        "redis>=5.0" \
        "mlflow-skinny>=2.19" \
        "pydantic-settings>=2.7"

# Our code is bind-mounted at /opt/project rather than copied, so a code change
# does not require an image rebuild.
ENV PYTHONPATH=/opt/project
WORKDIR /opt/project
