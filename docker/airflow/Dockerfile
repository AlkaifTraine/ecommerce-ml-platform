# Airflow image with this project's runtime dependencies.
#
# The stock apache/airflow image has pandas and numpy but not duckdb, polars,
# lightgbm or scikit-learn, so the pipeline cannot run in it as shipped.
#
# Base tag pinned to 2.10.0 because that image is ALREADY present on this
# machine. Any other tag is a ~2GB pull over a ~100 KB/s link.
#
# ONE PACKAGE PER LAYER, deliberately. This link times out partway through
# large wheels and pip cannot resume, so a single combined RUN meant a failure
# on the 57MB polars wheel threw away the 21.5MB duckdb wheel that had already
# downloaded. With separate layers Docker caches each success and a rebuild
# resumes at the first failure. The long --timeout and --retries matter for the
# same reason.

FROM apache/airflow:2.10.0

USER airflow

ARG PIP_OPTS="--no-cache-dir --timeout 300 --retries 20"

# Heaviest first, each isolated so its layer is cached on success.
RUN pip install ${PIP_OPTS} "duckdb>=1.1"
RUN pip install ${PIP_OPTS} "polars>=1.17"
RUN pip install ${PIP_OPTS} "pyarrow>=18"

# numpy MUST stay below 2.0 in this image. apache/airflow:2.10.0 ships
# pandas 2.1.4, which declares numpy<2 and fails at import against numpy 2.x
# with a binary incompatibility.
#
# Pinning numpy in an earlier layer is NOT enough, and that failed here twice.
# pip resolves each RUN independently, and scikit-learn pulls scipy, whose
# current release requires numpy>=2 - so numpy came back through a transitive
# dependency even with an explicit "numpy<2" installed first. scipy has to be
# capped as well, and the three installed together so the resolver sees all
# the constraints at once.
RUN pip install ${PIP_OPTS} \
        "numpy<2" \
        "scipy<1.14" \
        "scikit-learn>=1.5,<1.6" \
        "lightgbm>=4.5,<4.7"
RUN pip install ${PIP_OPTS} "mlflow-skinny>=2.19"
RUN pip install ${PIP_OPTS} "redis>=5.0" "pydantic-settings>=2.7"
RUN pip install ${PIP_OPTS} "dbt-core>=1.9" "dbt-duckdb>=1.9"

# Build-time smoke test - see smoke_test.py for why this is a build layer and
# not something checked at runtime.
COPY smoke_test.py /tmp/smoke_test.py
RUN python /tmp/smoke_test.py

# Code is bind-mounted at /opt/project rather than copied, so a code change
# does not require an image rebuild.
ENV PYTHONPATH=/opt/project
WORKDIR /opt/project
