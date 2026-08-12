"""Central configuration.

Every path in the project resolves through here so that nothing accidentally
writes to C:, which has very little free space on this machine.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- storage ----
    data_root: Path = Field(default=PROJECT_ROOT / "data")
    duckdb_temp_dir: Path = Field(default=PROJECT_ROOT / "data" / ".duckdb_tmp")
    duckdb_memory_limit: str = "8GB"

    # ---- source dataset ----
    kaggle_dataset: str = "mkechinov/ecommerce-behavior-data-from-multi-category-store"

    # ---- OLTP ----
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "storefront"
    postgres_user: str = "storefront"
    postgres_password: str = "change_me_locally"

    # ---- online store ----
    redis_url: str = "redis://localhost:6379/0"

    # ---- tracking ----
    # Empty means "derive a SQLite path under data_root" (see mlflow_uri). The
    # registry needs a database-backed store - a plain file:// store supports
    # tracking but NOT model versions or aliases, which the champion/challenger
    # promotion flow depends on. SQLite gives us that with no server to run and
    # no container image to pull.
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "session_purchase_intent"
    mlflow_model_name: str = "purchase_intent"

    # ---- replayer ----
    replay_speed: float = 100_000.0
    replay_backfill_until: datetime = datetime(2019, 11, 20)

    # ---- modelling constants ----
    # A session is "truncated" at k events; we predict the outcome from only
    # those first k. Multiple k values let us plot the AUC-vs-k curve.
    truncation_ks: tuple[int, ...] = (5, 10, 20)
    # Inactivity gap used when we re-derive sessions instead of trusting
    # the provider's user_session column.
    session_gap_minutes: int = 30

    # Days excluded from training and evaluation because the source pipeline
    # failed, not because shoppers behaved differently. Surfaced by
    # scripts/data_quality_audit.py and confirmed by review:
    #   2019-11-15  zero purchase events recorded all day
    #   2019-11-16  4.1x normal volume, buy/cart half the local norm
    #   2019-11-17  4.2x normal volume, buy/cart inflated by the backlog flush
    #   2019-11-14  2.0x volume and buy/cart 13% vs ~33% either side - the
    #               leading edge of the same incident
    # Sessions with ANY event on these days are dropped: a session straddling
    # the outage has an unknowable label.
    quarantine_dates: tuple[str, ...] = (
        "2019-11-14", "2019-11-15", "2019-11-16", "2019-11-17",
    )

    # ---- derived paths ----
    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def parquet_dir(self) -> Path:
        return self.data_root / "parquet"

    @property
    def features_dir(self) -> Path:
        return self.data_root / "features"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_root / "artifacts"

    @property
    def events_dir(self) -> Path:
        """Partitioned event-level parquet - the immutable source of truth."""
        return self.parquet_dir / "events"

    @property
    def mlflow_uri(self) -> str:
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        db = (self.data_root / "mlflow.db").as_posix()
        return f"sqlite:///{db}"

    @property
    def mlflow_artifacts(self) -> str:
        return (self.data_root / "mlartifacts").as_uri()

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def ensure_dirs(self) -> None:
        for p in (
            self.data_root,
            self.raw_dir,
            self.parquet_dir,
            self.features_dir,
            self.artifacts_dir,
            self.duckdb_temp_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
