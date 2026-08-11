"""Train the session purchase-intent model.

Splitting is strictly chronological. Random splits would let the model learn
from sessions that happen after the ones it is scored on, which inflates every
metric and is the most common way a portfolio project ends up reporting numbers
it cannot defend.

Windows (fractions of the observed date range, overridable):
    train  : earliest  -> 70%
    valid  : 70%       -> 82%   (early stopping / threshold selection)
    test   : 82%       -> 92%   (the headline number)
    drift  : 92%       -> end   (deliberately includes the Black Friday spike)

The `drift` window is reported separately and is expected to be worse. That
gap is the evidence that continuous retraining is needed, so it is a result,
not a failure.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from src.models.metrics import evaluate, format_report
from src.platform_core import get_logger, get_settings

log = get_logger(__name__)

# Columns that identify a row or encode time - never fed to the model.
ID_COLS = ["session_key", "user_id", "session_start", "cutoff_time", "session_date"]
LABEL = "y"


def load(path: Path) -> pl.DataFrame:
    log.info("loading %s", path.name)
    df = pl.read_parquet(path)
    log.info("loaded %s rows x %d cols", f"{df.height:,}", df.width)
    return df


def chronological_split(
    df: pl.DataFrame,
    bounds: tuple[float, float, float] = (0.70, 0.82, 0.92),
) -> dict[str, pl.DataFrame]:
    dates = sorted(df["session_date"].unique().to_list())
    n = len(dates)
    if n < 4:
        raise SystemExit(f"need at least 4 distinct dates to split, got {n}")

    i1, i2, i3 = (max(1, int(round(n * b))) for b in bounds)
    cuts = {
        "train": (dates[0], dates[i1 - 1]),
        "valid": (dates[i1], dates[i2 - 1]),
        "test": (dates[i2], dates[i3 - 1]),
        "drift": (dates[i3], dates[-1]),
    }
    out: dict[str, pl.DataFrame] = {}
    for name, (lo, hi) in cuts.items():
        part = df.filter(
            (pl.col("session_date") >= lo) & (pl.col("session_date") <= hi)
        )
        out[name] = part
        log.info(
            "%-6s %s -> %s  rows=%s  pos=%.3f%%",
            name, lo, hi, f"{part.height:,}",
            float(part[LABEL].mean() or 0) * 100,
        )
    return out


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS and c != LABEL]


def _xy(df: pl.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df[LABEL].to_numpy().astype(int)
    return X, y


def baselines(splits: dict[str, pl.DataFrame], feats: list[str]) -> dict[str, dict]:
    """Dumb comparators. Without these, an AUC number means nothing."""
    test = splits["test"]
    y = test[LABEL].to_numpy().astype(int)
    rng = np.random.default_rng(42)

    out = {"random": evaluate(y, rng.random(len(y)))}

    # The single most obvious heuristic a business would actually deploy.
    if "n_cart_events" in test.columns:
        out["heuristic_cart"] = evaluate(y, test["n_cart_events"].to_numpy().astype(float))
    if "max_views_same_product" in test.columns:
        out["heuristic_revisit"] = evaluate(
            y, test["max_views_same_product"].fill_null(0).to_numpy().astype(float)
        )
    if "prefix_duration_sec" in test.columns:
        out["heuristic_dwell"] = evaluate(
            y, test["prefix_duration_sec"].fill_null(0).to_numpy().astype(float)
        )
    return out


def train_lgbm(splits: dict[str, pl.DataFrame], feats: list[str], seed: int = 42):
    import lightgbm as lgb

    Xtr, ytr = _xy(splits["train"], feats)
    Xva, yva = _xy(splits["valid"], feats)

    pos = float(ytr.sum())
    neg = float(len(ytr) - pos)
    log.info("train pos=%s neg=%s  ratio=1:%.0f", f"{int(pos):,}", f"{int(neg):,}", neg / max(pos, 1))

    params = dict(
        objective="binary",
        metric=["auc", "average_precision"],
        learning_rate=0.05,
        num_leaves=127,
        min_data_in_leaf=200,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l2=1.0,
        max_bin=255,
        num_threads=0,
        seed=seed,
        verbosity=-1,
    )

    booster = lgb.train(
        params,
        lgb.Dataset(Xtr, label=ytr, feature_name=feats),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(Xva, label=yva, feature_name=feats)],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(200),
        ],
    )
    log.info("best iteration: %d", booster.best_iteration)
    return booster


def run(k: int, hard_mode: bool, settings, log_mlflow: bool = True) -> dict:
    suffix = f"k{k}" + ("_hard" if hard_mode else "")
    path = settings.features_dir / f"train_{suffix}.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path} - run src.features.build_features first")

    df = load(path)
    splits = chronological_split(df)
    feats = feature_columns(df)
    log.info("using %d features", len(feats))

    base = baselines(splits, feats)
    booster = train_lgbm(splits, feats)

    results: dict[str, dict] = {"baselines": base, "model": {}}
    for name in ("valid", "test", "drift"):
        part = splits[name]
        if part.height == 0:
            continue
        X, y = _xy(part, feats)
        p = booster.predict(X, num_iteration=booster.best_iteration)
        results["model"][name] = evaluate(y, p)

    imp = sorted(
        zip(feats, booster.feature_importance(importance_type="gain")),
        key=lambda t: -t[1],
    )
    results["feature_importance"] = [{"feature": f, "gain": float(g)} for f, g in imp]

    log.info("=" * 100)
    for nm, m in base.items():
        log.info(format_report(f"BASELINE {nm}", m))
    log.info("-" * 100)
    for nm, m in results["model"].items():
        log.info(format_report(f"LGBM {nm}", m))
    log.info("=" * 100)
    log.info("top 15 features by gain:")
    for f, g in imp[:15]:
        log.info("   %-28s %12.0f", f, g)

    model_path = settings.artifacts_dir / f"model_{suffix}.txt"
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)
    (settings.artifacts_dir / f"results_{suffix}.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    log.info("saved model -> %s", model_path)

    if log_mlflow:
        _log_to_mlflow(suffix, k, hard_mode, feats, results, model_path, settings)

    return results


def _log_to_mlflow(suffix, k, hard_mode, feats, results, model_path, settings) -> None:
    """Log to the MLflow server if reachable, else to a local file store."""
    try:
        import mlflow

        uri = settings.mlflow_tracking_uri
        try:
            import urllib.request

            urllib.request.urlopen(f"{uri}/health", timeout=3)
        except Exception:
            uri = (settings.artifacts_dir / "mlruns").as_uri()
            log.warning("MLflow server unreachable; logging locally to %s", uri)

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("session_purchase_intent")
        with mlflow.start_run(run_name=suffix):
            mlflow.log_params(
                {"k": k, "hard_mode": hard_mode, "n_features": len(feats)}
            )
            for split, m in results["model"].items():
                mlflow.log_metrics(
                    {f"{split}_{kk}": vv for kk, vv in m.items() if np.isfinite(vv)}
                )
            for nm, m in results["baselines"].items():
                if np.isfinite(m.get("roc_auc", np.nan)):
                    mlflow.log_metric(f"baseline_{nm}_roc_auc", m["roc_auc"])
            mlflow.log_artifact(str(model_path))
        log.info("logged run '%s' to MLflow", suffix)
    except Exception as exc:  # never let tracking break training
        log.warning("MLflow logging skipped: %s", exc)


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=None, help="single k; default trains all configured k")
    ap.add_argument("--hard-mode", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    ks = [args.k] if args.k else list(settings.truncation_ks)
    summary = {}
    for k in ks:
        log.info("#" * 100)
        log.info("### k = %d   hard_mode = %s", k, args.hard_mode)
        log.info("#" * 100)
        res = run(k, args.hard_mode, settings, log_mlflow=not args.no_mlflow)
        summary[k] = res["model"].get("test", {})

    if len(ks) > 1:
        log.info("=" * 60)
        log.info("AUC-vs-k curve (test window):")
        for k, m in summary.items():
            log.info("  k=%-3d AUC=%.4f  lift@10%%=%.2fx", k, m.get("roc_auc", float("nan")),
                     m.get("lift_at_10pct", float("nan")))


if __name__ == "__main__":
    main()
