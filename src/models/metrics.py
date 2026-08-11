"""Evaluation metrics, written around the business decision rather than around
the loss function.

The store can only afford to act on a slice of traffic, so the number that
matters is: if we intervene on the top X% of sessions by score, what share of
the actual buyers did we capture, and how much better is that than picking at
random? That is `recall_at_k` and `lift_at_k`.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

DECILES = (0.01, 0.05, 0.10, 0.20, 0.50)


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float) -> float:
    """Share of all positives captured in the top-k fraction by score."""
    n = len(y_score)
    if n == 0:
        return float("nan")
    total_pos = float(y_true.sum())
    if total_pos == 0:
        return float("nan")
    cut = max(1, int(round(n * k)))
    idx = np.argsort(-y_score, kind="stable")[:cut]
    return float(y_true[idx].sum() / total_pos)


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float) -> float:
    """How many times better than random targeting at the same budget."""
    r = recall_at_k(y_true, y_score, k)
    return float(r / k) if k > 0 else float("nan")


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    out: dict[str, float] = {
        "n": float(len(y_true)),
        "base_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "brier": float("nan"),
    }
    # A single-class window (can happen in short drift slices) has no AUC.
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
        out["brier"] = float(brier_score_loss(y_true, np.clip(y_score, 0, 1)))

    for k in DECILES:
        pct = int(k * 100)
        out[f"recall_at_{pct}pct"] = recall_at_k(y_true, y_score, k)
        out[f"lift_at_{pct}pct"] = lift_at_k(y_true, y_score, k)
    return out


def format_report(name: str, m: dict[str, float]) -> str:
    return (
        f"{name:<22} n={int(m['n']):>9,}  base={m['base_rate']*100:5.2f}%  "
        f"AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
        f"recall@10%={m['recall_at_10pct']*100:5.1f}%  lift@10%={m['lift_at_10pct']:.2f}x"
    )
