from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd


def _safe_float(x):
    try:
        if x is None:
            return 0.0
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        return float(x)
    except Exception:
        return 0.0


def compute_wfe(
    fold_metrics: pd.DataFrame,
    full_sharpe: float,
) -> float:
    """
    Walk Forward Efficiency (proxy):
    full sharpe / average fold sharpe
    """
    if fold_metrics.empty:
        return 0.0

    avg_fold_sharpe = fold_metrics["sharpe"].mean()

    if avg_fold_sharpe == 0:
        return 0.0

    return _safe_float(full_sharpe / avg_fold_sharpe)


def build_institutional_report(
    metrics: Dict[str, float],
    monte_carlo: Dict[str, Any],
    walk_forward: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build final institutional report.
    """

    aggregate = walk_forward.get("aggregate_metrics", {})
    degradation = walk_forward.get("degradation_analysis", {})
    fold_metrics = walk_forward.get("fold_metrics", pd.DataFrame())

    sharpe = _safe_float(metrics.get("sharpe", 0))
    sortino = _safe_float(metrics.get("sortino", 0))
    calmar = _safe_float(metrics.get("calmar", 0))
    total_return = _safe_float(metrics.get("total_return", 0))
    max_dd = _safe_float(metrics.get("max_drawdown", 0))

    mc_prob = _safe_float(monte_carlo.get("probability_positive", 0))
    mc_sharpe = _safe_float(monte_carlo.get("sharpe_mean", 0))
    mc_score = _safe_float(monte_carlo.get("robustness_score", 0))

    wfe = compute_wfe(
        fold_metrics=fold_metrics,
        full_sharpe=_safe_float(aggregate.get("full_period_sharpe", sharpe)),
    )

    # score institucional (heurístico)
    score = (
        0.30 * max(sharpe, 0)
        + 0.20 * max(sortino, 0)
        + 0.15 * max(calmar, 0)
        + 0.15 * mc_prob
        + 0.10 * mc_score
        + 0.10 * max(wfe, 0)
    )

    report = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "version": "1.0_institutional",
        },
        "performance": {
            "total_return": total_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
        },
        "risk": {
            "max_drawdown": max_dd,
            "calmar": calmar,
        },
        "monte_carlo": {
            "probability_positive": mc_prob,
            "sharpe_mean": mc_sharpe,
            "robustness_score": mc_score,
            "confidence_interval": monte_carlo.get("confidence_interval_5_95"),
        },
        "walk_forward": {
            "folds": int(len(fold_metrics)) if isinstance(fold_metrics, pd.DataFrame) else 0,
            "mean_sharpe": _safe_float(aggregate.get("mean_fold_sharpe", 0)),
            "stability": _safe_float(degradation.get("sharpe_stability", 0)),
            "wfe": wfe,
        },
        "scores": {
            "institutional_score": score,
        },
    }

    return report


def save_report(
    report: Dict[str, Any],
    path: str = "institutional_report.json",
) -> None:
    """
    Save report to JSON file.
    """
    with open(path, "w") as f:
        json.dump(report, f, indent=4)
