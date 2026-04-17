from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


@dataclass
class MLEngineConfig:
    horizon: int = 1
    volatility_window: int = 20
    trend_window: int = 20
    ma_fast: int = 10
    ma_slow: int = 30
    min_history: int = 40
    random_state: int = 42
    rf_n_estimators: int = 200
    rf_max_depth: int = 5
    gb_n_estimators: int = 150
    gb_learning_rate: float = 0.05
    gb_max_depth: int = 2


class EnsembleProbabilityModel:
    """
    Ensemble classifier:
    - RandomForestClassifier
    - GradientBoostingClassifier

    Output:
    - probability of positive future return
    """

    def __init__(self, config: MLEngineConfig | None = None):
        self.config = config or MLEngineConfig()

        self.rf = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=10,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

        self.gb = GradientBoostingClassifier(
            n_estimators=self.config.gb_n_estimators,
            learning_rate=self.config.gb_learning_rate,
            max_depth=self.config.gb_max_depth,
            random_state=self.config.random_state,
        )

        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleProbabilityModel":
        X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = pd.Series(y).astype(int)

        self.feature_names_ = list(X.columns)

        self.rf.fit(X, y)
        self.gb.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        rf_proba = self.rf.predict_proba(X)[:, 1]
        gb_proba = self.gb.predict_proba(X)[:, 1]

        ensemble_proba = 0.5 * rf_proba + 0.5 * gb_proba
        return np.column_stack([1.0 - ensemble_proba, ensemble_proba])


def _safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


def _rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std()


def _trend_strength(price: pd.Series, window: int) -> pd.Series:
    ma = price.rolling(window).mean()
    return (price / ma) - 1.0


def _moving_average_ratio(price: pd.Series, fast: int, slow: int) -> Tuple[pd.Series, pd.Series]:
    ma_fast = price.rolling(fast).mean()
    ma_slow = price.rolling(slow).mean()

    fast_ratio = (price / ma_fast) - 1.0
    slow_ratio = (price / ma_slow) - 1.0
    return fast_ratio, slow_ratio


def build_feature_panel(
    prices: pd.DataFrame,
    config: MLEngineConfig | None = None,
) -> pd.DataFrame:
    """
    Build long-form feature panel:
    columns:
        date, asset, <features>, target
    """
    if config is None:
        config = MLEngineConfig()

    if prices.empty:
        raise ValueError("prices is empty")

    prices = prices.sort_index().copy()
    prices.index = pd.to_datetime(prices.index)

    all_frames = []

    for asset in prices.columns:
        px = pd.to_numeric(prices[asset], errors="coerce").dropna().copy()

        if len(px) < config.min_history:
            continue

        ret_1 = _safe_pct_change(px, 1)
        ret_5 = _safe_pct_change(px, 5)
        ret_10 = _safe_pct_change(px, 10)

        volatility = _rolling_volatility(ret_1, config.volatility_window)
        trend_strength = _trend_strength(px, config.trend_window)

        ma_fast_ratio, ma_slow_ratio = _moving_average_ratio(
            px,
            fast=config.ma_fast,
            slow=config.ma_slow,
        )

        forward_return = px.shift(-config.horizon) / px - 1.0
        target = (forward_return > 0).astype(int)

        df_asset = pd.DataFrame(
            {
                "date": px.index,
                "asset": asset,
                "ret_1": ret_1.values,
                "ret_5": ret_5.values,
                "ret_10": ret_10.values,
                "volatility": volatility.values,
                "trend_strength": trend_strength.values,
                "ma_fast_ratio": ma_fast_ratio.values,
                "ma_slow_ratio": ma_slow_ratio.values,
                "target": target.values,
                "forward_return": forward_return.values,
            }
        )

        all_frames.append(df_asset)

    if not all_frames:
        raise ValueError("No valid assets with enough history to build features")

    feature_panel = pd.concat(all_frames, ignore_index=True)
    feature_panel["date"] = pd.to_datetime(feature_panel["date"])
    feature_panel = feature_panel.sort_values(["date", "asset"]).reset_index(drop=True)

    feature_panel = feature_panel.replace([np.inf, -np.inf], np.nan)

    required_feature_cols = [
        "ret_1",
        "ret_5",
        "ret_10",
        "volatility",
        "trend_strength",
        "ma_fast_ratio",
        "ma_slow_ratio",
    ]

    feature_panel = feature_panel.dropna(subset=required_feature_cols + ["target"])
    feature_panel["target"] = feature_panel["target"].astype(int)

    return feature_panel


def get_feature_columns(feature_panel: pd.DataFrame) -> List[str]:
    excluded = {"date", "asset", "target", "forward_return"}
    return [c for c in feature_panel.columns if c not in excluded]


def build_prediction_frame(
    feature_panel: pd.DataFrame,
    model: EnsembleProbabilityModel,
) -> pd.DataFrame:
    """
    Utility to score a full feature panel after model.fit.
    Returns:
        date, asset, probability
    """
    feature_cols = get_feature_columns(feature_panel)

    X = feature_panel[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    proba = model.predict_proba(X)[:, 1]

    out = feature_panel[["date", "asset"]].copy()
    out["probability"] = proba
    return out


def build_train_test_matrices(
    feature_panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience helper for manual training outside walk-forward.
    """
    feature_cols = get_feature_columns(feature_panel)
    X = feature_panel[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = feature_panel["target"].astype(int)
    return X, y
