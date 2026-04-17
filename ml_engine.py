from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


@dataclass
class MLEngineConfig:
    horizon: int = 1
    random_state: int = 42


class EnsembleProbabilityModel:
    def __init__(self, config: MLEngineConfig | None = None):
        self.config = config or MLEngineConfig()

        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

        self.gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
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

        p1 = self.rf.predict_proba(X)[:, 1]
        p2 = self.gb.predict_proba(X)[:, 1]
        p = 0.5 * p1 + 0.5 * p2
        return np.column_stack([1.0 - p, p])


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-9)


def build_feature_panel(prices: pd.DataFrame, config: MLEngineConfig | None = None) -> pd.DataFrame:
    if config is None:
        config = MLEngineConfig()

    if prices.empty:
        raise ValueError("prices is empty")

    prices = prices.sort_index().copy()
    prices.index = pd.to_datetime(prices.index)

    all_data = []

    for asset in prices.columns:
        px = pd.to_numeric(prices[asset], errors="coerce").dropna()

        if len(px) < 120:
            continue

        df = pd.DataFrame(index=px.index)
        df["date"] = px.index
        df["asset"] = asset

        df["ret_1"] = px.pct_change(1)
        df["ret_5"] = px.pct_change(5)
        df["ret_10"] = px.pct_change(10)

        df["mom_20"] = px.pct_change(20)
        df["mom_60"] = px.pct_change(60)

        df["vol_20"] = df["ret_1"].rolling(20).std()

        ma_20 = px.rolling(20).mean()
        ma_50 = px.rolling(50).mean()
        ma_100 = px.rolling(100).mean()

        df["trend_20_50"] = (ma_20 / (ma_50 + 1e-9)) - 1.0
        df["trend_20_100"] = (ma_20 / (ma_100 + 1e-9)) - 1.0

        df["rsi"] = compute_rsi(px, period=14)
        df["zscore_20"] = zscore(px, window=20)

        forward_return = px.shift(-config.horizon) / px - 1.0
        df["forward_return"] = forward_return
        df["target"] = (forward_return > 0).astype(int)

        all_data.append(df)

    if not all_data:
        raise ValueError("No valid assets with enough history to build features")

    data = pd.concat(all_data).reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"])
    data = data.replace([np.inf, -np.inf], np.nan)

    feature_cols = [
        "ret_1",
        "ret_5",
        "ret_10",
        "mom_20",
        "mom_60",
        "vol_20",
        "trend_20_50",
        "trend_20_100",
        "rsi",
        "zscore_20",
    ]

    data = data.dropna(subset=feature_cols + ["target"])

    data[feature_cols] = data.groupby("date")[feature_cols].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )

    data = data.sort_values(["date", "asset"]).reset_index(drop=True)
    data["target"] = data["target"].astype(int)

    return data


def get_feature_columns(feature_panel: pd.DataFrame) -> List[str]:
    excluded = {"date", "asset", "target", "forward_return"}
    return [c for c in feature_panel.columns if c not in excluded]
