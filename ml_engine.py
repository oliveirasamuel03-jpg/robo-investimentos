from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class MLEngineConfig:
    horizon: int = 1
    random_state: int = 42


class EnsembleProbabilityModel:
    """
    Nível 2:
    - base models: RandomForest + GradientBoosting
    - meta model: LogisticRegression sobre probabilidades e dispersão
    """

    def __init__(self, config: MLEngineConfig | None = None):
        self.config = config or MLEngineConfig()

        self.rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

        self.gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            random_state=self.config.random_state,
        )

        self.meta = LogisticRegression(
            random_state=self.config.random_state,
            max_iter=1000,
        )

        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def _clean_X(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(X).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = self._clean_X(X)
        y = pd.Series(y).astype(int)

        self.feature_names_ = list(X.columns)

        self.rf.fit(X, y)
        self.gb.fit(X, y)

        rf_p = self.rf.predict_proba(X)[:, 1]
        gb_p = self.gb.predict_proba(X)[:, 1]

        meta_X = pd.DataFrame(
            {
                "rf_p": rf_p,
                "gb_p": gb_p,
                "avg_p": 0.5 * (rf_p + gb_p),
                "diff_p": np.abs(rf_p - gb_p),
            }
        )
        self.meta.fit(meta_X, y)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame):
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before predict_proba")

        X = self._clean_X(X)

        rf_p = self.rf.predict_proba(X)[:, 1]
        gb_p = self.gb.predict_proba(X)[:, 1]

        meta_X = pd.DataFrame(
            {
                "rf_p": rf_p,
                "gb_p": gb_p,
                "avg_p": 0.5 * (rf_p + gb_p),
                "diff_p": np.abs(rf_p - gb_p),
            }
        )

        final_p = self.meta.predict_proba(meta_X)[:, 1]
        return np.column_stack([1.0 - final_p, final_p])


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


def build_feature_panel(prices: pd.DataFrame, config: MLEngineConfig | None = None):
    config = config or MLEngineConfig()

    if prices.empty:
        raise ValueError("prices is empty")

    spy = prices["SPY"] if "SPY" in prices.columns else None
    rows = []

    for asset in prices.columns:
        px = pd.to_numeric(prices[asset], errors="coerce").dropna()

        if len(px) < 220:
            continue

        df = pd.DataFrame(index=px.index)
        df["date"] = px.index
        df["asset"] = asset

        ret_1 = px.pct_change()

        df["ret_1"] = ret_1
        df["ret_5"] = px.pct_change(5)
        df["ret_10"] = px.pct_change(10)

        df["mom_20"] = px.pct_change(20)
        df["mom_60"] = px.pct_change(60)
        df["mom_120"] = px.pct_change(120)

        df["vol_20"] = ret_1.rolling(20).std()
        df["vol_60"] = ret_1.rolling(60).std()

        ma20 = px.rolling(20).mean()
        ma50 = px.rolling(50).mean()
        ma100 = px.rolling(100).mean()
        ma200 = px.rolling(200).mean()

        df["trend_20_50"] = (ma20 / (ma50 + 1e-9)) - 1.0
        df["trend_50_200"] = (ma50 / (ma200 + 1e-9)) - 1.0
        df["price_vs_ma200"] = (px / (ma200 + 1e-9)) - 1.0

        df["rsi"] = compute_rsi(px, period=14)
        df["zscore_20"] = zscore(px, window=20)

        if spy is not None:
            spy_ret_1 = spy.pct_change()
            spy_mom_20 = spy.pct_change(20)
            spy_ma200 = spy.rolling(200).mean()

            df["rel_strength_1"] = ret_1.reindex(df.index).values - spy_ret_1.reindex(df.index).values
            df["rel_strength_20"] = px.pct_change(20).reindex(df.index).values - spy_mom_20.reindex(df.index).values
            df["bull_regime"] = (spy.reindex(df.index) > spy_ma200.reindex(df.index)).astype(int).values
            df["spy_vol_20"] = spy_ret_1.reindex(df.index).rolling(20).std().values
        else:
            df["rel_strength_1"] = 0.0
            df["rel_strength_20"] = 0.0
            df["bull_regime"] = 1
            df["spy_vol_20"] = 0.0

        future = px.shift(-config.horizon) / px - 1.0
        df["forward_return"] = future
        df["target"] = (future > 0).astype(int)

        rows.append(df)

    if not rows:
        raise ValueError("No valid assets with enough history to build features")

    data = pd.concat(rows).reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"])
    data = data.replace([np.inf, -np.inf], np.nan)

    feature_cols = [
        "ret_1", "ret_5", "ret_10",
        "mom_20", "mom_60", "mom_120",
        "vol_20", "vol_60",
        "trend_20_50", "trend_50_200", "price_vs_ma200",
        "rsi", "zscore_20",
        "rel_strength_1", "rel_strength_20",
        "bull_regime", "spy_vol_20",
    ]

    data = data.dropna(subset=feature_cols + ["target"])

    # normalização cross-sectional diária
    data[feature_cols] = data.groupby("date")[feature_cols].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )

    data = data.sort_values(["date", "asset"]).reset_index(drop=True)
    data["target"] = data["target"].astype(int)

    return data


def get_feature_columns(feature_panel: pd.DataFrame) -> List[str]:
    excluded = {"date", "asset", "target", "forward_return"}
    return [c for c in feature_panel.columns if c not in excluded]
