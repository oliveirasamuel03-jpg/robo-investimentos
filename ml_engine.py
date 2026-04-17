from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


@dataclass
class MLEngineConfig:
    horizon: int = 1
    random_state: int = 42


class EnsembleProbabilityModel:
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

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.rf.fit(X, y)
        self.gb.fit(X, y)

    def predict_proba(self, X: pd.DataFrame):
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        p1 = self.rf.predict_proba(X)[:, 1]
        p2 = self.gb.predict_proba(X)[:, 1]

        p = (p1 + p2) / 2
        return np.column_stack([1 - p, p])


def compute_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def build_feature_panel(prices: pd.DataFrame, config: MLEngineConfig | None = None):
    config = config or MLEngineConfig()

    spy = prices["SPY"] if "SPY" in prices.columns else None

    rows = []

    for asset in prices.columns:
        px = prices[asset].dropna()

        if len(px) < 120:
            continue

        df = pd.DataFrame(index=px.index)

        df["date"] = px.index
        df["asset"] = asset

        ret = px.pct_change()

        # retornos básicos
        df["ret_1"] = ret
        df["ret_5"] = px.pct_change(5)
        df["ret_10"] = px.pct_change(10)

        # momentum
        df["mom_20"] = px.pct_change(20)
        df["mom_60"] = px.pct_change(60)

        # volatilidade
        df["vol_20"] = ret.rolling(20).std()

        # tendência
        ma20 = px.rolling(20).mean()
        ma50 = px.rolling(50).mean()
        df["trend"] = (ma20 / (ma50 + 1e-9)) - 1

        # RSI
        df["rsi"] = compute_rsi(px)

        # 🔥 FORÇA RELATIVA VS SPY
        if spy is not None and asset != "SPY":
            spy_ret = spy.pct_change()
            df["rel_strength"] = ret - spy_ret
        else:
            df["rel_strength"] = 0

        # 🔥 REGIME COMO FEATURE
        if spy is not None:
            spy_ma200 = spy.rolling(200).mean()
            df["bull_regime"] = (spy > spy_ma200).astype(int)
        else:
            df["bull_regime"] = 1

        # target
        future = px.shift(-config.horizon) / px - 1
        df["target"] = (future > 0).astype(int)

        rows.append(df)

    data = pd.concat(rows)

    # 🔥 NORMALIZAÇÃO CROSS-SECTIONAL
    feature_cols = [
        "ret_1", "ret_5", "ret_10",
        "mom_20", "mom_60",
        "vol_20",
        "trend",
        "rsi",
        "rel_strength",
        "bull_regime",
    ]

    data = data.dropna()

    data[feature_cols] = data.groupby("date")[feature_cols].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )

    data = data.sort_values(["date", "asset"])

    return data
