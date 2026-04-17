from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


@dataclass
class MLEngineConfig:
    horizon: int = 1
    random_state: int = 42


class EnsembleModel:
    def __init__(self):
        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
        )

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.gb.fit(X, y)

    def predict_proba(self, X):
        p1 = self.rf.predict_proba(X)[:, 1]
        p2 = self.gb.predict_proba(X)[:, 1]
        p = (p1 + p2) / 2
        return np.column_stack([1 - p, p])


# =========================
# FEATURES AVANÇADAS
# =========================

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def zscore(series, window=20):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-9)


def build_feature_panel(prices: pd.DataFrame):
    all_data = []

    for asset in prices.columns:
        px = prices[asset].dropna()

        if len(px) < 100:
            continue

        df = pd.DataFrame()
        df["date"] = px.index
        df["asset"] = asset

        # RETURNS
        df["ret_1"] = px.pct_change(1)
        df["ret_5"] = px.pct_change(5)
        df["ret_10"] = px.pct_change(10)

        # MOMENTUM
        df["mom_20"] = px.pct_change(20)
        df["mom_60"] = px.pct_change(60)

        # VOL
        df["vol_20"] = df["ret_1"].rolling(20).std()

        # TREND
        ma_20 = px.rolling(20).mean()
        ma_50 = px.rolling(50).mean()

        df["trend"] = ma_20 / ma_50 - 1

        # RSI
        df["rsi"] = compute_rsi(px)

        # ZSCORE
        df["zscore"] = zscore(px)

        # TARGET
        future_return = px.shift(-1) / px - 1
        df["target"] = (future_return > 0).astype(int)

        all_data.append(df)

    data = pd.concat(all_data).dropna()

    # =========================
    # CROSS SECTIONAL NORMALIZATION
    # =========================
    features = [c for c in data.columns if c not in ["date", "asset", "target"]]

    data[features] = data.groupby("date")[features].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )

    return data
