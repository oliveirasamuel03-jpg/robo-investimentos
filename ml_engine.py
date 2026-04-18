from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class MLEngineConfig:
    lookahead: int = 5
    quantile_top: float = 0.30
    use_scaler: bool = True
    random_state: int = 42
    min_history: int = 220


def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def _rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = df.rolling(window).mean()
    std = df.rolling(window).std()
    return (df - mean) / (std + 1e-8)


def build_feature_panel(
    prices: pd.DataFrame,
    config: MLEngineConfig | None = None,
) -> pd.DataFrame:
    """
    Versão anti-vazamento:
    - todas as features usam apenas informação disponível até t-1
    - target usa retorno futuro a partir de t
    - ranking cross-sectional por data
    """
    if config is None:
        config = MLEngineConfig()

    if prices.empty:
        raise ValueError("prices is empty")

    px = prices.sort_index().copy()
    px.index = pd.to_datetime(px.index)

    valid_cols = [c for c in px.columns if px[c].notna().sum() >= config.min_history]
    px = px[valid_cols]

    if px.empty:
        raise ValueError("No assets with sufficient history")

    # =========================
    # BASE SERIES
    # =========================
    ret_1 = px.pct_change(1)
    ret_3 = px.pct_change(3)
    ret_5 = px.pct_change(5)
    ret_10 = px.pct_change(10)
    mom_20 = px.pct_change(20)
    mom_60 = px.pct_change(60)
    mom_120 = px.pct_change(120)

    vol_20 = ret_1.rolling(20).std()
    vol_60 = ret_1.rolling(60).std()

    ma_20 = px.rolling(20).mean()
    ma_50 = px.rolling(50).mean()
    ma_100 = px.rolling(100).mean()
    ma_200 = px.rolling(200).mean()

    dist_ma20 = (px / (ma_20 + 1e-8)) - 1.0
    dist_ma50 = (px / (ma_50 + 1e-8)) - 1.0
    trend_20_50 = (ma_20 / (ma_50 + 1e-8)) - 1.0
    trend_50_100 = (ma_50 / (ma_100 + 1e-8)) - 1.0
    price_vs_ma200 = (px / (ma_200 + 1e-8)) - 1.0

    zret_20 = _rolling_zscore(ret_1, 20)
    zpx_20 = _rolling_zscore(px, 20)

    # =========================
    # RELATIVE FEATURES VS SPY
    # =========================
    if "SPY" in px.columns:
        spy = px["SPY"]

        spy_ret_1 = spy.pct_change(1)
        spy_mom_20 = spy.pct_change(20)
        spy_mom_60 = spy.pct_change(60)

        rel_ret_1 = ret_1.sub(spy_ret_1, axis=0)
        rel_mom_20 = mom_20.sub(spy_mom_20, axis=0)
        rel_mom_60 = mom_60.sub(spy_mom_60, axis=0)
    else:
        rel_ret_1 = ret_1 * 0.0
        rel_mom_20 = mom_20 * 0.0
        rel_mom_60 = mom_60 * 0.0

    # =========================
    # CROSS-SECTIONAL FEATURES
    # =========================
    cs_ret_3 = _cs_rank(ret_3)
    cs_ret_5 = _cs_rank(ret_5)
    cs_ret_10 = _cs_rank(ret_10)
    cs_mom_20 = _cs_rank(mom_20)
    cs_mom_60 = _cs_rank(mom_60)
    cs_mom_120 = _cs_rank(mom_120)

    cs_vol_20 = _cs_rank(-vol_20)
    cs_vol_60 = _cs_rank(-vol_60)

    cs_dist_ma20 = _cs_rank(dist_ma20)
    cs_dist_ma50 = _cs_rank(dist_ma50)
    cs_trend_20_50 = _cs_rank(trend_20_50)
    cs_trend_50_100 = _cs_rank(trend_50_100)
    cs_price_vs_ma200 = _cs_rank(price_vs_ma200)

    cs_zret_20 = _cs_rank(zret_20)
    cs_zpx_20 = _cs_rank(zpx_20)

    cs_rel_ret_1 = _cs_rank(rel_ret_1)
    cs_rel_mom_20 = _cs_rank(rel_mom_20)
    cs_rel_mom_60 = _cs_rank(rel_mom_60)

    feature_map = {
        "ret_3": ret_3,
        "ret_5": ret_5,
        "ret_10": ret_10,
        "mom_20": mom_20,
        "mom_60": mom_60,
        "mom_120": mom_120,
        "vol_20": vol_20,
        "vol_60": vol_60,
        "dist_ma20": dist_ma20,
        "dist_ma50": dist_ma50,
        "trend_20_50": trend_20_50,
        "trend_50_100": trend_50_100,
        "price_vs_ma200": price_vs_ma200,
        "zret_20": zret_20,
        "zpx_20": zpx_20,
        "rel_ret_1": rel_ret_1,
        "rel_mom_20": rel_mom_20,
        "rel_mom_60": rel_mom_60,
        "cs_ret_3": cs_ret_3,
        "cs_ret_5": cs_ret_5,
        "cs_ret_10": cs_ret_10,
        "cs_mom_20": cs_mom_20,
        "cs_mom_60": cs_mom_60,
        "cs_mom_120": cs_mom_120,
        "cs_vol_20": cs_vol_20,
        "cs_vol_60": cs_vol_60,
        "cs_dist_ma20": cs_dist_ma20,
        "cs_dist_ma50": cs_dist_ma50,
        "cs_trend_20_50": cs_trend_20_50,
        "cs_trend_50_100": cs_trend_50_100,
        "cs_price_vs_ma200": cs_price_vs_ma200,
        "cs_zret_20": cs_zret_20,
        "cs_zpx_20": cs_zpx_20,
        "cs_rel_ret_1": cs_rel_ret_1,
        "cs_rel_mom_20": cs_rel_mom_20,
        "cs_rel_mom_60": cs_rel_mom_60,
    }

    # =========================
    # ANTI-LEAKAGE: SHIFT 1 BAR
    # =========================
    feature_map = {name: frame.shift(1) for name, frame in feature_map.items()}

    # =========================
    # TARGET FUTURO
    # retorno entre t e t+lookahead
    # =========================
    target_ret = px.shift(-config.lookahead) / px - 1.0

    rows = []
    for dt in px.index:
        for asset in px.columns:
            row = {
                "date": dt,
                "asset": asset,
                "target_ret": target_ret.loc[dt, asset],
            }
            for name, frame in feature_map.items():
                row[name] = frame.loc[dt, asset]
            rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # =========================
    # TARGET CROSS-SECTIONAL
    # top quantile do retorno futuro por data
    # =========================
    df["target"] = (
        df.groupby("date")["target_ret"]
        .transform(lambda x: x >= x.quantile(1 - config.quantile_top))
        .astype(int)
    )

    return df.sort_values(["date", "asset"]).reset_index(drop=True)


class EnsembleProbabilityModel:
    def __init__(self, config: MLEngineConfig | None = None):
        if config is None:
            config = MLEngineConfig()

        self.config = config

        steps = [("imputer", SimpleImputer(strategy="median"))]
        if config.use_scaler:
            steps.append(("scaler", StandardScaler()))

        self.preprocess = Pipeline(steps)

        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            random_state=config.random_state,
            n_jobs=-1,
        )

        self.gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=config.random_state,
        )

        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
        y = pd.Series(y).astype(int)

        self.feature_names_ = list(X.columns)
        Xp = self.preprocess.fit_transform(X)

        self.rf.fit(Xp, y)
        self.gb.fit(Xp, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
        Xp = self.preprocess.transform(X)

        p1 = self.rf.predict_proba(Xp)[:, 1]
        p2 = self.gb.predict_proba(Xp)[:, 1]
        p = (p1 + p2) / 2.0

        return np.column_stack([1.0 - p, p])
