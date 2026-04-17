from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    probability_threshold: float = 0.60
    top_n: int = 3
    max_weight_per_asset: float = 0.40
    cash_threshold: float = 0.58
    target_portfolio_vol: float = 0.15


def market_filter(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    fast_ma_window: int = 50,
    slow_ma_window: int = 200,
) -> pd.Series:
    if benchmark not in prices.columns:
        return pd.Series(True, index=prices.index, name="regime")

    benchmark_px = prices[benchmark].astype(float)
    fast_ma = benchmark_px.rolling(fast_ma_window).mean()
    slow_ma = benchmark_px.rolling(slow_ma_window).mean()

    regime = (benchmark_px > slow_ma) & (fast_ma > slow_ma)
    regime = regime.fillna(False)
    regime.name = "regime"
    return regime


def volatility_filter(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    vol_window: int = 20,
    vol_threshold: float = 0.025,
) -> pd.Series:
    if benchmark not in prices.columns:
        return pd.Series(True, index=prices.index, name="vol_filter")

    benchmark_px = prices[benchmark].astype(float)
    returns = benchmark_px.pct_change()
    realized_vol = returns.rolling(vol_window).std()

    vol_ok = realized_vol < vol_threshold
    vol_ok = vol_ok.fillna(False)
    vol_ok.name = "vol_filter"
    return vol_ok


def combined_market_filter(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    fast_ma_window: int = 50,
    slow_ma_window: int = 200,
    vol_window: int = 20,
    vol_threshold: float = 0.025,
    use_regime_filter: bool = True,
    use_volatility_filter: bool = True,
) -> pd.DataFrame:
    regime = market_filter(
        prices=prices,
        benchmark=benchmark,
        fast_ma_window=fast_ma_window,
        slow_ma_window=slow_ma_window,
    )
    vol_ok = volatility_filter(
        prices=prices,
        benchmark=benchmark,
        vol_window=vol_window,
        vol_threshold=vol_threshold,
    )

    df = pd.DataFrame(index=prices.index)
    df["regime"] = regime if use_regime_filter else True
    df["vol_filter"] = vol_ok if use_volatility_filter else True
    df["trade_allowed"] = df["regime"] & df["vol_filter"]
    return df


def generate_target_weights(
    predictions: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    config: StrategyConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = StrategyConfig()

    required_cols = {"date", "asset", "probability"}
    missing = required_cols - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")

    df = predictions.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["asset"] = df["asset"].astype(str)
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "asset", "probability"])

    df = (
        df.groupby(["date", "asset"], as_index=False)["probability"]
        .mean()
        .sort_values(["date", "probability"], ascending=[True, False])
        .reset_index(drop=True)
    )

    results = []

    for dt, group in df.groupby("date", sort=True):
        g = group.copy().sort_values("probability", ascending=False)

        selected = g[g["probability"] >= config.probability_threshold].copy()

        if selected.empty:
            selected = g.head(config.top_n).copy()
            defensive_scale = 0.30
        else:
            selected = selected.head(config.top_n).copy()
            defensive_scale = 1.00

        probs = selected["probability"].values.astype(float)

        if len(selected) == 1:
            weights = np.array([1.0], dtype=float)
        else:
            weights = probs - probs.min() + 1e-6
            weights = weights / weights.sum()

        max_prob = float(selected["probability"].max())
        if max_prob < config.cash_threshold:
            defensive_scale *= 0.50

        weights = weights * defensive_scale
        weights = np.clip(weights, 0.0, config.max_weight_per_asset)

        if weights.sum() > 0:
            weights = weights / weights.sum() * min(defensive_scale, 1.0)

        selected["weight"] = weights

        g["selected"] = False
        g["cash_mode"] = False
        g["weight"] = 0.0

        g.loc[selected.index, "selected"] = True
        g.loc[selected.index, "weight"] = selected["weight"].values
        g["cash_mode"] = g["weight"].sum() == 0.0

        results.append(g)

    result = pd.concat(results, ignore_index=True)

    result = (
        result.groupby(["date", "asset"], as_index=False)
        .agg(
            probability=("probability", "mean"),
            selected=("selected", "max"),
            weight=("weight", "sum"),
            cash_mode=("cash_mode", "max"),
        )
        .sort_values(["date", "asset"])
        .reset_index(drop=True)
    )

    result["rank"] = result.groupby("date")["probability"].rank(method="first", ascending=False).astype(int)

    return result[["date", "asset", "probability", "rank", "selected", "weight", "cash_mode"]]


def build_weight_matrix(
    weighted_predictions: pd.DataFrame,
    index: pd.Index | None = None,
    columns: pd.Index | None = None,
) -> pd.DataFrame:
    required_cols = {"date", "asset", "weight"}
    missing = required_cols - set(weighted_predictions.columns)
    if missing:
        raise ValueError(f"weighted_predictions missing required columns: {sorted(missing)}")

    df = weighted_predictions.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["asset"] = df["asset"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    df = df.groupby(["date", "asset"], as_index=False)["weight"].sum()

    matrix = df.pivot_table(
        index="date",
        columns="asset",
        values="weight",
        aggfunc="sum",
        fill_value=0.0,
    )

    if index is not None:
        matrix = matrix.reindex(index=index).fillna(0.0)

    if columns is not None:
        matrix = matrix.reindex(columns=columns).fillna(0.0)

    return matrix.sort_index()
