from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


WeightMethod = Literal["equal", "score", "rank"]


@dataclass
class StrategyConfig:
    probability_threshold: float = 0.60
    top_n: int = 2
    max_weight_per_asset: float = 0.40
    long_only: bool = True
    weight_method: WeightMethod = "score"
    min_assets: int = 1
    cash_threshold: float = 0.62
    cash_buffer: float = 0.02
    target_portfolio_vol: float = 0.12


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


def _clip_and_normalize(weights: pd.Series, max_weight_per_asset: float) -> pd.Series:
    weights = weights.clip(lower=0.0, upper=max_weight_per_asset)
    total = float(weights.sum())
    if total <= 0:
        return pd.Series(0.0, index=weights.index)
    return weights / total


def _build_equal_weights(selected_assets: pd.Index) -> pd.Series:
    if len(selected_assets) == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(selected_assets), index=selected_assets, dtype=float)


def _build_score_weights(selected_probs: pd.Series) -> pd.Series:
    if selected_probs.empty:
        return pd.Series(dtype=float)

    shifted = selected_probs - selected_probs.min()
    shifted = shifted + 1e-6
    total = float(shifted.sum())

    if total <= 0:
        return pd.Series(1.0 / len(selected_probs), index=selected_probs.index, dtype=float)

    return shifted / total


def _build_rank_weights(selected_probs: pd.Series) -> pd.Series:
    if selected_probs.empty:
        return pd.Series(dtype=float)

    sorted_assets = selected_probs.sort_values(ascending=False).index
    ranks = pd.Series(np.arange(len(sorted_assets), 0, -1, dtype=float), index=sorted_assets)
    return ranks / ranks.sum()


def _portfolio_volatility_scale(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    assets: list[str],
    target_portfolio_vol: float,
    lookback: int = 20,
    annualization: int = 252,
) -> float:
    if len(assets) == 0:
        return 0.0

    hist = prices[assets].loc[:date].pct_change().dropna().tail(lookback)
    if hist.empty or len(hist) < 5:
        return 1.0

    avg_vol = float(hist.std(ddof=0).mean() * np.sqrt(annualization))
    if avg_vol <= 0:
        return 1.0

    scale = target_portfolio_vol / avg_vol
    return float(np.clip(scale, 0.0, 1.0))


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
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "asset", "probability"])
    df = df.sort_values(["date", "probability"], ascending=[True, False]).reset_index(drop=True)

    output_rows = []

    for dt, group in df.groupby("date", sort=True):
        g = group.copy()
        g["rank"] = g["probability"].rank(method="first", ascending=False).astype(int)

        eligible = g[g["probability"] >= config.probability_threshold].copy()
        eligible = eligible.sort_values("probability", ascending=False).head(config.top_n)

        # cash filter inteligente
        if eligible.empty:
            g["selected"] = False
            g["weight"] = 0.0
            g["cash_mode"] = True
            output_rows.append(g)
            continue

        top_prob = float(eligible["probability"].max())
        avg_prob = float(eligible["probability"].mean())

        if (top_prob < config.cash_threshold) or ((top_prob - avg_prob) < config.cash_buffer):
            g["selected"] = False
            g["weight"] = 0.0
            g["cash_mode"] = True
            output_rows.append(g)
            continue

        if len(eligible) < config.min_assets:
            g["selected"] = False
            g["weight"] = 0.0
            g["cash_mode"] = True
            output_rows.append(g)
            continue

        if config.weight_method == "equal":
            weights = _build_equal_weights(eligible["asset"])
        elif config.weight_method == "rank":
            weights = _build_rank_weights(eligible.set_index("asset")["probability"])
        elif config.weight_method == "score":
            weights = _build_score_weights(eligible.set_index("asset")["probability"])
        else:
            raise ValueError(f"Unknown weight_method: {config.weight_method}")

        weights = _clip_and_normalize(weights, config.max_weight_per_asset)

        # volatility sizing
        if prices is not None:
            scale = _portfolio_volatility_scale(
                prices=prices,
                date=dt,
                assets=list(weights.index),
                target_portfolio_vol=config.target_portfolio_vol,
            )
            weights = weights * scale

        g["selected"] = g["asset"].isin(weights.index)
        g["weight"] = g["asset"].map(weights).fillna(0.0)
        g["cash_mode"] = False

        if config.long_only:
            g["weight"] = g["weight"].clip(lower=0.0)

        total_weight = float(g["weight"].sum())
        if total_weight > 0:
            g["weight"] = g["weight"] / max(total_weight, 1e-12) * min(total_weight, 1.0)

        output_rows.append(g)

    result = pd.concat(output_rows, ignore_index=True)
    return result[["date", "asset", "probability", "rank", "selected", "weight", "cash_mode"]]
