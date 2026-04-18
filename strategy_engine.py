from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    probability_threshold: float = 0.50
    top_n: int = 2
    max_weight_per_asset: float = 0.35
    cash_threshold: float = 0.50
    target_portfolio_vol: float = 0.08

    # Hedge Fund 2
    min_weight_per_asset: float = 0.00
    conviction_spread_min: float = 0.03
    min_gross_exposure: float = 0.20
    max_gross_exposure: float = 1.00
    neutral_gross_exposure: float = 0.60
    stress_gross_exposure: float = 0.25
    rank_smoothing_power: float = 1.25


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


def regime_state(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    fast_ma_window: int = 50,
    slow_ma_window: int = 200,
    vol_window: int = 20,
    vol_threshold: float = 0.025,
) -> pd.Series:
    if benchmark not in prices.columns:
        return pd.Series("neutral", index=prices.index, name="regime_state")

    px = prices[benchmark].astype(float)
    fast_ma = px.rolling(fast_ma_window).mean()
    slow_ma = px.rolling(slow_ma_window).mean()
    vol = px.pct_change().rolling(vol_window).std()

    state = pd.Series("neutral", index=prices.index, dtype="object")
    state[(px > slow_ma) & (fast_ma > slow_ma) & (vol < vol_threshold)] = "bull"
    state[(px <= slow_ma) | (fast_ma <= slow_ma)] = "stress"
    state.name = "regime_state"
    return state


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
    state = regime_state(
        prices=prices,
        benchmark=benchmark,
        fast_ma_window=fast_ma_window,
        slow_ma_window=slow_ma_window,
        vol_window=vol_window,
        vol_threshold=vol_threshold,
    )

    df = pd.DataFrame(index=prices.index)
    df["regime"] = regime if use_regime_filter else True
    df["vol_filter"] = vol_ok if use_volatility_filter else True
    df["trade_allowed"] = df["regime"] & df["vol_filter"]
    df["regime_state"] = state
    return df


def _clip_and_normalize(weights: pd.Series, cfg: StrategyConfig) -> pd.Series:
    weights = weights.clip(lower=cfg.min_weight_per_asset, upper=cfg.max_weight_per_asset)
    total = float(weights.sum())
    if total <= 0:
        return pd.Series(0.0, index=weights.index, dtype=float)
    return weights / total


def _score_weights(selected_probs: pd.Series, cfg: StrategyConfig) -> pd.Series:
    if selected_probs.empty:
        return pd.Series(dtype=float)

    base = selected_probs - selected_probs.min() + 1e-6
    base = np.power(base, cfg.rank_smoothing_power)
    weights = pd.Series(base, index=selected_probs.index, dtype=float)

    # penaliza concentração excessiva
    weights = np.sqrt(weights)
    total = float(weights.sum())
    if total <= 0:
        return pd.Series(1.0 / len(selected_probs), index=selected_probs.index, dtype=float)

    return weights / total


def _estimate_portfolio_vol(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    assets: list[str],
    lookback: int = 20,
    annualization: int = 252,
) -> float:
    if len(assets) == 0:
        return 0.0

    hist = prices[assets].loc[:date].pct_change().dropna().tail(lookback)
    if hist.empty or len(hist) < 5:
        return 0.0

    cov = hist.cov().values
    n = len(assets)
    ew = np.repeat(1.0 / n, n)
    vol = float(np.sqrt(np.maximum(ew @ cov @ ew, 0.0)) * np.sqrt(annualization))
    return vol


def _regime_exposure_multiplier(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    cfg: StrategyConfig,
) -> float:
    if "SPY" not in prices.columns or date not in prices.index:
        return cfg.neutral_gross_exposure

    px = prices["SPY"].astype(float).loc[:date]
    if len(px) < 200:
        return cfg.neutral_gross_exposure

    fast = px.rolling(50).mean().iloc[-1]
    slow = px.rolling(200).mean().iloc[-1]
    last = px.iloc[-1]
    vol = px.pct_change().rolling(20).std().iloc[-1]

    if pd.isna(fast) or pd.isna(slow) or pd.isna(vol):
        return cfg.neutral_gross_exposure

    if (last > slow) and (fast > slow) and (vol < 0.025):
        return cfg.max_gross_exposure

    if (last <= slow) or (fast <= slow):
        return cfg.stress_gross_exposure

    return cfg.neutral_gross_exposure


def _conviction_exposure_multiplier(
    selected_probs: pd.Series,
    cfg: StrategyConfig,
) -> float:
    if selected_probs.empty:
        return 0.0

    top_prob = float(selected_probs.max())
    avg_prob = float(selected_probs.mean())
    spread = top_prob - avg_prob

    if top_prob < cfg.cash_threshold:
        return cfg.min_gross_exposure

    if spread < cfg.conviction_spread_min:
        return cfg.neutral_gross_exposure

    raw = cfg.neutral_gross_exposure + 2.5 * spread
    return float(np.clip(raw, cfg.min_gross_exposure, cfg.max_gross_exposure))


def _volatility_target_scale(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    assets: list[str],
    cfg: StrategyConfig,
) -> float:
    est_vol = _estimate_portfolio_vol(prices, date, assets)
    if est_vol <= 0:
        return 1.0

    scale = cfg.target_portfolio_vol / est_vol
    return float(np.clip(scale, 0.25, 1.25))


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

    output_rows = []

    for dt, group in df.groupby("date", sort=True):
        g = group.copy().sort_values("probability", ascending=False)
        g["rank"] = np.arange(1, len(g) + 1)

        selected = g[g["probability"] >= config.probability_threshold].head(config.top_n).copy()

        # fallback defensivo: se nada passar, usa top_n com baixa exposição
        fallback_mode = False
        if selected.empty:
            selected = g.head(config.top_n).copy()
            fallback_mode = True

        probs = selected.set_index("asset")["probability"]
        weights = _score_weights(probs, config)

        if prices is not None and not weights.empty:
            regime_scale = _regime_exposure_multiplier(prices, dt, config)
            conviction_scale = _conviction_exposure_multiplier(probs, config)
            vol_scale = _volatility_target_scale(prices, dt, list(weights.index), config)

            gross_target = min(regime_scale, conviction_scale) * vol_scale
            if fallback_mode:
                gross_target = min(gross_target, config.min_gross_exposure)

            gross_target = float(np.clip(gross_target, 0.0, config.max_gross_exposure))
            weights = weights * gross_target

        g["selected"] = g["asset"].isin(weights.index)
        g["weight"] = g["asset"].map(weights).fillna(0.0)
        g["cash_mode"] = float(g["weight"].sum()) <= 1e-12

        output_rows.append(g)

    result = pd.concat(output_rows, ignore_index=True)

    result = (
        result.groupby(["date", "asset"], as_index=False)
        .agg(
            probability=("probability", "mean"),
            rank=("rank", "min"),
            selected=("selected", "max"),
            weight=("weight", "sum"),
            cash_mode=("cash_mode", "max"),
        )
        .sort_values(["date", "asset"])
        .reset_index(drop=True)
    )

    return result[["date", "asset", "probability", "rank", "selected", "weight", "cash_mode"]]
