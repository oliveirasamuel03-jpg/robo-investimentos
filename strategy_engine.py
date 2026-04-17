from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


WeightMethod = Literal["equal", "score", "rank"]


@dataclass
class StrategyConfig:
    probability_threshold: float = 0.65
    top_n: int = 5
    max_weight_per_asset: float = 0.40
    long_only: bool = True
    weight_method: WeightMethod = "score"
    min_assets: int = 1


def market_filter(prices: pd.DataFrame, benchmark: str = "SPY", window: int = 200) -> pd.Series:
    if benchmark not in prices.columns:
        return pd.Series(True, index=prices.index)

    benchmark_px = prices[benchmark].copy()
    ma = benchmark_px.rolling(window).mean()
    regime = benchmark_px > ma
    regime = regime.fillna(False)
    regime.name = "regime"
    return regime


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
    ranks = pd.Series(
        np.arange(len(sorted_assets), 0, -1, dtype=float),
        index=sorted_assets,
    )
    return ranks / ranks.sum()


def generate_target_weights(
    predictions: pd.DataFrame,
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

        if len(eligible) < config.min_assets:
            g["selected"] = False
            g["weight"] = 0.0
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

        g["selected"] = g["asset"].isin(weights.index)
        g["weight"] = g["asset"].map(weights).fillna(0.0)

        if config.long_only:
            g["weight"] = g["weight"].clip(lower=0.0)

        total_weight = float(g["weight"].sum())
        if total_weight > 0:
            g["weight"] = g["weight"] / total_weight

        output_rows.append(g)

    result = pd.concat(output_rows, ignore_index=True)
    result = result[["date", "asset", "probability", "rank", "selected", "weight"]]
    return result


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


def generate_signals_from_probabilities(
    predictions: pd.DataFrame,
    probability_threshold: float = 0.65,
) -> pd.DataFrame:
    required_cols = {"date", "asset", "probability"}
    missing = required_cols - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")

    df = predictions.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["signal"] = (df["probability"] >= probability_threshold).astype(int)
    return df[["date", "asset", "probability", "signal"]]
