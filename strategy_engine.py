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


def generate_target_weights(predictions: pd.DataFrame, config: StrategyConfig):
    df = predictions.copy()
    df["date"] = pd.to_datetime(df["date"])

    results = []

    for dt, group in df.groupby("date"):
        g = group.copy().sort_values("probability", ascending=False)

        # ==============================
        # 1. Seleção principal
        # ==============================
        selected = g[g["probability"] >= config.probability_threshold]

        # ==============================
        # 2. Fallback (NUNCA ficar zerado)
        # ==============================
        if selected.empty:
            selected = g.head(config.top_n)

            # peso reduzido (modo defensivo)
            weights = np.ones(len(selected)) / len(selected)
            weights *= 0.3  # só 30% do capital

        else:
            selected = selected.head(config.top_n)

            # ==============================
            # 3. Peso por score
            # ==============================
            probs = selected["probability"].values
            weights = probs - probs.min() + 1e-6
            weights = weights / weights.sum()

        # ==============================
        # 4. Cash inteligente (reduz, não trava)
        # ==============================
        max_prob = selected["probability"].max()

        if max_prob < config.cash_threshold:
            weights *= 0.5  # reduz exposição, mas não zera

        # ==============================
        # 5. Limite por ativo
        # ==============================
        weights = np.clip(weights, 0, config.max_weight_per_asset)

        # normaliza
        if weights.sum() > 0:
            weights = weights / weights.sum()

        selected["weight"] = weights
        g["weight"] = 0.0

        g.loc[selected.index, "weight"] = selected["weight"]

        results.append(g)

    result = pd.concat(results)

    # ==============================
    # GARANTIA FINAL (sem duplicata)
    # ==============================
    result = (
        result.groupby(["date", "asset"], as_index=False)
        .agg({
            "probability": "mean",
            "weight": "sum"
        })
    )

    return result
