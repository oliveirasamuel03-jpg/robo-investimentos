from dataclasses import dataclass
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
@dataclass
class MLEngineConfig:
    lookback: int = 50


# =========================
# RSI
# =========================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


# =========================
# FEATURES
# =========================
def build_feature_panel(df: pd.DataFrame, config: MLEngineConfig):
    df = df.copy()

    # garantir close
    if "close" not in df.columns and "Close" in df.columns:
        df["close"] = df["Close"]

    df["ma50"] = df["close"].rolling(50).mean()
    df["rsi"] = calculate_rsi(df["close"])
    df["volatility"] = df["close"].pct_change().rolling(10).std()

    df = df.fillna(0)

    return df[["close", "ma50", "rsi", "volatility"]]


def create_labels(df: pd.DataFrame):
    df = df.copy()

    if "close" not in df.columns and "Close" in df.columns:
        df["close"] = df["Close"]

    future_return = df["close"].pct_change().shift(-1)

    return (future_return > 0).astype(int).fillna(0)


# =========================
# MODELO COM ESTRATÉGIA REAL
# =========================
class EnsembleProbabilityModel:

    def __init__(self):
        pass

    def fit(self, X, y):
        pass  # não precisa treinar (estratégia manual)

    def predict_proba(self, X):
        probs = []

        for i in range(len(X)):
            row = X.iloc[i]

            price = row["close"]
            ma50 = row["ma50"]
            rsi = row["rsi"]
            vol = row["volatility"]

            prob = 0.5  # neutro

            # filtro de volatilidade
            if vol < 0.005:
                prob = 0.3

            # tendência + pullback
            elif price > ma50 and 40 < rsi < 60:
                prob = 0.8  # compra forte

            # sobrecomprado / reversão
            elif rsi > 70 or price < ma50:
                prob = 0.2  # venda

            probs.append(prob)

        probs = np.array(probs)

        return np.vstack([1 - probs, probs]).T


# compatibilidade
EnsembleModel = EnsembleProbabilityModel
