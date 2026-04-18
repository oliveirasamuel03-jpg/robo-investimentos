from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class MLEngineConfig:
    lookback: int = 20


def _normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}
    for col in df.columns:
        col_str = str(col)
        lower = col_str.lower()

        if lower == "close":
            rename_map[col] = "close"
        elif lower == "open":
            rename_map[col] = "open"
        elif lower == "high":
            rename_map[col] = "high"
        elif lower == "low":
            rename_map[col] = "low"
        elif lower == "volume":
            rename_map[col] = "volume"

    df = df.rename(columns=rename_map)

    if "close" not in df.columns:
        raise ValueError("DataFrame precisa ter coluna 'close' ou 'Close'.")

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_price_columns(df)

    out = df.copy()

    out["return"] = out["close"].pct_change()
    out["ma9"] = out["close"].rolling(9).mean()
    out["ma21"] = out["close"].rolling(21).mean()
    out["volatility"] = out["return"].rolling(10).std()

    if "volume" not in out.columns:
        out["volume"] = 0.0

    out = out.dropna().reset_index(drop=True)
    return out


def build_feature_panel(df: pd.DataFrame, config: MLEngineConfig):
    feat = create_features(df)
    X = feat[["return", "ma9", "ma21", "volatility"]].copy()
    return X


def create_labels(df: pd.DataFrame):
    df = _normalize_price_columns(df).copy()
    future_return = df["close"].pct_change().shift(-1)
    y = (future_return > 0).astype(int)
    return y.dropna().reset_index(drop=True)


class EnsembleProbabilityModel:
    def __init__(self):
        self.trained = False
        self.default_prob = 0.5

    def fit(self, X, y):
        self.trained = True
        if len(y) > 0:
            y_arr = np.asarray(y, dtype=float)
            self.default_prob = float(np.clip(np.nanmean(y_arr), 0.05, 0.95))

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X):
        n = len(X)
        probs = np.full(n, self.default_prob, dtype=float)
        return np.vstack([1 - probs, probs]).T


# compatibilidade com imports antigos
EnsembleModel = EnsembleProbabilityModel
