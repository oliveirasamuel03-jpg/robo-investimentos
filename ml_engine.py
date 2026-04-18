from dataclasses import dataclass
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
@dataclass
class MLEngineConfig:
    lookback: int = 20


# =========================
# FEATURE ENGINEERING
# =========================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return"] = df["close"].pct_change()
    df["ma9"] = df["close"].rolling(9).mean()
    df["ma21"] = df["close"].rolling(21).mean()
    df["volatility"] = df["return"].rolling(10).std()

    df = df.dropna()

    return df


def build_feature_panel(df: pd.DataFrame, config: MLEngineConfig):
    df_feat = create_features(df)

    X = df_feat[["return", "ma9", "ma21", "volatility"]]

    return X


def create_labels(df: pd.DataFrame):
    future_return = df["close"].pct_change().shift(-1)
    return (future_return > 0).astype(int)


# =========================
# MODEL
# =========================
class EnsembleProbabilityModel:
    def __init__(self):
        self.trained = False

    def fit(self, X, y):
        self.trained = True

    def predict(self, X):
        if not self.trained:
            return np.zeros(len(X))
        return np.random.randint(0, 2, size=len(X))

    def predict_proba(self, X):
        probs = np.random.rand(len(X))
        return np.vstack([1 - probs, probs]).T


# compatibilidade com código antigo
EnsembleModel = EnsembleProbabilityModel
