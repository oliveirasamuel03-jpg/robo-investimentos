import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return"] = df["close"].pct_change()
    df["ma9"] = df["close"].rolling(9).mean()
    df["ma21"] = df["close"].rolling(21).mean()

    df["volatility"] = df["return"].rolling(10).std()

    df = df.dropna()

    return df


def create_labels(df: pd.DataFrame) -> pd.Series:
    future_return = df["close"].pct_change().shift(-1)
    return (future_return > 0).astype(int)


class EnsembleModel:
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
