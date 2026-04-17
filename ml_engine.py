import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def create_features(prices):
    returns = prices.pct_change()

    features = pd.DataFrame(index=prices.index)

    for col in prices.columns:
        features[f"{col}_ret1"] = returns[col]
        features[f"{col}_ret5"] = prices[col].pct_change(5)
        features[f"{col}_vol"] = returns[col].rolling(10).std()
        features[f"{col}_ma"] = prices[col].rolling(10).mean() / prices[col]

    features = features.dropna()
    return features

def create_labels(prices):
    future_ret = prices.pct_change().shift(-1)
    return (future_ret > 0).astype(int)

class EnsembleModel:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100)
        self.gb = GradientBoostingClassifier()

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.gb.fit(X, y)

    def predict_proba(self, X):
        p1 = self.rf.predict_proba(X)[:, 1]
        p2 = self.gb.predict_proba(X)[:, 1]
        return (p1 + p2) / 2