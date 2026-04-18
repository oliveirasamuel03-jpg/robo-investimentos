from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class MLEngineConfig:
    lookahead: int = 5
    quantile_top: float = 0.3
    use_scaler: bool = True
    random_state: int = 42


def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def build_feature_panel(prices: pd.DataFrame, config: MLEngineConfig | None = None) -> pd.DataFrame:
    if config is None:
        config = MLEngineConfig()

    px = prices.sort_index().copy()
    rets = px.pct_change()

    mom_20 = px.pct_change(20)
    mom_60 = px.pct_change(60)
    mom_120 = px.pct_change(120)

    vol_20 = rets.rolling(20).std()

    # z-score do retorno atual contra janela de 20 dias
    mean_20 = rets.rolling(20).mean()
    std_20 = rets.rolling(20).std()
    z_20 = (rets - mean_20) / (std_20 + 1e-8)

    if "SPY" in px.columns:
        spy = px["SPY"]
        rel = px.divide(spy, axis=0)
        rel_mom_60 = rel.pct_change(60)
    else:
        rel_mom_60 = mom_60 * 0.0

    cs_mom_20 = _cs_rank(mom_20)
    cs_mom_60 = _cs_rank(mom_60)
    cs_mom_120 = _cs_rank(mom_120)
    cs_vol_20 = _cs_rank(-vol_20)
    cs_z_20 = _cs_rank(z_20)
    cs_rel = _cs_rank(rel_mom_60)

    features = {
        "mom_20": mom_20,
        "mom_60": mom_60,
        "mom_120": mom_120,
        "vol_20": vol_20,
        "z_20": z_20,
        "rel_mom_60": rel_mom_60,
        "cs_mom_20": cs_mom_20,
        "cs_mom_60": cs_mom_60,
        "cs_mom_120": cs_mom_120,
        "cs_vol_20": cs_vol_20,
        "cs_z_20": cs_z_20,
        "cs_rel": cs_rel,
    }

    fwd_ret = px.pct_change(config.lookahead).shift(-config.lookahead)

    rows = []
    for dt in px.index:
        for asset in px.columns:
            row = {
                "date": dt,
                "asset": asset,
                "target_ret": fwd_ret.loc[dt, asset],
            }
            for k, df in features.items():
                row[k] = df.loc[dt, asset]
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    import numpy as np
df["target"] = np.random.randint(0, 2, len(df)) (
        df.groupby("date")["target_ret"]
        .transform(lambda x: x >= x.quantile(1 - config.quantile_top))
        .astype(int)
    )

    return df


class EnsembleProbabilityModel:
    def __init__(self, config: MLEngineConfig | None = None):
        if config is None:
            config = MLEngineConfig()

        self.config = config

        steps = [("imputer", SimpleImputer(strategy="median"))]
        if config.use_scaler:
            steps.append(("scaler", StandardScaler()))

        self.preprocess = Pipeline(steps)

        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            random_state=config.random_state,
            n_jobs=-1,
        )

        self.gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=config.random_state,
        )

        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        Xp = self.preprocess.fit_transform(X)
        self.feature_names = list(X.columns)

        self.rf.fit(Xp, y)
        self.gb.fit(Xp, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xp = self.preprocess.transform(X)

        p1 = self.rf.predict_proba(Xp)[:, 1]
        p2 = self.gb.predict_proba(Xp)[:, 1]

        p = (p1 + p2) / 2.0
        return np.column_stack([1.0 - p, p])
