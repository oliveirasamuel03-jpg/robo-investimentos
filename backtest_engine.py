from __future__ import annotations

import numpy as np
import pandas as pd


def _ensure_close(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if data is None or len(data) == 0:
        return pd.DataFrame({"close": np.linspace(100, 101, 50)})

    cols = {str(c).lower(): c for c in data.columns}

    if "close" in cols:
        data["close"] = data[cols["close"]]
    elif "adj close" in cols:
        data["close"] = data[cols["adj close"]]
    else:
        data["close"] = np.linspace(100, 101, len(data))

    return data


def _normalize_signals(signals, target_len: int) -> pd.Series:
    if isinstance(signals, pd.DataFrame):
        arr = signals.to_numpy()
    elif isinstance(signals, pd.Series):
        arr = signals.to_numpy()
    else:
        arr = np.array(signals)

    if arr.ndim == 2:
        arr = arr[:, -1]

    arr = np.asarray(arr).reshape(-1)

    if len(arr) == 0:
        arr = np.zeros(target_len)

    signal_series = pd.Series(arr).fillna(0).reset_index(drop=True)

    if len(signal_series) > target_len:
        signal_series = signal_series.iloc[-target_len:].reset_index(drop=True)
    elif len(signal_series) < target_len:
        pad = pd.Series(np.zeros(target_len - len(signal_series)))
        signal_series = pd.concat([pad, signal_series], ignore_index=True)

    return signal_series.astype(float)


def _prepare_price_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or len(prices) == 0:
        fallback = pd.DataFrame({"close": np.linspace(100, 101, 50)})
        fallback.index = pd.RangeIndex(start=0, stop=len(fallback))
        return fallback

    data = prices.copy()

    if isinstance(data, pd.Series):
        data = data.to_frame(name=str(data.name or "close"))

    if "close" in {str(c).lower() for c in data.columns} and len(data.columns) == 1:
        close_col = next(c for c in data.columns if str(c).lower() == "close")
        data = data[[close_col]].rename(columns={close_col: "close"})

    data.index = pd.to_datetime(data.index, errors="coerce")
    if data.index.isna().any():
        data.index = pd.RangeIndex(start=0, stop=len(data))

    data = data.sort_index()
    data.columns = [str(c) for c in data.columns]
    data = data.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    data = data.ffill().bfill()

    if data.empty:
        return _prepare_price_matrix(pd.DataFrame())

    return data


def _coerce_weight_matrix(target_weights, index: pd.Index, columns: list[str]) -> pd.DataFrame:
    if target_weights is None:
        return pd.DataFrame(0.0, index=index, columns=columns)

    if isinstance(target_weights, pd.Series):
        if len(columns) == 1:
            matrix = pd.DataFrame(target_weights, index=index, columns=[columns[0]])
        else:
            repeated = _normalize_signals(target_weights, len(index))
            matrix = pd.DataFrame(
                np.repeat(repeated.to_numpy().reshape(-1, 1), len(columns), axis=1) / max(len(columns), 1),
                index=index,
                columns=columns,
            )
    elif isinstance(target_weights, pd.DataFrame):
        matrix = target_weights.copy()
        if "date" in matrix.columns and "asset" in matrix.columns and "weight" in matrix.columns:
            matrix["date"] = pd.to_datetime(matrix["date"], errors="coerce")
            matrix["asset"] = matrix["asset"].astype(str)
            matrix["weight"] = pd.to_numeric(matrix["weight"], errors="coerce").fillna(0.0)
            matrix = matrix.pivot_table(
                index="date",
                columns="asset",
                values="weight",
                aggfunc="sum",
                fill_value=0.0,
            )
    else:
        normalized = _normalize_signals(target_weights, len(index))
        base_column = columns[0] if columns else "close"
        matrix = pd.DataFrame({base_column: normalized.to_numpy()}, index=index)

    matrix = matrix.copy()
    matrix.index = pd.to_datetime(matrix.index, errors="coerce")
    if matrix.index.isna().any():
        matrix.index = index

    matrix = matrix.reindex(index=index, columns=columns).fillna(0.0)
    matrix = matrix.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return matrix


def backtest(df: pd.DataFrame, signals):
    prices = _ensure_close(df).copy().reset_index(drop=True)
    signal_series = _normalize_signals(signals, len(prices))

    prices["signal"] = signal_series
    prices["return"] = prices["close"].pct_change().fillna(0.0)
    prices["strategy"] = prices["signal"].shift(1).fillna(0.0) * prices["return"]
    prices["equity"] = (1.0 + prices["strategy"]).cumprod()

    total_return = float(prices["equity"].iloc[-1] - 1.0) if len(prices) else 0.0
    trades = int((prices["signal"].diff().abs().fillna(0) > 0).sum()) if len(prices) else 0

    return {
        "total_return": total_return,
        "trades": trades,
    }


def backtest_portfolio(
    prices: pd.DataFrame,
    signals=None,
    target_weights=None,
    initial_capital: float = 100000.0,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    execution_delay: int = 1,
):
    """
    Compativel com o contrato usado em walk_forward.py e com a chamada legada
    `backtest_portfolio(prices, signals)`.
    """
    price_matrix = _prepare_price_matrix(prices)
    weights_input = target_weights if target_weights is not None else signals
    weight_matrix = _coerce_weight_matrix(weights_input, price_matrix.index, list(price_matrix.columns))

    asset_returns = price_matrix.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    delay = max(int(execution_delay), 0)
    effective_weights = weight_matrix.shift(delay).fillna(0.0) if delay > 0 else weight_matrix.copy()

    turnover = effective_weights.diff().abs().sum(axis=1).fillna(effective_weights.abs().sum(axis=1))
    trading_cost = turnover * float(fee_rate + slippage_rate)

    asset_contribution = effective_weights * asset_returns
    portfolio_gross_return = asset_contribution.sum(axis=1)
    portfolio_net_return = (portfolio_gross_return - trading_cost).astype(float)

    base_capital = float(initial_capital) if float(initial_capital) > 0 else 100000.0
    equity = base_capital * (1.0 + portfolio_net_return).cumprod()

    returns_df = pd.DataFrame({"return": portfolio_net_return}, index=price_matrix.index)
    equity_df = pd.DataFrame({"equity": equity}, index=price_matrix.index)
    turnover_df = pd.DataFrame({"turnover": turnover}, index=price_matrix.index)

    total_return = float((equity.iloc[-1] / base_capital) - 1.0) if len(equity) else 0.0
    trades = int((turnover > 1e-9).sum()) if len(turnover) else 0

    metrics = {
        "total_return": total_return,
        "trades": trades,
    }

    return {
        "returns": returns_df,
        "equity_curve": equity_df,
        "turnover": turnover_df,
        "target_weights": weight_matrix,
        "effective_weights": effective_weights,
        "asset_returns": asset_returns,
        "asset_contribution": asset_contribution,
        "trading_cost": pd.DataFrame({"cost": trading_cost}, index=price_matrix.index),
        "metrics": metrics,
    }
