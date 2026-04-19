from __future__ import annotations

import pandas as pd

from metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    expectancy,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


def _to_equity_frame(equity_input) -> pd.DataFrame:
    if equity_input is None:
        return pd.DataFrame(columns=["timestamp", "equity"])

    if isinstance(equity_input, pd.Series):
        df = equity_input.to_frame(name=str(equity_input.name or "equity"))
    elif isinstance(equity_input, pd.DataFrame):
        df = equity_input.copy()
    else:
        df = pd.DataFrame({"equity": equity_input})

    if "equity" not in df.columns:
        first_col = df.columns[0] if len(df.columns) else "equity"
        df = df.rename(columns={first_col: "equity"})

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        ts = pd.to_datetime(df.index, errors="coerce")

    if ts.isna().all():
        ts = pd.RangeIndex(start=0, stop=len(df))

    equity = pd.to_numeric(df["equity"], errors="coerce")
    out = pd.DataFrame({"timestamp": ts, "equity": equity}).dropna(subset=["equity"]).reset_index(drop=True)
    return out


def build_equity_curve_df(equity_input) -> pd.DataFrame:
    curve = _to_equity_frame(equity_input)
    if curve.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "returns"])

    curve["returns"] = curve["equity"].pct_change().fillna(0.0)
    return curve


def build_drawdown_series(equity_input) -> pd.Series:
    curve = build_equity_curve_df(equity_input)
    if curve.empty:
        return pd.Series(dtype=float, name="drawdown")

    peak = curve["equity"].cummax()
    drawdown = curve["equity"] / peak - 1.0
    drawdown.name = "drawdown"
    return drawdown


def summarize_performance(equity_input, returns_input=None, trades_df: pd.DataFrame | None = None) -> dict:
    curve = build_equity_curve_df(equity_input)
    if returns_input is None:
        returns = pd.to_numeric(curve.get("returns", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    elif isinstance(returns_input, pd.DataFrame):
        if "return" in returns_input.columns:
            returns = pd.to_numeric(returns_input["return"], errors="coerce").fillna(0.0)
        else:
            returns = pd.to_numeric(returns_input.iloc[:, 0], errors="coerce").fillna(0.0)
    else:
        returns = pd.to_numeric(pd.Series(returns_input), errors="coerce").fillna(0.0)

    trades_count = 0
    trade_returns = pd.Series(dtype=float)
    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
        trades_count = int(len(trades_df))
        if "return" in trades_df.columns:
            trade_returns = pd.to_numeric(trades_df["return"], errors="coerce").dropna()
        elif "realized_pnl" in trades_df.columns and "gross_value" in trades_df.columns:
            base = pd.to_numeric(trades_df["gross_value"], errors="coerce").replace(0.0, pd.NA)
            pnl = pd.to_numeric(trades_df["realized_pnl"], errors="coerce")
            trade_returns = (pnl / base).dropna()

    stats_source = trade_returns if not trade_returns.empty else returns
    total_return = float(curve["equity"].iloc[-1] / curve["equity"].iloc[0] - 1.0) if len(curve) > 1 else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return(returns),
        "volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(curve["equity"]) if not curve.empty else 0.0,
        "calmar": calmar_ratio(returns, equity=curve["equity"] if not curve.empty else None),
        "win_rate": win_rate(stats_source),
        "profit_factor": profit_factor(stats_source),
        "expectancy": expectancy(stats_source),
        "trades_count": trades_count,
    }
