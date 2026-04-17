import numpy as np
import pandas as pd


def backtest_portfolio(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float = 100000.0,
    fee_rate: float = 0.0005,
    slippage_rate: float = 0.0005,
    execution_delay: int = 1,
    max_gross_exposure: float = 1.0,
) -> dict:
    if prices.empty:
        raise ValueError("prices is empty")

    if target_weights.empty:
        raise ValueError("target_weights is empty")

    if execution_delay < 0:
        raise ValueError("execution_delay must be >= 0")

    prices = prices.sort_index().copy()
    target_weights = target_weights.sort_index().copy()

    common_index = prices.index.intersection(target_weights.index)
    common_cols = prices.columns.intersection(target_weights.columns)

    if len(common_index) == 0 or len(common_cols) == 0:
        raise ValueError("prices and target_weights have no overlapping index/columns")

    prices = prices.loc[common_index, common_cols].ffill().dropna(how="all")
    target_weights = target_weights.loc[common_index, common_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    asset_returns = prices.pct_change().fillna(0.0)
    executed_weights = target_weights.shift(execution_delay).fillna(0.0)

    current_weights = pd.Series(0.0, index=common_cols, dtype=float)
    nav = float(initial_capital)
    running_peak = float(initial_capital)

    equity_records = []
    return_records = []
    portfolio_records = []
    trade_records = []

    for dt in prices.index:
        desired_weights = executed_weights.loc[dt].astype(float).copy()

        gross_exposure = desired_weights.abs().sum()
        if gross_exposure > max_gross_exposure and gross_exposure > 0:
            desired_weights = desired_weights / gross_exposure * max_gross_exposure

        turnover = float((desired_weights - current_weights).abs().sum())
        trading_cost = turnover * (fee_rate + slippage_rate)

        day_asset_returns = asset_returns.loc[dt].astype(float)
        gross_portfolio_return = float((current_weights * day_asset_returns).sum())
        net_portfolio_return = gross_portfolio_return - trading_cost

        nav *= (1.0 + net_portfolio_return)
        running_peak = max(running_peak, nav)
        drawdown = (nav / running_peak) - 1.0

        changed_assets = desired_weights[desired_weights.ne(current_weights)]
        for asset in changed_assets.index:
            old_w = float(current_weights[asset])
            new_w = float(desired_weights[asset])
            delta_w = new_w - old_w

            if abs(delta_w) > 1e-12:
                side = "BUY" if delta_w > 0 else "SELL"
                trade_records.append(
                    {
                        "date": dt,
                        "asset": asset,
                        "side": side,
                        "price": float(prices.loc[dt, asset]),
                        "old_weight": old_w,
                        "new_weight": new_w,
                        "delta_weight": delta_w,
                        "estimated_cost": abs(delta_w) * (fee_rate + slippage_rate),
                        "turnover_total": turnover,
                    }
                )

        current_weights = desired_weights.copy()

        equity_records.append(
            {
                "date": dt,
                "equity": nav,
                "gross_return": gross_portfolio_return,
                "net_return": net_portfolio_return,
                "turnover": turnover,
                "cost": trading_cost,
                "drawdown": drawdown,
            }
        )

        row = {"date": dt}
        for asset in common_cols:
            row[f"w_{asset}"] = float(current_weights[asset])
        portfolio_records.append(row)

        return_records.append(
            {
                "date": dt,
                "return": net_portfolio_return,
            }
        )

    equity_curve = pd.DataFrame(equity_records).set_index("date")
    portfolio_history = pd.DataFrame(portfolio_records).set_index("date")
    returns_df = pd.DataFrame(return_records).set_index("date")
    trade_log = pd.DataFrame(trade_records)

    return {
        "equity_curve": equity_curve,
        "returns": returns_df,
        "drawdown": equity_curve["drawdown"],
        "trade_log": trade_log,
        "portfolio_history": portfolio_history,
    }
