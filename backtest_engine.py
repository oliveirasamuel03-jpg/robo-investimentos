import numpy as np
import pandas as pd


def backtest_portfolio(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float = 100000.0,
    fee_rate: float = 0.0005,
    slippage_rate: float = 0.0005,
    execution_delay: int = 1,
) -> dict:
    """
    Realistic portfolio backtest with:
    - multi-asset weights
    - 1-bar execution delay
    - transaction costs
    - slippage
    - portfolio history
    - trade log

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices indexed by date, columns = assets.
    target_weights : pd.DataFrame
        Desired target weights by date, same index/columns as prices.
        Example: 0.25, 0.10, 0.0, etc.
    initial_capital : float
        Starting NAV.
    fee_rate : float
        Transaction fee as fraction of traded notional.
    slippage_rate : float
        Slippage as fraction of traded notional.
    execution_delay : int
        Number of bars of delay before signal execution.

    Returns
    -------
    dict with:
        equity_curve
        returns
        drawdown
        trade_log
        portfolio_history
    """
    prices = prices.sort_index().copy()
    target_weights = target_weights.sort_index().copy()

    common_index = prices.index.intersection(target_weights.index)
    common_cols = prices.columns.intersection(target_weights.columns)

    prices = prices.loc[common_index, common_cols].ffill().dropna(how="all")
    target_weights = target_weights.loc[common_index, common_cols].fillna(0.0)

    asset_returns = prices.pct_change().fillna(0.0)

    executed_weights = target_weights.shift(execution_delay).fillna(0.0)

    dates = prices.index
    current_weights = pd.Series(0.0, index=common_cols)
    nav = initial_capital

    equity_records = []
    return_records = []
    portfolio_records = []
    trade_records = []

    running_peak = initial_capital

    for dt in dates:
        desired_weights = executed_weights.loc[dt].copy()

        gross_exposure = desired_weights.abs().sum()
        if gross_exposure > 1.0 and gross_exposure > 0:
            desired_weights = desired_weights / gross_exposure

        turnover = (desired_weights - current_weights).abs().sum()

        trading_cost = turnover * (fee_rate + slippage_rate)

        day_asset_returns = asset_returns.loc[dt]
        gross_portfolio_return = float((current_weights * day_asset_returns).sum())
        net_portfolio_return = gross_portfolio_return - trading_cost

        nav = nav * (1.0 + net_portfolio_return)
        running_peak = max(running_peak, nav)
        drawdown = (nav / running_peak) - 1.0

        changed_assets = desired_weights[desired_weights.ne(current_weights)]
        for asset in changed_assets.index:
            old_w = float(current_weights[asset])
            new_w = float(desired_weights[asset])
            delta_w = new_w - old_w

            if abs(delta_w) > 1e-12:
                trade_records.append(
                    {
                        "date": dt,
                        "asset": asset,
                        "old_weight": old_w,
                        "new_weight": new_w,
                        "delta_weight": delta_w,
                        "estimated_cost": abs(delta_w) * (fee_rate + slippage_rate),
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
