from __future__ import annotations


def compute_wallet_snapshot(state: dict) -> dict:
    positions = state.get("positions", [])
    capital_in_use = sum(float(p.get("value", 0.0)) for p in positions if p.get("status") == "OPEN")
    return {
        "wallet_value": float(state.get("wallet_value", 0.0)),
        "cash": float(state.get("cash", 0.0)),
        "capital_in_use": capital_in_use,
        "realized_pnl": float(state.get("realized_pnl", 0.0)),
        "open_positions": sum(1 for p in positions if p.get("status") == "OPEN"),
    }


def reserve_cash_amount(state: dict) -> float:
    wallet = float(state.get("wallet_value", 0.0))
    reserve_pct = float(state.get("reserve_cash_pct", 0.10))
    return wallet * reserve_pct


def available_for_new_trade(state: dict) -> float:
    reserve_cash = reserve_cash_amount(state)
    cash = float(state.get("cash", 0.0))
    return max(cash - reserve_cash, 0.0)
