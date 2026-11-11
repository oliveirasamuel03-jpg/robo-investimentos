"""Read-only diagnostics for the existing PAPER ledger contract.

No caller in the execution path uses these findings as a trading signal or guard.
Pass a consistent snapshot. Global reconciliation requires every trade since the
last reset, in persistence order; a limited query is not a complete ledger.
The PAPER engine changes cash by gross_value, charges zero cost, rounds cash and
PnL to cents, and stores one long position per asset. Equity history is bounded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IntegrityFinding:
    code: str
    path: str
    message: str


def diagnose_paper_integrity(
    state: dict[str, Any],
    trades: list[dict[str, Any]],
    *,
    complete_trade_history: bool = False,
) -> list[IntegrityFinding]:
    """Inspect synthetic/exported data without mutating or repairing it.

    Missing legacy IDs are reported, never generated. A repeated trade_id on
    BUY and SELL is valid: it refers to the same position. Repeated (ID, side)
    is a duplicate. Decimal strings are accepted as numeric, booleans are not.
    One cent is allowed for persisted financial rounding comparisons.
    """
    findings: list[IntegrityFinding] = []

    def add(code: str, path: str, message: str) -> None:
        findings.append(IntegrityFinding(code, path, message))

    def number(row: dict, key: str, path: str, *, positive: bool = False,
               nonnegative: bool = False) -> float | None:
        value = row.get(key)
        try:
            parsed = float(value) if not isinstance(value, bool) else math.nan
        except (TypeError, ValueError, OverflowError):
            parsed = math.nan
        if not math.isfinite(parsed):
            add("invalid_number", f"{path}.{key}", "Expected a finite numeric value.")
            return None
        if (positive and parsed <= 0) or (nonnegative and parsed < 0):
            add("invalid_amount", f"{path}.{key}", "Value is outside the PAPER amount contract.")
        return parsed

    def differs(actual: float | None, expected: float | None) -> bool:
        return actual is not None and expected is not None and abs(actual - expected) > 0.010000001

    if not isinstance(state, dict) or not isinstance(trades, list):
        add("invalid_structure", "snapshot", "Expected a state object and a trade list.")
        return findings

    capital = number(state, "initial_capital", "state", nonnegative=True)
    cash = number(state, "cash", "state", nonnegative=True)
    equity = number(state, "equity", "state", nonnegative=True)
    realized = number(state, "realized_pnl", "state")
    positions = state.get("positions")
    if not isinstance(positions, dict):
        add("invalid_structure", "state.positions", "Expected positions keyed by asset.")
        positions = {}
    last_prices = state.get("last_prices", {})
    if not isinstance(last_prices, dict):
        add("invalid_structure", "state.last_prices", "Expected prices keyed by asset.")
        last_prices = {}

    asset_keys: set[str] = set()
    position_ids: set[str] = set()
    position_refs: dict[str, tuple[str, float | None]] = {}
    market_value: float | None = 0.0
    for asset, position in positions.items():
        path = f"state.positions.{asset}"
        asset_key = str(asset).strip().upper()
        if not asset_key or asset_key in asset_keys:
            add("duplicate_position", path, "Empty or repeated normalized asset key.")
        asset_keys.add(asset_key)
        if not isinstance(position, dict):
            add("invalid_structure", path, "Expected a position object.")
            market_value = None
            continue
        trade_id = str(position.get("trade_id") or "").strip()
        if not trade_id:
            add("invalid_reference", f"{path}.trade_id", "Position is missing its trade reference.")
        elif trade_id in position_ids:
            add("duplicate_position", f"{path}.trade_id", "Trade reference belongs to multiple open positions.")
        position_ids.add(trade_id)
        qty = number(position, "quantity", path, positive=True)
        number(position, "avg_price", path, positive=True)
        price = number({"price": last_prices.get(asset, position.get("last_price", 0.0))},
                       "price", path, positive=True)
        if "gross_entry_value" in position:
            number(position, "gross_entry_value", path, positive=True)
        position_refs[trade_id] = (asset_key, qty)
        if market_value is not None and qty is not None and price is not None:
            market_value += qty * price
        else:
            market_value = None
    if cash is not None and market_value is not None and differs(equity, round(cash + market_value, 2)):
        add("equity_mismatch", "state.equity", "Equity differs from cash plus marked open positions.")

    history = state.get("history")
    if not isinstance(history, list):
        add("invalid_structure", "state.history", "Expected bounded equity snapshot history.")
        history = []
    for index, snapshot in enumerate(history):
        path = f"state.history[{index}]"
        if not isinstance(snapshot, dict):
            add("invalid_structure", path, "Expected an equity snapshot object.")
            continue
        h_cash = number(snapshot, "cash", path, nonnegative=True)
        h_equity = number(snapshot, "equity", path, nonnegative=True)
        h_count = number(snapshot, "open_positions", path, nonnegative=True)
        if h_count is not None and not h_count.is_integer():
            add("invalid_position_count", f"{path}.open_positions", "Position count must be an integer.")
        if index == len(history) - 1:
            if differs(h_cash, cash) or differs(h_equity, equity) or h_count != len(positions):
                add("history_mismatch", path, "Last equity snapshot differs from current state.")

    seen: set[tuple[str, str]] = set()
    open_buys: dict[str, dict[str, Any]] = {}
    rolling_cash = capital
    rolling_realized: float | None = 0.0
    for index, trade in enumerate(trades):
        path = f"trades[{index}]"
        if not isinstance(trade, dict):
            add("invalid_structure", path, "Expected a trade object.")
            rolling_cash = rolling_realized = None
            continue
        trade_id = str(trade.get("trade_id") or "").strip()
        asset = str(trade.get("asset") or "").strip().upper()
        side = str(trade.get("side") or "").upper()
        if not trade_id or not asset or side not in {"BUY", "SELL"}:
            add("invalid_reference", path, "Trade requires an ID, asset and BUY/SELL side.")
        key = (trade_id, side)
        if trade_id and key in seen:
            add("duplicate_trade", path, "Repeated trade ID and side.")
        seen.add(key)
        qty = number(trade, "quantity", path, positive=True)
        price = number(trade, "price", path, positive=True)
        gross = number(trade, "gross_value", path, positive=True)
        number(trade, "cost", path, nonnegative=True)
        after = number(trade, "cash_after", path, nonnegative=True)
        pnl = number(trade, "realized_pnl", path)
        if side == "BUY":
            open_buys[trade_id] = {"asset": asset, "quantity": qty, "price": price}
        elif side == "SELL":
            buy = open_buys.pop(trade_id, None)
            if complete_trade_history and buy is None:
                add("invalid_reference", path, "SELL has no preceding unmatched BUY.")
            if buy is not None:
                if asset != buy["asset"] or qty != buy["quantity"]:
                    add("invalid_reference", path, "SELL asset/quantity differs from its BUY.")
                if all(value is not None for value in (pnl, price, buy["price"], qty)):
                    # Prices are persisted to six decimals; allow that rounding per unit.
                    tolerance = 0.01 + abs(qty) * 0.000001
                    if abs(pnl - round((price - buy["price"]) * qty, 2)) > tolerance:
                        add("trade_pnl_mismatch", path, "SELL PnL differs from entry/exit prices.")
        if complete_trade_history:
            if rolling_cash is not None and gross is not None and side in {"BUY", "SELL"}:
                rolling_cash = round(rolling_cash + (gross if side == "SELL" else -gross), 2)
                if differs(after, rolling_cash):
                    add("cash_update_mismatch", f"{path}.cash_after", "Cash does not follow PAPER gross-value updates.")
            else:
                rolling_cash = None
            if rolling_realized is not None and pnl is not None:
                rolling_realized = round(rolling_realized + pnl, 2)
            else:
                rolling_realized = None
    if complete_trade_history:
        if differs(cash, rolling_cash):
            add("ledger_cash_mismatch", "state.cash", "Current cash differs from the complete trade ledger.")
        if differs(realized, rolling_realized):
            add("ledger_pnl_mismatch", "state.realized_pnl", "Current realized PnL differs from the complete trade ledger.")
        expected_refs = {key: (value["asset"], value["quantity"]) for key, value in open_buys.items()}
        if position_refs != expected_refs:
            add("ledger_positions_mismatch", "state.positions", "Open positions differ from unmatched BUY records.")
    return findings
