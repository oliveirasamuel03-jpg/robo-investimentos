from __future__ import annotations

from core.config import (
    MAX_HOLDING_MINUTES,
    MAX_TICKET,
    MIN_HOLDING_MINUTES,
    MIN_TICKET,
    TRADER_ORDERS_COLUMNS,
    TRADER_ORDERS_FILE,
)
from core.state_store import (
    load_bot_state,
    log_event,
    read_storage_table,
    replace_storage_table,
    save_bot_state,
)
from core.trader_profiles import get_trader_profile_config
from engines.quant_bridge import (
    PaperTradingConfig,
    build_paper_report,
    ensure_paper_files,
    load_paper_state,
    read_paper_equity,
    read_paper_trades,
    reset_paper_state,
    run_paper_cycle,
)


def build_paper_cfg_from_platform_state(state: dict) -> PaperTradingConfig:
    trader = state["trader"]
    ticket_value = float(trader["ticket_value"])
    holding_minutes = int(trader["holding_minutes"])
    max_open_positions = int(trader.get("max_open_positions", 3))

    if ticket_value < MIN_TICKET or ticket_value > MAX_TICKET:
        raise ValueError("O ticket do trader precisa ficar entre R$10 e R$10.000.")
    if holding_minutes < MIN_HOLDING_MINUTES or holding_minutes > MAX_HOLDING_MINUTES:
        raise ValueError("O holding do trader precisa ficar entre 1 minuto e 48 horas.")

    watchlist = [str(x).strip().upper() for x in (trader.get("watchlist", []) or []) if str(x).strip()]
    profile_config = get_trader_profile_config(
        trader.get("profile"),
        base_ticket_value=ticket_value,
        base_holding_minutes=holding_minutes,
        base_max_open_positions=max_open_positions,
    )

    return PaperTradingConfig(
        initial_capital=float(state.get("wallet_value", 10000.0)),
        custom_tickers=watchlist or ["AAPL"],
        profile_name=str(profile_config["name"]),
        ticket_value=float(profile_config["ticket_value"]),
        min_trade_notional=max(MIN_TICKET, float(profile_config["ticket_value"])),
        max_open_positions=max(1, int(profile_config["max_open_positions"])),
        holding_minutes=int(profile_config["holding_minutes"]),
        history_limit=500,
        allow_new_entries=state.get("bot_status") == "RUNNING",
        min_signal_score=float(profile_config["min_signal_score"]),
        min_atr_pct=float(profile_config["min_atr_pct"]),
        max_atr_pct=float(profile_config["max_atr_pct"]),
        trailing_stop_atr_mult=float(profile_config["trailing_stop_atr_mult"]),
        rsi_entry_min=float(profile_config["rsi_entry_min"]),
        rsi_entry_max=float(profile_config["rsi_entry_max"]),
        pullback_min=float(profile_config["pullback_min"]),
        pullback_max=float(profile_config["pullback_max"]),
        momentum_min=float(profile_config["momentum_min"]),
        momentum_8_min=float(profile_config["momentum_8_min"]),
        breakout_min=float(profile_config["breakout_min"]),
        ma20_slope_min=float(profile_config["ma20_slope_min"]),
        ma50_slope_min=float(profile_config["ma50_slope_min"]),
    )


def sync_platform_positions_from_paper() -> dict:
    platform_state = load_bot_state()
    paper_state = load_paper_state()
    last_prices = paper_state.get("last_prices", {}) or {}

    positions = []
    for asset, pos in (paper_state.get("positions", {}) or {}).items():
        qty = float(pos.get("quantity", 0.0))
        last_price = float(last_prices.get(asset, pos.get("last_price", 0.0)))
        avg_price = float(pos.get("avg_price", last_price))
        positions.append(
            {
                "module": "TRADER",
                "asset": asset,
                "profile": pos.get("profile"),
                "status": "OPEN",
                "qty": qty,
                "entry_price": avg_price,
                "last_price": last_price,
                "value": round(qty * last_price, 2),
                "unrealized_pnl": round((last_price - avg_price) * qty, 2),
            }
        )

    platform_state["positions"] = [
        position for position in platform_state.get("positions", []) if position.get("module") != "TRADER"
    ] + positions
    platform_state["cash"] = float(paper_state.get("cash", platform_state.get("cash", 0.0)))
    platform_state["realized_pnl"] = float(
        paper_state.get("realized_pnl", platform_state.get("realized_pnl", 0.0))
    )
    save_bot_state(platform_state)
    return platform_state


def _sync_trader_orders_into_storage() -> None:
    trades = read_paper_trades(limit=200)
    if not trades:
        return

    df_existing = read_storage_table(TRADER_ORDERS_FILE, columns=TRADER_ORDERS_COLUMNS)

    existing_keys = set(
        zip(
            df_existing.get("timestamp", []),
            df_existing.get("asset", []),
            df_existing.get("side", []),
        )
    )

    rows_to_store = df_existing.to_dict(orient="records")

    for trade in trades:
        key = (trade.get("timestamp"), trade.get("asset"), trade.get("side"))
        if key in existing_keys:
            continue

        rows_to_store.append(
            {
                "timestamp": trade.get("timestamp"),
                "asset": trade.get("asset"),
                "side": trade.get("side"),
                "quantity": trade.get("quantity", 0.0),
                "price": trade.get("price", 0.0),
                "gross_value": trade.get("gross_value", 0.0),
                "cost": trade.get("cost", 0.0),
                "cash_after": trade.get("cash_after", 0.0),
                "source": "paper_trading_engine",
            }
        )
        existing_keys.add(key)

    replace_storage_table(TRADER_ORDERS_FILE, rows_to_store, columns=TRADER_ORDERS_COLUMNS)


def run_trader_cycle() -> dict:
    state = load_bot_state()

    if state.get("bot_status") == "STOPPED":
        log_event("INFO", "Trader parado pelo status STOPPED")
        return {"status": "STOPPED"}

    if not state["trader"].get("enabled", True):
        log_event("INFO", "Trader desabilitado")
        return {"status": "DISABLED"}

    cfg = build_paper_cfg_from_platform_state(state)
    ensure_paper_files(cfg)

    cycle_result = run_paper_cycle(cfg)
    _sync_trader_orders_into_storage()
    platform_state = sync_platform_positions_from_paper()
    report = build_paper_report(initial_capital=float(platform_state.get("wallet_value", cfg.initial_capital)))
    equity_df = read_paper_equity(limit=200)

    return {
        "status": state.get("bot_status"),
        "cycle_result": cycle_result,
        "paper_report": report,
        "paper_equity": equity_df.to_dict(orient="records") if not equity_df.empty else [],
    }


def reset_trader_module() -> dict:
    state = load_bot_state()
    cfg = build_paper_cfg_from_platform_state(state)
    reset_paper_state(cfg)
    platform_state = sync_platform_positions_from_paper()
    log_event("INFO", "Trader resetado")
    return platform_state
