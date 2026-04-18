from __future__ import annotations

import pandas as pd

from core.config import (
    MAX_HOLDING_MINUTES,
    MAX_TICKET,
    MIN_HOLDING_MINUTES,
    MIN_TICKET,
    TRADER_ORDERS_FILE,
)
from core.state_store import append_csv_row, load_bot_state, log_event, save_bot_state
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

    if ticket_value < MIN_TICKET or ticket_value > MAX_TICKET:
        raise ValueError("O ticket do trader precisa ficar entre R$10 e R$10.000.")
    if holding_minutes < MIN_HOLDING_MINUTES or holding_minutes > MAX_HOLDING_MINUTES:
        raise ValueError("O holding do trader precisa ficar entre 1 minuto e 48 horas.")

    watchlist = trader.get("watchlist", []) or []
    watchlist = [str(x).strip().upper() for x in watchlist if str(x).strip()]

    # Trader precisa ser leve: usa só watchlist + menos histórico
    return PaperTradingConfig(
        start_date=state.get("start_date", "2023-01-01"),
        initial_capital=float(state["wallet_value"]),
        probability_threshold=float(state.get("probability_threshold", 0.60)),
        top_n=min(int(state.get("top_n", 2)), max(1, len(watchlist) or 2)),
        use_regime_filter=bool(state.get("use_regime_filter", True)),
        use_volatility_filter=bool(state.get("use_volatility_filter", False)),
        vol_threshold=float(state.get("vol_threshold", 0.03)),
        cash_threshold=float(state.get("cash_threshold", 0.50)),
        target_portfolio_vol=float(state.get("target_portfolio_vol", 0.08)),
        include_brazil_stocks=False,
        include_us_stocks=False,
        include_etfs=False,
        include_fiis=False,
        include_crypto=False,
        include_grains=False,
        custom_tickers=watchlist,
        min_trade_notional=max(10.0, ticket_value),
        history_limit=500,
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
                "status": "OPEN",
                "qty": qty,
                "entry_price": avg_price,
                "last_price": last_price,
                "value": round(qty * last_price, 2),
                "unrealized_pnl": round((last_price - avg_price) * qty, 2),
            }
        )

    platform_state["positions"] = [
        p for p in platform_state.get("positions", []) if p.get("module") != "TRADER"
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

    df_existing = pd.read_csv(TRADER_ORDERS_FILE)
    existing_cols = set(df_existing.columns.tolist())

    needed_cols = [
        "timestamp",
        "asset",
        "side",
        "quantity",
        "price",
        "gross_value",
        "cost",
        "cash_after",
        "source",
    ]
    if not set(needed_cols).issubset(existing_cols):
        df_existing = pd.DataFrame(columns=needed_cols)
        df_existing.to_csv(TRADER_ORDERS_FILE, index=False)

    df_existing = pd.read_csv(TRADER_ORDERS_FILE)
    existing_keys = set(
        zip(
            df_existing.get("timestamp", []),
            df_existing.get("asset", []),
            df_existing.get("side", []),
        )
    )

    for row in trades:
        key = (row.get("timestamp"), row.get("asset"), row.get("side"))
        if key in existing_keys:
            continue

        append_csv_row(
            TRADER_ORDERS_FILE,
            {
                "timestamp": row.get("timestamp"),
                "asset": row.get("asset"),
                "side": row.get("side"),
                "quantity": row.get("quantity", 0.0),
                "price": row.get("price", 0.0),
                "gross_value": row.get("gross_value", 0.0),
                "cost": row.get("cost", 0.0),
                "cash_after": row.get("cash_after", 0.0),
                "source": "paper_trading_engine",
            },
        )


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

    result = run_paper_cycle(cfg)
    _sync_trader_orders_into_storage()
    platform_state = sync_platform_positions_from_paper()
    report = build_paper_report(initial_capital=float(platform_state["wallet_value"]))
    equity_df = read_paper_equity(limit=200)

    return {
        "status": state.get("bot_status"),
        "cycle_result": result,
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
