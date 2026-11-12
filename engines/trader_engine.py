from __future__ import annotations

from core.config import (
    MAX_HOLDING_MINUTES,
    MAX_TICKET,
    MIN_HOLDING_MINUTES,
    MIN_TICKET,
    SWING_VALIDATION_RECOMMENDED_WATCHLIST,
    VALIDATION_INITIAL_CAPITAL_BRL,
    TRADER_ORDERS_COLUMNS,
    TRADER_ORDERS_FILE,
)
from core.persistence import (transactional_persistence, PersistenceConflictError,
                              database_enabled, load_json_rows, append_table_row)
from datetime import datetime, timezone
from uuid import uuid4
from copy import deepcopy
from core.broker import probe_broker_status
from core.daily_risk import evaluate_daily_loss_guard
from core.macro_alerts import default_macro_alert_state, evaluate_configured_macro_alert
from core.state_store import (
    load_bot_state,
    log_event,
    read_storage_table,
    replace_storage_table,
    save_bot_state,
    update_bot_state,
    update_broker_status,
    update_macro_alert_status,
    update_market_context_status,
    update_market_data_status,
    update_risk_status,
)
from core.swing_validation import SWING_VALIDATION_MODE, apply_swing_validation_overrides
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


def build_paper_cfg_from_platform_state(state: dict, risk_state: dict | None = None) -> PaperTradingConfig:
    trader = state["trader"]
    risk = risk_state or (state.get("risk", {}) or {})
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
    validation_state = state.get("validation", {}) or {}
    validation_mode = str(validation_state.get("validation_mode") or "").strip().lower()
    if validation_mode == SWING_VALIDATION_MODE:
        profile_config = apply_swing_validation_overrides(str(profile_config["name"]), profile_config)

    return PaperTradingConfig(
        initial_capital=float(state.get("wallet_value", VALIDATION_INITIAL_CAPITAL_BRL)),
        custom_tickers=watchlist or list(SWING_VALIDATION_RECOMMENDED_WATCHLIST),
        period=str(profile_config.get("period", "6mo")),
        interval=str(profile_config.get("interval", "1h")),
        profile_name=str(profile_config["name"]),
        ticket_value=float(profile_config["ticket_value"]),
        min_trade_notional=max(MIN_TICKET, float(profile_config["ticket_value"])),
        max_open_positions=max(1, int(profile_config["max_open_positions"])),
        holding_minutes=int(profile_config["holding_minutes"]),
        history_limit=int(profile_config.get("history_limit", 500)),
        allow_new_entries=state.get("bot_status") == "RUNNING" and not bool(risk.get("daily_loss_block_active", False)),
        daily_loss_limit_brl=float(risk.get("daily_loss_limit_brl", 0.0) or 0.0),
        daily_loss_day_key=str(risk.get("daily_loss_day_key") or ""),
        daily_realized_pnl_brl=float(risk.get("daily_realized_pnl_brl", 0.0) or 0.0),
        daily_loss_block_active=bool(risk.get("daily_loss_block_active", False)),
        daily_loss_block_reason=str(risk.get("daily_loss_block_reason") or ""),
        macro_alert=dict(state.get("macro_alert", {}) or default_macro_alert_state()),
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
        reentry_cooldown_minutes=int(profile_config["reentry_cooldown_minutes"]),
        min_position_age_minutes=int(profile_config["min_position_age_minutes"]),
        take_profit_rsi=float(profile_config["take_profit_rsi"]),
        weak_momentum_exit_threshold=float(profile_config["weak_momentum_exit_threshold"]),
        trend_break_buffer_pct=float(profile_config["trend_break_buffer_pct"]),
        hard_stop_loss_pct=float(profile_config["hard_stop_loss_pct"]),
    )


def _refresh_daily_loss_guard(state: dict) -> tuple[dict, dict]:
    previous_risk = dict(state.get("risk", {}) or {})
    paper_trades = read_paper_trades(limit=5000)
    risk_state = evaluate_daily_loss_guard(previous_risk, paper_trades)
    persisted = update_risk_status(risk_state)

    prev_block = bool(previous_risk.get("daily_loss_block_active", False))
    now_block = bool(persisted.get("daily_loss_block_active", False))
    transition = str(persisted.get("last_transition") or "none")
    if not prev_block and now_block:
        log_event("WARNING", str(persisted.get("daily_loss_block_reason") or "Limite de perda diaria atingido."))
    elif transition == "reset_day":
        log_event("INFO", f"Trava diaria resetada na virada do dia UTC {persisted.get('daily_loss_day_key')}.")
    elif prev_block and not now_block:
        log_event("INFO", "Trava diaria liberada. Novas entradas permitidas novamente.")

    return previous_risk, persisted


def refresh_daily_loss_guard(state: dict | None = None) -> dict:
    base_state = state or load_bot_state()
    _previous, persisted = _refresh_daily_loss_guard(base_state)
    return persisted


def refresh_macro_alert_state() -> dict:
    try:
        macro_alert = evaluate_configured_macro_alert()
    except Exception as exc:
        macro_alert = default_macro_alert_state(reason=f"Avaliacao macro falhou com seguranca: {exc}")
        log_event("WARNING", f"[macro_alert_summary] evaluation_failed;error={exc}")
    return update_macro_alert_status(macro_alert)


@transactional_persistence
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


@transactional_persistence
def _sync_trader_orders_into_storage() -> None:
    trades = read_paper_trades(limit=200)
    if not trades:
        return

    if database_enabled():
        # Validate every existing JSON row, retain the same identity contract,
        # then append only absent orders in this SAME financial transaction.
        # Existing seq, timestamps, payload and metadata must never be rewritten.
        existing = load_json_rows('table:trader_orders.csv', TRADER_ORDERS_FILE.with_suffix('.csv.json'))
        existing_keys = {(row.get('timestamp'), row.get('asset'), row.get('side')) for row in existing}
        rows_to_store = []
    else:
        df_existing = read_storage_table(TRADER_ORDERS_FILE, columns=TRADER_ORDERS_COLUMNS)
        existing_keys = set(zip(df_existing.get("timestamp", []), df_existing.get("asset", []),
                                df_existing.get("side", [])))
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

    if database_enabled():
        for row in rows_to_store:
            append_table_row(TRADER_ORDERS_FILE, row, columns=TRADER_ORDERS_COLUMNS)
    else:
        replace_storage_table(TRADER_ORDERS_FILE, rows_to_store, columns=TRADER_ORDERS_COLUMNS)


def _operational_fingerprint(state):
    # Heartbeat/chart/diagnostic writes do not invalidate an acquisition. Trading
    # settings, risk, control intent, current platform positions and capital do.
    return deepcopy({key: state.get(key) for key in (
        "bot_status", "bot_mode", "trader", "security", "risk", "wallet_value",
        "positions", "cash", "realized_pnl", "stop_control", "macro_alert",
    )})


def _cycle_phase(cycle_id, phase):
    now = datetime.now(timezone.utc).isoformat()
    def update(state):
        previous = state.get("cycle_activity", {})
        # Completion of an old attempt cannot hide a newer attempt's activity.
        if phase != "PREPARING" and previous.get("cycle_id") != cycle_id:
            return state
        state["cycle_activity"] = {
            "cycle_id": cycle_id, "phase": phase,
            "started_at": now if phase == "PREPARING" else previous.get("started_at", ""),
            "finished_at": "" if phase == "PREPARING" else now,
        }
        return state
    update_bot_state(update)


def run_trader_cycle(*, persist_market_data: bool = True) -> dict:
    from core.stop_control import acknowledge_stop
    state = load_bot_state()
    if state.get("bot_status") == "STOPPED":
        acknowledge_stop(worker_id="trader_cycle")
        return {"status": "STOPPED"}
    if not state["trader"].get("enabled", True):
        return {"status": "DISABLED"}

    from core.commit_protocol import assert_execution_allowed, post_commit_failure, reconcile
    assert_execution_allowed()
    cycle_result = None
    cycle_id = uuid4().hex
    _cycle_phase(cycle_id, "PREPARING")
    try:
        risk_state = refresh_daily_loss_guard(load_bot_state())
        broker_status = probe_broker_status(load_bot_state().get("security", {}), requested_by="worker_cycle")
        update_broker_status(broker_status)
        refresh_macro_alert_state()
        state = load_bot_state()
        if state.get("bot_status") == "STOPPED":
            acknowledge_stop(worker_id="trader_cycle")
            _cycle_phase(cycle_id, "IDLE")
            return {"status": "STOPPED"}
        cfg = build_paper_cfg_from_platform_state(state, risk_state=risk_state)
        expected = _operational_fingerprint(state)

        def validate(latest):
            if _operational_fingerprint(latest) != expected:
                raise PersistenceConflictError("Operational state changed during acquisition; reload and review.")

        def finalize():
            # This callback is inside the SAME transaction as the PAPER ledger.
            # It performs persistence and local calculations only, never feeds.
            _sync_trader_orders_into_storage()
            sync_platform_positions_from_paper()
            refresh_daily_loss_guard(load_bot_state())
            _cycle_phase(cycle_id, "IDLE")

        cycle_result = run_paper_cycle(cfg, commit_guard=validate, commit_finalize=finalize)
        if cycle_result.get("status") == "STOPPED":
            _cycle_phase(cycle_id, "IDLE")
            return {"status": "STOPPED", "committed": False, "cycle_result": cycle_result}
        if cycle_result.get('commit_outcome') == 'POST_COMMIT_DIAGNOSTIC_FAILURE':
            return {**cycle_result, 'status': state.get('bot_status')}
        if persist_market_data:
            update_market_data_status(cycle_result.get("market_data_status"))
        update_market_context_status(cycle_result.get("market_context"))
        report = build_paper_report()
        equity_df = read_paper_equity(limit=200)
        current_status = load_bot_state().get("bot_status")
        current_risk = load_bot_state().get("risk", {})
        resolution = reconcile(cycle_result['cycle_id'])
        if resolution['outcome'] == 'UNKNOWN_COMMIT' or not resolution.get('resolution_recorded', False):
            failure = post_commit_failure(cycle_result['cycle_id'], confirmed=True)
            return {**failure, 'commit_outcome': failure['outcome'], 'status': current_status}
        return {
            "commit_outcome": cycle_result['commit_outcome'], "cycle_id": cycle_result['cycle_id'],
            "status": current_status, "committed": True,
            "cycle_result": cycle_result, "paper_report": report,
            "paper_equity": equity_df.to_dict(orient="records") if not equity_df.empty else [],
            "broker_status": broker_status, "risk": current_risk,
        }
    except Exception:
        if cycle_result and cycle_result.get('committed'):
            failure = post_commit_failure(cycle_result['cycle_id'], confirmed=True)
            return {**failure, 'commit_outcome': failure['outcome'], 'status': state.get('bot_status')}
        # Best-effort diagnostic completion only; never retry the financial unit.
        # A database outage may leave PREPARING as last observed, not proof of life.
        try:
            _cycle_phase(cycle_id, "FAILED")
        except Exception:
            pass
        raise


@transactional_persistence
def reset_trader_module() -> dict:
    from core.commit_protocol import assert_execution_allowed
    assert_execution_allowed()
    state = load_bot_state()
    cfg = build_paper_cfg_from_platform_state(state)
    reset_paper_state(cfg)
    platform_state = sync_platform_positions_from_paper()
    log_event("INFO", "Trader resetado")
    return platform_state
