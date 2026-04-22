from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.config import (
    LEGACY_VALIDATION_INITIAL_CAPITAL_BRL,
    MARKET_DATA_HISTORY_LIMIT,
    RUNTIME_DIR,
    SWING_VALIDATION_RECOMMENDED_WATCHLIST,
    VALIDATION_DEFAULT_MAX_OPEN_POSITIONS,
    VALIDATION_INITIAL_CAPITAL_BRL,
    ensure_app_directories,
)
from core.market_context import apply_context_filter, get_market_context
from core.market_data import (
    fallback_data as build_fallback_market_data,
    fetch_market_data_frame,
    fetch_market_data_map,
    frame_data_source as get_frame_data_source,
    is_live_market_source,
)
from core.persistence import (
    append_json_row,
    database_enabled,
    load_json_rows,
    load_json_state,
    replace_json_rows,
    save_json_state,
)
from core.trader_reports import append_trade_report, build_closed_trade_report


PAPER_STATE_FILE = RUNTIME_DIR / "paper_state.json"
PAPER_TRADES_FILE = RUNTIME_DIR / "paper_trades.json"
PAPER_STATE_NAMESPACE = "paper_state"
PAPER_TRADES_NAMESPACE = "paper_trades"


@dataclass
class PaperTradingConfig:
    initial_capital: float = VALIDATION_INITIAL_CAPITAL_BRL
    custom_tickers: list[str] = field(default_factory=lambda: list(SWING_VALIDATION_RECOMMENDED_WATCHLIST))
    period: str = "6mo"
    interval: str = "1h"
    profile_name: str = "Equilibrado"
    ticket_value: float = 100.0
    min_trade_notional: float = 100.0
    max_open_positions: int = VALIDATION_DEFAULT_MAX_OPEN_POSITIONS
    holding_minutes: int = 60
    history_limit: int = MARKET_DATA_HISTORY_LIMIT
    allow_new_entries: bool = True
    daily_loss_limit_brl: float = 0.0
    daily_loss_day_key: str = ""
    daily_realized_pnl_brl: float = 0.0
    daily_loss_block_active: bool = False
    daily_loss_block_reason: str = ""
    min_signal_score: float = 0.62
    min_atr_pct: float = 0.003
    max_atr_pct: float = 0.08
    trailing_stop_atr_mult: float = 2.4
    rsi_entry_min: float = 42.0
    rsi_entry_max: float = 64.0
    pullback_min: float = -0.02
    pullback_max: float = 0.035
    momentum_min: float = -0.005
    momentum_8_min: float = -0.02
    breakout_min: float = -0.04
    ma20_slope_min: float = -0.003
    ma50_slope_min: float = -0.004
    reentry_cooldown_minutes: int = 20
    min_position_age_minutes: int = 5
    take_profit_rsi: float = 74.0
    weak_momentum_exit_threshold: float = -0.002
    trend_break_buffer_pct: float = 0.99
    hard_stop_loss_pct: float = 0.04


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _utc_day_key() -> str:
    return _utc_now().strftime("%Y-%m-%d")


def _new_trade_id(asset: str) -> str:
    return f"{str(asset).upper()}-{uuid4().hex[:12]}"


def _default_state(initial_capital: float) -> dict[str, Any]:
    capital = round(float(initial_capital), 2)
    return {
        "initial_capital": capital,
        "cash": capital,
        "equity": capital,
        "realized_pnl": 0.0,
        "positions": {},
        "asset_cooldowns": {},
        "last_prices": {},
        "history": [],
        "run_count": 0,
        "updated_at": "",
        "last_trade_at": "",
    }


def _is_close_number(current: float | int | None, expected: float) -> bool:
    try:
        return abs(float(current or 0.0) - float(expected)) < 1e-9
    except (TypeError, ValueError):
        return False


def _should_migrate_fresh_paper_state(state: dict[str, Any], desired_initial_capital: float) -> bool:
    if _is_close_number(desired_initial_capital, LEGACY_VALIDATION_INITIAL_CAPITAL_BRL):
        return False

    positions = state.get("positions", {}) or {}
    history = state.get("history", []) or []
    if positions or history:
        return False
    if int(state.get("run_count", 0) or 0) != 0:
        return False
    if str(state.get("last_trade_at") or "").strip():
        return False
    if not _is_close_number(state.get("realized_pnl", 0.0), 0.0):
        return False

    return (
        _is_close_number(state.get("initial_capital"), LEGACY_VALIDATION_INITIAL_CAPITAL_BRL)
        and _is_close_number(state.get("cash"), LEGACY_VALIDATION_INITIAL_CAPITAL_BRL)
        and _is_close_number(state.get("equity"), LEGACY_VALIDATION_INITIAL_CAPITAL_BRL)
    )


def ensure_paper_files(config: PaperTradingConfig | None = None) -> None:
    ensure_app_directories()

    initial_capital = float(config.initial_capital) if config else VALIDATION_INITIAL_CAPITAL_BRL

    if database_enabled():
        load_json_state(PAPER_STATE_NAMESPACE, lambda: _default_state(initial_capital), PAPER_STATE_FILE)
        return

    if not PAPER_STATE_FILE.exists():
        save_paper_state(_default_state(initial_capital))

    if not PAPER_TRADES_FILE.exists():
        PAPER_TRADES_FILE.write_text("[]", encoding="utf-8")


def load_paper_state(config: PaperTradingConfig | None = None) -> dict[str, Any]:
    ensure_paper_files(config)
    desired_initial_capital = float(config.initial_capital) if config else VALIDATION_INITIAL_CAPITAL_BRL
    state = load_json_state(
        PAPER_STATE_NAMESPACE,
        lambda: _default_state(desired_initial_capital),
        PAPER_STATE_FILE,
    )
    merged = _default_state(float(state.get("initial_capital", desired_initial_capital) or desired_initial_capital))
    merged.update(state)
    merged["positions"] = merged.get("positions", {}) or {}
    merged["asset_cooldowns"] = merged.get("asset_cooldowns", {}) or {}
    merged["last_prices"] = merged.get("last_prices", {}) or {}
    merged["history"] = merged.get("history", []) or []
    if _should_migrate_fresh_paper_state(merged, desired_initial_capital):
        merged = _default_state(desired_initial_capital)
        save_paper_state(merged)
    return merged


def save_paper_state(state: dict[str, Any]) -> None:
    ensure_app_directories()
    save_json_state(PAPER_STATE_NAMESPACE, state, PAPER_STATE_FILE)


def reset_paper_state(config: PaperTradingConfig | None = None) -> dict[str, Any]:
    initial_capital = float(config.initial_capital) if config else 10000.0
    state = _default_state(initial_capital)
    save_paper_state(state)
    replace_json_rows(PAPER_TRADES_NAMESPACE, [], PAPER_TRADES_FILE)
    return state


def read_paper_trades(limit: int | None = None) -> list[dict[str, Any]]:
    ensure_paper_files()
    trades = load_json_rows(PAPER_TRADES_NAMESPACE, PAPER_TRADES_FILE, limit=limit)
    if limit is None:
        return trades
    return trades[-int(limit):]


def _append_paper_trade(trade: dict[str, Any]) -> None:
    append_json_row(PAPER_TRADES_NAMESPACE, trade, PAPER_TRADES_FILE)


def read_paper_equity(limit: int | None = None) -> pd.DataFrame:
    state = load_paper_state()
    history = state.get("history", []) or []

    if not history:
        df = pd.DataFrame(
            [
                {
                    "timestamp": state.get("updated_at") or _utc_now_iso(),
                    "cash": float(state.get("cash", 0.0)),
                    "equity": float(state.get("equity", state.get("cash", 0.0))),
                }
            ]
        )
    else:
        df = pd.DataFrame(history)

    if limit is not None and not df.empty:
        df = df.tail(int(limit)).reset_index(drop=True)

    return df


def fallback_data(symbol: str, rows: int = 240) -> pd.DataFrame:
    return build_fallback_market_data(symbol, rows=rows)


def load_market_data_result(symbols: list[str], config: PaperTradingConfig, requested_by: str = "worker_cycle") -> dict[str, Any]:
    result = fetch_market_data_map(
        symbols,
        period=config.period,
        interval=config.interval,
        history_limit=int(config.history_limit),
        allow_stale=True,
        requested_by=requested_by,
    )
    return {"frames": result.frames, "status": result.status}


def load_market_data_map(symbols: list[str], config: PaperTradingConfig) -> dict[str, pd.DataFrame]:
    return load_market_data_result(symbols, config)["frames"]


def load_prices(symbol: str, config: PaperTradingConfig) -> pd.DataFrame:
    frame, _status = fetch_market_data_frame(
        symbol,
        period=config.period,
        interval=config.interval,
        history_limit=int(config.history_limit),
        allow_stale=True,
        requested_by="trader_chart",
    )
    return frame


def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(data.get("high"), errors="coerce")
    low = pd.to_numeric(data.get("low"), errors="coerce")
    close = pd.to_numeric(data.get("close"), errors="coerce")
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(period).mean()


def _clip_score(value: float, floor: float = 0.0, ceil: float = 1.0) -> float:
    return float(min(max(value, floor), ceil))


def _dynamic_stop_pct(atr_pct: float, multiplier: float = 2.4) -> float:
    if atr_pct <= 0:
        return 0.03
    return float(min(max(atr_pct * multiplier, 0.018), 0.055))


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["open"] = pd.to_numeric(data.get("open"), errors="coerce")
    data["high"] = pd.to_numeric(data.get("high"), errors="coerce")
    data["low"] = pd.to_numeric(data.get("low"), errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["returns"] = data["close"].pct_change(fill_method=None)
    data["ma20"] = data["close"].rolling(20).mean()
    data["ma50"] = data["close"].rolling(50).mean()
    data["ma20_slope"] = data["ma20"].pct_change(5, fill_method=None).fillna(0.0)
    data["ma50_slope"] = data["ma50"].pct_change(5, fill_method=None).fillna(0.0)
    data["rsi"] = _calculate_rsi(data["close"])
    data["atr"] = _calculate_atr(data)
    data["atr_pct"] = (data["atr"] / data["close"]).replace([np.inf, -np.inf], np.nan)
    data["volatility"] = data["returns"].rolling(12).std().fillna(0.0)
    data["momentum"] = data["close"].pct_change(3, fill_method=None).fillna(0.0)
    data["momentum_8"] = data["close"].pct_change(8, fill_method=None).fillna(0.0)
    data["breakout_20"] = (data["close"] / data["close"].rolling(20).max()) - 1.0
    data["pullback_to_ma20"] = (data["close"] / data["ma20"]) - 1.0
    return data.dropna(subset=["close", "ma20", "ma50", "atr_pct"]).reset_index(drop=True)


def _normalize_symbol_list(config: PaperTradingConfig) -> list[str]:
    symbols = [str(symbol).strip().upper() for symbol in (config.custom_tickers or []) if str(symbol).strip()]
    return symbols or list(SWING_VALIDATION_RECOMMENDED_WATCHLIST)


def _symbols_for_cycle(state: dict[str, Any], config: PaperTradingConfig) -> list[str]:
    ordered_symbols: list[str] = []
    seen: set[str] = set()

    for symbol in _normalize_symbol_list(config):
        if symbol not in seen:
            seen.add(symbol)
            ordered_symbols.append(symbol)

    for asset in (state.get("positions", {}) or {}).keys():
        normalized = str(asset).strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered_symbols.append(normalized)

    return ordered_symbols


def _frame_data_source(data: pd.DataFrame | None) -> str:
    return get_frame_data_source(data if data is not None else pd.DataFrame())


def _is_live_market_source(data_source: str) -> bool:
    return is_live_market_source(data_source)


def _compute_buy_score(latest: pd.Series) -> float:
    price = float(latest["close"])
    ma20 = float(latest.get("ma20", price))
    ma50 = float(latest.get("ma50", price))
    rsi = float(latest.get("rsi", 50.0))
    atr_pct = float(latest.get("atr_pct", 0.0))
    momentum = float(latest.get("momentum", 0.0))
    momentum_8 = float(latest.get("momentum_8", 0.0))
    breakout_20 = float(latest.get("breakout_20", 0.0))
    pullback_to_ma20 = float(latest.get("pullback_to_ma20", 0.0))
    ma20_slope = float(latest.get("ma20_slope", 0.0))
    ma50_slope = float(latest.get("ma50_slope", 0.0))

    trend_score = 1.0 if price >= ma20 >= ma50 else 0.55 if price >= ma50 else 0.0
    rsi_score = _clip_score(1.0 - abs(rsi - 54.0) / 16.0)
    atr_score = _clip_score(1.0 - abs(atr_pct - 0.018) / 0.03)
    slope_score = _clip_score((ma20_slope + 0.01) / 0.02) * 0.65 + _clip_score((ma50_slope + 0.01) / 0.02) * 0.35
    breakout_score = _clip_score((breakout_20 + 0.04) / 0.08)
    pullback_score = _clip_score(1.0 - abs(pullback_to_ma20) / 0.035)
    momentum_score = 0.55 * _clip_score((momentum + 0.01) / 0.04) + 0.45 * _clip_score((momentum_8 + 0.02) / 0.06)

    return round(
        0.28 * trend_score
        + 0.20 * rsi_score
        + 0.12 * atr_score
        + 0.16 * slope_score
        + 0.10 * breakout_score
        + 0.08 * pullback_score
        + 0.06 * momentum_score,
        4,
    )


def _signal_snapshot_key(asset: str, latest: pd.Series, data_source: str) -> str:
    raw_datetime = latest.get("datetime")
    if isinstance(raw_datetime, pd.Timestamp):
        signal_dt = raw_datetime.isoformat()
    else:
        signal_dt = str(raw_datetime or "")
    return f"{str(asset).upper()}|{signal_dt}|{str(data_source or 'unknown').lower()}"


def _evaluate_buy_signal(latest: pd.Series, config: PaperTradingConfig) -> dict[str, Any]:
    score = _compute_buy_score(latest)
    price = float(latest["close"])
    ma20 = float(latest.get("ma20", price))
    ma50 = float(latest.get("ma50", price))
    rsi = float(latest.get("rsi", 50.0))
    atr_pct = float(latest.get("atr_pct", 0.0))
    momentum = float(latest.get("momentum", 0.0))
    momentum_8 = float(latest.get("momentum_8", 0.0))
    breakout_20 = float(latest.get("breakout_20", 0.0))
    pullback_to_ma20 = float(latest.get("pullback_to_ma20", 0.0))
    ma20_slope = float(latest.get("ma20_slope", 0.0))
    ma50_slope = float(latest.get("ma50_slope", 0.0))

    dominant_trend = "aligned" if price >= ma20 >= ma50 else "countertrend" if price < ma50 else "neutral"
    trend_ok = (
        price >= ma50
        and ma20 >= ma50
        and ma20_slope >= float(config.ma20_slope_min)
        and ma50_slope >= float(config.ma50_slope_min)
    )
    rsi_ok = float(config.rsi_entry_min) <= rsi <= float(config.rsi_entry_max)
    atr_ok = float(config.min_atr_pct) <= atr_pct <= float(config.max_atr_pct)
    structure_ok = float(config.pullback_min) <= pullback_to_ma20 <= float(config.pullback_max)
    momentum_ok = (
        momentum >= float(config.momentum_min)
        and momentum_8 >= float(config.momentum_8_min)
        and breakout_20 >= float(config.breakout_min)
    )

    setup_ok = trend_ok and rsi_ok and atr_ok and structure_ok and momentum_ok
    score_ok = score >= float(config.min_signal_score)

    rejection_reasons: list[str] = []
    if not trend_ok:
        rejection_reasons.append("against_trend")
    if not score_ok:
        rejection_reasons.append("weak_score")
    if not rsi_ok:
        rejection_reasons.append("rsi_filter")
    if not atr_ok:
        rejection_reasons.append("atr_filter")
    if not structure_ok:
        rejection_reasons.append("structure_filter")
    if not momentum_ok:
        rejection_reasons.append("momentum_filter")

    return {
        "buy": bool(setup_ok and score_ok),
        "score": score,
        "trend_ok": trend_ok,
        "score_ok": score_ok,
        "rsi_ok": rsi_ok,
        "atr_ok": atr_ok,
        "structure_ok": structure_ok,
        "momentum_ok": momentum_ok,
        "trend_alignment": dominant_trend,
        "rejection_reasons": rejection_reasons,
    }


def _should_buy(latest: pd.Series, config: PaperTradingConfig) -> tuple[bool, float]:
    evaluation = _evaluate_buy_signal(latest, config)
    return bool(evaluation["buy"]), float(evaluation["score"])


def _is_position_expired(opened_at: str | None, holding_minutes: int) -> bool:
    opened = _parse_iso_datetime(opened_at)
    if opened is None:
        return False
    return _utc_now() >= opened + timedelta(minutes=int(holding_minutes))


def _position_age_minutes(opened_at: str | None) -> float | None:
    opened = _parse_iso_datetime(opened_at)
    if opened is None:
        return None
    return max((_utc_now() - opened).total_seconds() / 60.0, 0.0)


def _asset_in_cooldown(state: dict[str, Any], asset: str, cooldown_minutes: int) -> bool:
    if cooldown_minutes <= 0:
        return False
    normalized_asset = str(asset).upper()
    cooldown_map = state.get("asset_cooldowns", {}) or {}
    closed_at = _parse_iso_datetime(cooldown_map.get(normalized_asset) or cooldown_map.get(str(asset)))
    if closed_at is None:
        return False
    return _utc_now() < closed_at + timedelta(minutes=int(cooldown_minutes))


def _should_sell(position: dict[str, Any], latest: pd.Series, holding_minutes: int, config: PaperTradingConfig) -> tuple[bool, str]:
    price = float(latest["close"])
    holding_limit = int(position.get("holding_minutes_limit", holding_minutes) or holding_minutes)
    ma20 = float(latest.get("ma20", price))
    ma50 = float(latest.get("ma50", price))
    rsi = float(latest.get("rsi", 50.0))
    momentum = float(latest.get("momentum", 0.0))
    atr_pct = float(latest.get("atr_pct", position.get("atr_pct", 0.0) or 0.0))
    avg_price = float(position.get("avg_price", price) or price)
    peak_price = max(float(position.get("peak_price", position.get("avg_price", price) or price)), price)
    age_minutes = _position_age_minutes(position.get("opened_at"))
    allow_soft_exit = age_minutes is None or age_minutes >= float(config.min_position_age_minutes)
    trailing_stop = max(
        float(position.get("trailing_stop", 0.0) or 0.0),
        round(peak_price * (1.0 - _dynamic_stop_pct(atr_pct, multiplier=float(config.trailing_stop_atr_mult))), 6),
    )
    hard_stop = round(avg_price * (1.0 - float(config.hard_stop_loss_pct)), 6)

    if hard_stop > 0 and price <= hard_stop:
        return True, "stop loss"
    if trailing_stop > 0 and price <= trailing_stop:
        return True, "trailing stop"
    if allow_soft_exit and rsi >= float(config.take_profit_rsi) and momentum <= 0.04:
        return True, "rsi esticado"
    if allow_soft_exit and price < ma20 and momentum < float(config.weak_momentum_exit_threshold):
        return True, "perdeu media curta"
    if allow_soft_exit and price < ma50 * float(config.trend_break_buffer_pct):
        return True, "perdeu tendencia"
    if _is_position_expired(position.get("opened_at"), holding_limit):
        return True, "holding expirado"
    return False, ""


def _position_size_notional(candidate: dict[str, Any], config: PaperTradingConfig, cash: float) -> float:
    base_notional = float(config.ticket_value)
    score = float(candidate.get("score", 0.0))
    atr_pct = float(candidate.get("atr_pct", 0.0) or 0.0)

    score_boost = _clip_score((score - float(config.min_signal_score)) / 0.18)
    score_multiplier = 0.85 + (score_boost * 0.20)

    if atr_pct > 0:
        volatility_multiplier = _clip_score(0.028 / atr_pct, 0.65, 1.0)
    else:
        volatility_multiplier = 0.85

    desired_notional = round(base_notional * min(score_multiplier, volatility_multiplier), 2)
    return min(desired_notional, cash)


def _portfolio_equity(state: dict[str, Any]) -> float:
    cash = float(state.get("cash", 0.0))
    equity = cash

    for asset, position in (state.get("positions", {}) or {}).items():
        last_price = float(state.get("last_prices", {}).get(asset, position.get("last_price", 0.0)))
        equity += float(position.get("quantity", 0.0)) * last_price

    return round(equity, 2)


def build_paper_report(
    result: dict[str, Any] | None = None,
    initial_capital: float | None = None,
) -> dict[str, Any]:
    state = load_paper_state()
    initial = float(initial_capital or state.get("initial_capital", VALIDATION_INITIAL_CAPITAL_BRL))
    equity = float(state.get("equity", state.get("cash", initial)))
    net_profit = round(equity - initial, 2)
    total_return = (net_profit / initial) if initial > 0 else 0.0
    trades = read_paper_trades()
    signals = len(result.get("signals", [])) if isinstance(result, dict) else 0

    return {
        "status": "running",
        "cash": round(float(state.get("cash", 0.0)), 2),
        "equity": equity,
        "realized_pnl": round(float(state.get("realized_pnl", 0.0)), 2),
        "net_profit": net_profit,
        "total_return": total_return,
        "trades": len(trades),
        "trades_count": len(trades),
        "signals": signals,
        "open_positions": len(state.get("positions", {}) or {}),
        "run_count": int(state.get("run_count", 0)),
    }


def run_paper_trading_demo(config: PaperTradingConfig = PaperTradingConfig()) -> dict[str, Any]:
    symbols = _normalize_symbol_list(config)
    first_symbol = symbols[0]
    prices = _enrich_indicators(load_market_data_map([first_symbol], config).get(first_symbol, pd.DataFrame()))
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=prices["datetime"], y=prices["close"], mode="lines", name=first_symbol))

    latest = prices.iloc[-1]
    should_buy, score = _should_buy(latest, config)

    return {
        "symbol": first_symbol,
        "signals": [{"asset": first_symbol, "score": score, "buy": should_buy, "data_source": _frame_data_source(prices)}],
        "result": {"total_return": 0.0, "trades": 0},
        "equity": prices["close"].tail(50).tolist(),
        "figure": figure,
    }


def run_paper_cycle(config: PaperTradingConfig = PaperTradingConfig()) -> dict[str, Any]:
    ensure_paper_files(config)
    state = load_paper_state(config)

    if int(state.get("run_count", 0)) == 0 and not state.get("positions") and float(state.get("cash", 0.0)) <= 0:
        state["cash"] = round(float(config.initial_capital), 2)

    symbols = _symbols_for_cycle(state, config)
    market_data_result = load_market_data_result(symbols, config, requested_by="worker_cycle")
    raw_market_data = market_data_result["frames"]
    market_data_status = market_data_result["status"]
    market_data: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        data = _enrich_indicators(raw_market_data.get(symbol, pd.DataFrame()))
        if data.empty:
            continue
        market_data[symbol] = data
        if _is_live_market_source(_frame_data_source(data)):
            state["last_prices"][symbol] = round(float(data["close"].iloc[-1]), 6)

    trades_executed: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    market_context = get_market_context(market_data)
    cycle_validation = {
        "signals_total": 0,
        "signals_approved": 0,
        "signals_rejected": 0,
        "entries_against_trend": 0,
        "context_blocked": 0,
        "market_context_status": str(market_context.get("market_context_status") or "NEUTRO"),
        "rejections": {
            "against_trend": 0,
            "weak_score": 0,
            "rsi_filter": 0,
            "atr_filter": 0,
            "structure_filter": 0,
            "momentum_filter": 0,
            "feed_unreliable": 0,
            "context_blocked": 0,
            "daily_loss_guard": 0,
        },
    }

    open_positions = dict(state.get("positions", {}) or {})
    for asset, position in list(open_positions.items()):
        data = market_data.get(asset)
        if data is None or data.empty:
            continue
        if not _is_live_market_source(_frame_data_source(data)):
            continue

        latest = data.iloc[-1]
        state["last_prices"][asset] = round(float(latest["close"]), 6)
        should_sell, reason = _should_sell(position, latest, config.holding_minutes, config)
        if not should_sell:
            peak_price = max(float(position.get("peak_price", position.get("avg_price", latest["close"]) or latest["close"])), float(latest["close"]))
            atr_pct = float(latest.get("atr_pct", position.get("atr_pct", 0.0) or 0.0))
            trailing_stop = max(
                float(position.get("trailing_stop", 0.0) or 0.0),
                round(peak_price * (1.0 - _dynamic_stop_pct(atr_pct, multiplier=float(config.trailing_stop_atr_mult))), 6),
            )
            position["last_price"] = round(float(latest["close"]), 6)
            position["peak_price"] = round(peak_price, 6)
            position["atr_pct"] = round(atr_pct, 6)
            position["trailing_stop"] = trailing_stop
            position["updated_at"] = _utc_now_iso()
            state["positions"][asset] = position
            continue

        quantity = float(position.get("quantity", 0.0))
        price = float(latest["close"])
        gross_value = round(quantity * price, 2)
        avg_price = float(position.get("avg_price", price))
        realized = round((price - avg_price) * quantity, 2)
        peak_price = max(float(position.get("peak_price", avg_price) or avg_price), price)
        atr_pct = float(latest.get("atr_pct", position.get("atr_pct", 0.0) or 0.0))
        trailing_stop = max(
            float(position.get("trailing_stop", 0.0) or 0.0),
            round(peak_price * (1.0 - _dynamic_stop_pct(atr_pct, multiplier=float(config.trailing_stop_atr_mult))), 6),
        )
        closed_at = _utc_now_iso()
        state["cash"] = round(float(state.get("cash", 0.0)) + gross_value, 2)
        state["realized_pnl"] = round(float(state.get("realized_pnl", 0.0)) + realized, 2)
        state["last_trade_at"] = closed_at
        state.setdefault("asset_cooldowns", {})
        state["asset_cooldowns"][str(asset).upper()] = closed_at
        del state["positions"][asset]

        trade = {
            "timestamp": closed_at,
            "asset": asset,
            "side": "SELL",
            "trade_id": position.get("trade_id"),
            "profile": position.get("profile"),
            "quantity": round(quantity, 6),
            "price": round(price, 6),
            "gross_value": gross_value,
            "cost": 0.0,
            "cash_after": round(float(state["cash"]), 2),
            "realized_pnl": realized,
            "reason": reason,
        }
        trades_executed.append(trade)
        _append_paper_trade(trade)
        append_trade_report(
            build_closed_trade_report(
                position=position,
                closed_at=closed_at,
                asset=asset,
                quantity=quantity,
                exit_price=price,
                realized_pnl=realized,
                exit_reason=reason,
                exit_snapshot=latest,
                trailing_stop_final=trailing_stop,
            )
        )

    current_day_key = str(config.daily_loss_day_key or _utc_day_key()).strip() or _utc_day_key()
    daily_realized_running = float(config.daily_realized_pnl_brl or 0.0)
    for trade in trades_executed:
        if str(trade.get("side") or "").upper() != "SELL":
            continue
        if str(trade.get("timestamp") or "").startswith(current_day_key):
            daily_realized_running += float(trade.get("realized_pnl", 0.0) or 0.0)

    daily_loss_limit = max(0.0, float(config.daily_loss_limit_brl or 0.0))
    daily_loss_consumed = max(0.0, -daily_realized_running)
    daily_loss_block_active = bool(config.daily_loss_block_active)
    daily_loss_block_reason = str(config.daily_loss_block_reason or "")
    if daily_loss_limit > 0 and daily_loss_consumed >= daily_loss_limit:
        daily_loss_block_active = True
        if not daily_loss_block_reason:
            daily_loss_block_reason = (
                f"Limite de perda diaria atingido ({daily_loss_consumed:.2f}/{daily_loss_limit:.2f}). "
                "Novas entradas bloqueadas."
            )

    slots_left = 0 if daily_loss_block_active else max(int(config.max_open_positions) - len(state.get("positions", {}) or {}), 0)
    candidates: list[dict[str, Any]] = []

    for asset, data in market_data.items():
        if asset in state.get("positions", {}):
            continue
        if _asset_in_cooldown(state, asset, int(config.reentry_cooldown_minutes)):
            continue

        latest = data.iloc[-1]
        data_source = _frame_data_source(data)
        evaluation = _evaluate_buy_signal(latest, config)
        should_buy = bool(evaluation["buy"])
        score = float(evaluation["score"])
        context_filter = apply_context_filter(
            {
                "asset": asset,
                "score": score,
                "base_min_signal_score": float(config.min_signal_score),
            },
            market_context,
        )
        adjusted_score = float(context_filter["adjusted_score"])
        effective_min_signal_score = float(context_filter["effective_min_signal_score"])
        if should_buy and (bool(context_filter["blocked"]) or adjusted_score < effective_min_signal_score):
            should_buy = False
            if "context_blocked" not in evaluation["rejection_reasons"]:
                evaluation["rejection_reasons"] = [*evaluation["rejection_reasons"], "context_blocked"]
        if daily_loss_block_active:
            should_buy = False
            if "daily_loss_guard" not in evaluation["rejection_reasons"]:
                evaluation["rejection_reasons"] = [*evaluation["rejection_reasons"], "daily_loss_guard"]
        if not _is_live_market_source(data_source):
            should_buy = False
            if "feed_unreliable" not in evaluation["rejection_reasons"]:
                evaluation["rejection_reasons"] = [*evaluation["rejection_reasons"], "feed_unreliable"]

        signal_key = _signal_snapshot_key(asset, latest, data_source)
        signal = {
            "asset": asset,
            "price": round(float(latest["close"]), 6),
            "rsi": round(float(latest.get("rsi", 50.0)), 2),
            "atr_pct": round(float(latest.get("atr_pct", 0.0) or 0.0), 4),
            "pullback_to_ma20": round(float(latest.get("pullback_to_ma20", 0.0) or 0.0), 4),
            "ma20": round(float(latest.get("ma20", latest["close"])), 6),
            "ma50": round(float(latest.get("ma50", latest["close"])), 6),
            "momentum": round(float(latest.get("momentum", 0.0) or 0.0), 6),
            "momentum_8": round(float(latest.get("momentum_8", 0.0) or 0.0), 6),
            "breakout_20": round(float(latest.get("breakout_20", 0.0) or 0.0), 6),
            "score": adjusted_score,
            "raw_score": score,
            "buy": should_buy,
            "data_source": data_source,
            "signal_key": signal_key,
            "trend_ok": bool(evaluation["trend_ok"]),
            "score_ok": bool(evaluation["score_ok"]),
            "rsi_ok": bool(evaluation["rsi_ok"]),
            "atr_ok": bool(evaluation["atr_ok"]),
            "structure_ok": bool(evaluation["structure_ok"]),
            "momentum_ok": bool(evaluation["momentum_ok"]),
            "trend_alignment": str(evaluation["trend_alignment"]),
            "rejection_reasons": list(evaluation["rejection_reasons"]),
            "base_min_signal_score": float(config.min_signal_score),
            "effective_min_signal_score": effective_min_signal_score,
            "context_status": str(context_filter["context_status"]),
            "context_impact": str(context_filter["impact_message"]),
        }
        signals.append(signal)
        cycle_validation["signals_total"] = int(cycle_validation["signals_total"]) + 1
        if should_buy:
            cycle_validation["signals_approved"] = int(cycle_validation["signals_approved"]) + 1
            if str(evaluation["trend_alignment"]).lower() == "countertrend":
                cycle_validation["entries_against_trend"] = int(cycle_validation["entries_against_trend"]) + 1
        else:
            cycle_validation["signals_rejected"] = int(cycle_validation["signals_rejected"]) + 1
            for reason in signal["rejection_reasons"]:
                if reason not in cycle_validation["rejections"]:
                    cycle_validation["rejections"][reason] = 0
                cycle_validation["rejections"][reason] = int(cycle_validation["rejections"][reason]) + 1
            if "context_blocked" in signal["rejection_reasons"]:
                cycle_validation["context_blocked"] = int(cycle_validation["context_blocked"]) + 1

        if should_buy:
            candidates.append(signal)

    candidates.sort(key=lambda item: item["score"], reverse=True)

    for candidate in candidates[:slots_left]:
        if not config.allow_new_entries:
            break

        cash = float(state.get("cash", 0.0))
        notional = _position_size_notional(candidate, config, cash)
        if notional < float(config.min_trade_notional):
            continue

        asset = str(candidate["asset"])
        price = float(candidate["price"])
        quantity = round(notional / price, 6)
        if quantity <= 0:
            continue

        trade_id = _new_trade_id(asset)
        opened_at = _utc_now_iso()
        state["cash"] = round(cash - notional, 2)
        state["positions"][asset] = {
            "trade_id": trade_id,
            "profile": str(config.profile_name),
            "quantity": quantity,
            "entry_price": round(price, 6),
            "avg_price": round(price, 6),
            "last_price": round(price, 6),
            "peak_price": round(price, 6),
            "atr_pct": round(float(candidate.get("atr_pct", 0.0) or 0.0), 6),
            "atr_pct_entry": round(float(candidate.get("atr_pct", 0.0) or 0.0), 6),
            "trailing_stop": round(
                price * (1.0 - _dynamic_stop_pct(float(candidate.get("atr_pct", 0.0) or 0.0), multiplier=float(config.trailing_stop_atr_mult))),
                6,
            ),
            "gross_entry_value": round(notional, 2),
            "entry_score": round(float(candidate.get("score", 0.0)), 4),
            "rsi_entry": round(float(candidate.get("rsi", 50.0) or 50.0), 2),
            "ma20_entry": round(float(candidate.get("ma20", price) or price), 6),
            "ma50_entry": round(float(candidate.get("ma50", price) or price), 6),
            "entry_trend_alignment": str(candidate.get("trend_alignment") or "aligned"),
            "entry_signal_key": str(candidate.get("signal_key") or ""),
            "entry_data_source": str(candidate.get("data_source") or "unknown"),
            "holding_minutes_limit": int(config.holding_minutes),
            "opened_at": opened_at,
            "updated_at": opened_at,
        }
        state["last_trade_at"] = opened_at

        trade = {
            "timestamp": opened_at,
            "asset": asset,
            "side": "BUY",
            "trade_id": trade_id,
            "profile": str(config.profile_name),
            "quantity": quantity,
            "price": round(price, 6),
            "gross_value": round(notional, 2),
            "cost": 0.0,
            "cash_after": round(float(state["cash"]), 2),
            "realized_pnl": 0.0,
            "reason": f"score {candidate['score']:.2f} | rsi {candidate['rsi']:.1f}",
        }
        trades_executed.append(trade)
        _append_paper_trade(trade)

    state["run_count"] = int(state.get("run_count", 0)) + 1
    state["updated_at"] = _utc_now_iso()
    state["equity"] = _portfolio_equity(state)

    history = state.get("history", []) or []
    history.append(
        {
            "timestamp": state["updated_at"],
            "cash": round(float(state["cash"]), 2),
            "equity": round(float(state["equity"]), 2),
            "open_positions": len(state.get("positions", {}) or {}),
            "total_return": round(
                (float(state["equity"]) - float(state.get("initial_capital", config.initial_capital)))
                / max(float(state.get("initial_capital", config.initial_capital)), 1e-9),
                6,
            ),
        }
    )
    state["history"] = history[-max(int(config.history_limit), 1) :]
    save_paper_state(state)

    report = build_paper_report({"signals": signals}, initial_capital=float(config.initial_capital))

    return {
        "state": state,
        "trades": read_paper_trades(limit=200),
        "equity": read_paper_equity(limit=200),
        "report": report,
        "signals": signals,
        "trades_executed": len(trades_executed),
        "entries_blocked_by_daily_loss": daily_loss_block_active,
        "entries_block_reason": daily_loss_block_reason,
        "daily_loss_limit_brl": round(float(daily_loss_limit), 2),
        "daily_loss_consumed_brl": round(float(daily_loss_consumed), 2),
        "daily_realized_pnl_brl": round(float(daily_realized_running), 2),
        "daily_loss_day_key": current_day_key,
        "market_data_status": market_data_status,
        "market_context": market_context,
        "validation_cycle": cycle_validation,
    }
