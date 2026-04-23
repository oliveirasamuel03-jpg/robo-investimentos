from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta

import pandas as pd

from core.config import (
    ALERT_EMAIL_ENABLED,
    ALERT_EMAIL_PROVIDER,
    BUILD_TIMESTAMP,
    BOT_LOG_COLUMNS,
    BOT_LOG_FILE,
    BOT_STATE_FILE,
    BROKER_MODE,
    BROKER_PROVIDER,
    DAILY_LOSS_LIMIT_BRL_DEFAULT,
    MARKET_DATA_FALLBACK_PROVIDER,
    MARKET_DATA_BUILD_LABEL,
    MARKET_DATA_PROVIDER,
    PRODUCTION_MODE,
    RAILWAY_GIT_COMMIT_SHA,
    REPORT_EMAIL_10DAY_ENABLED,
    REPORT_EMAIL_DAILY_ENABLED,
    REPORT_EMAIL_ENABLED,
    REPORT_EMAIL_FINAL_ENABLED,
    REPORT_EMAIL_PROVIDER,
    REPORT_EMAIL_TO,
    REPORT_EMAIL_WEEKLY_ENABLED,
    SERVICE_NAME,
    STATE_SCHEMA_VERSION,
    RETENTION_ARCHIVE_TRADER_ORDERS,
    RETENTION_DAYS,
    RETENTION_ENABLED,
    RETENTION_RUN_INTERVAL_HOURS,
    LEGACY_MIXED_DEFAULT_WATCHLIST,
    LEGACY_VALIDATION_INITIAL_CAPITAL_BRL,
    SWING_VALIDATION_RECOMMENDED_WATCHLIST,
    TRADER_REPORTS_COLUMNS,
    TRADER_REPORTS_FILE,
    TRADER_ORDERS_COLUMNS,
    TRADER_ORDERS_FILE,
    VALIDATION_DEFAULT_ENTRY_AMOUNT_BRL,
    VALIDATION_DEFAULT_MAX_OPEN_POSITIONS,
    VALIDATION_INITIAL_CAPITAL_BRL,
    VALIDATION_LIVE_TRADING_ENABLED,
    VALIDATION_MODE_DISPLAY,
    VALIDATION_TRADING_MODE,
    ensure_app_directories,
)
from core.market_data import classify_feed_status, legacy_market_status
from core.persistence import (
    append_table_row,
    database_enabled,
    load_json_state,
    read_table,
    replace_table,
    save_json_state,
)
from core.trader_profiles import DEFAULT_TRADER_PROFILE


def _normalize_watchlist_values(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        item = str(value or "").strip().upper()
        if item and item not in normalized:
            normalized.append(item)
    return normalized


def _should_migrate_watchlist(values: list[str]) -> bool:
    return _normalize_watchlist_values(values) == _normalize_watchlist_values(LEGACY_MIXED_DEFAULT_WATCHLIST)


def _apply_trader_watchlist_defaults(state: dict) -> dict:
    trader_state = state.get("trader", {}) or {}
    current_watchlist = _normalize_watchlist_values(trader_state.get("watchlist", []))

    if not current_watchlist or _should_migrate_watchlist(current_watchlist):
        trader_state["watchlist"] = list(SWING_VALIDATION_RECOMMENDED_WATCHLIST)
    else:
        trader_state["watchlist"] = current_watchlist

    state["trader"] = trader_state
    return state


def _is_close_number(current: float | int | None, expected: float) -> bool:
    try:
        return abs(float(current or 0.0) - float(expected)) < 1e-9
    except (TypeError, ValueError):
        return False


def _looks_like_fresh_validation_cycle(state: dict) -> bool:
    positions = state.get("positions", []) or []
    validation_state = state.get("validation", {}) or {}

    if positions:
        return False
    if not _is_close_number(state.get("realized_pnl", 0.0), 0.0):
        return False
    if str(state.get("last_run_at") or "").strip():
        return False
    if str(validation_state.get("validation_started_at") or "").strip():
        return False
    if validation_state.get("last_report"):
        return False

    return True


def _apply_validation_operating_defaults(state: dict) -> dict:
    trader_state = state.get("trader", {}) or {}
    validation_state = state.get("validation", {}) or {}
    security_state = state.get("security", {}) or {}

    if _looks_like_fresh_validation_cycle(state):
        if _is_close_number(state.get("wallet_value"), LEGACY_VALIDATION_INITIAL_CAPITAL_BRL):
            state["wallet_value"] = float(VALIDATION_INITIAL_CAPITAL_BRL)
        if _is_close_number(state.get("cash"), LEGACY_VALIDATION_INITIAL_CAPITAL_BRL):
            state["cash"] = float(VALIDATION_INITIAL_CAPITAL_BRL)
        if _is_close_number(trader_state.get("ticket_value"), VALIDATION_DEFAULT_ENTRY_AMOUNT_BRL):
            trader_state["ticket_value"] = float(VALIDATION_DEFAULT_ENTRY_AMOUNT_BRL)
        if int(trader_state.get("max_open_positions", 0) or 0) == 3:
            trader_state["max_open_positions"] = int(VALIDATION_DEFAULT_MAX_OPEN_POSITIONS)

    validation_state["validation_mode_label"] = VALIDATION_MODE_DISPLAY
    validation_state["trading_mode"] = VALIDATION_TRADING_MODE
    validation_state["live_trading_enabled"] = bool(security_state.get("real_mode_enabled", VALIDATION_LIVE_TRADING_ENABLED))
    validation_state["paper_only"] = True

    state["trader"] = trader_state
    state["validation"] = validation_state
    return state


def _normalize_market_data_state(state: dict) -> dict:
    market_state = state.get("market_data", {}) or {}

    def normalize_payload(payload: dict) -> dict:
        normalized = dict(payload or {})
        normalized["provider"] = str(normalized.get("provider") or MARKET_DATA_PROVIDER)
        normalized["provider_effective"] = str(
            normalized.get("provider_effective") or normalized.get("provider") or MARKET_DATA_PROVIDER
        )
        normalized["configured_provider"] = str(normalized.get("configured_provider") or MARKET_DATA_PROVIDER)
        normalized["fallback_provider"] = str(normalized.get("fallback_provider") or MARKET_DATA_FALLBACK_PROVIDER)
        normalized["provider_chain"] = [str(item) for item in (normalized.get("provider_chain") or []) if str(item)]
        normalized["provider_breakdown"] = dict(normalized.get("provider_breakdown") or {})
        normalized["provider_diagnostics"] = dict(normalized.get("provider_diagnostics") or {})
        normalized["build_active"] = str(normalized.get("build_active") or "")
        normalized["git_sha"] = str(normalized.get("git_sha") or "")
        normalized["build_timestamp"] = str(normalized.get("build_timestamp") or "")
        normalized["runtime_started_at"] = str(normalized.get("runtime_started_at") or "")
        normalized["service_name"] = str(normalized.get("service_name") or "")
        normalized["process_role"] = str(normalized.get("process_role") or "")
        normalized["api_key_present"] = bool(normalized.get("api_key_present", False))
        normalized["request_prepared"] = bool(normalized.get("request_prepared", False))
        normalized["request_attempted"] = bool(normalized.get("request_attempted", False))
        normalized["response_received"] = bool(normalized.get("response_received", False))
        normalized["response_status_code"] = normalized.get("response_status_code")
        normalized["last_stage"] = str(normalized.get("last_stage") or "")
        normalized["requested_symbols"] = [str(item).upper() for item in (normalized.get("requested_symbols") or []) if str(item)]
        normalized["live_symbols"] = [str(item).upper() for item in (normalized.get("live_symbols") or []) if str(item)]
        normalized["cached_symbols"] = [str(item).upper() for item in (normalized.get("cached_symbols") or []) if str(item)]
        normalized["fallback_symbols"] = [str(item).upper() for item in (normalized.get("fallback_symbols") or []) if str(item)]
        normalized["unknown_symbols"] = [str(item).upper() for item in (normalized.get("unknown_symbols") or []) if str(item)]
        normalized["state_writer"] = str(normalized.get("state_writer") or "")
        normalized["state_written_at"] = str(normalized.get("state_written_at") or "")
        normalized["state_build_sha"] = str(normalized.get("state_build_sha") or "")
        normalized["state_schema_version"] = str(normalized.get("state_schema_version") or "")
        normalized["ui_audit_probe"] = str(normalized.get("ui_audit_probe") or "")
        source_breakdown = normalized.get("source_breakdown", {}) or {}
        last_source = normalized.get("last_source")
        status_value = normalized.get("status")
        normalized["status_legacy"] = legacy_market_status(
            status=status_value,
            last_source=last_source,
            source_breakdown=source_breakdown,
        )
        normalized["status"] = normalized["status_legacy"]
        normalized["feed_status"] = classify_feed_status(
            status=normalized.get("feed_status") or status_value,
            last_source=last_source,
            source_breakdown=source_breakdown,
        )
        return normalized

    market_state = normalize_payload(market_state)
    contexts = market_state.get("contexts", {}) or {}
    market_state["contexts"] = {
        str(name): normalize_payload(dict(payload or {}))
        for name, payload in contexts.items()
        if isinstance(payload, dict)
    }
    state["market_data"] = market_state
    return state


DEFAULT_STATE = {
    "wallet_value": VALIDATION_INITIAL_CAPITAL_BRL,
    "cash": VALIDATION_INITIAL_CAPITAL_BRL,
    "bot_status": "PAUSED",
    "bot_mode": "Automatico",
    "realized_pnl": 0.0,
    "positions": [],
    "last_action": "Nenhuma acao recente",
    "last_run_at": "",
    "next_run_at": "",
    "worker_status": "offline",
    "worker_heartbeat": "",
    "market_data": {
        "provider": MARKET_DATA_PROVIDER,
        "provider_effective": MARKET_DATA_PROVIDER,
        "configured_provider": MARKET_DATA_PROVIDER,
        "fallback_provider": MARKET_DATA_FALLBACK_PROVIDER,
        "provider_chain": [MARKET_DATA_PROVIDER, MARKET_DATA_FALLBACK_PROVIDER],
        "provider_breakdown": {},
        "provider_diagnostics": {},
        "status": "unknown",
        "status_legacy": "unknown",
        "feed_status": "UNKNOWN",
        "last_sync_at": "",
        "last_success_at": "",
        "last_error": "",
        "last_source": "",
        "fallback_since_at": "",
        "source_breakdown": {},
        "symbols": [],
        "requested_symbols": [],
        "live_symbols": [],
        "cached_symbols": [],
        "fallback_symbols": [],
        "unknown_symbols": [],
        "requested_by": "",
        "build_active": "",
        "git_sha": "",
        "build_timestamp": "",
        "runtime_started_at": "",
        "service_name": "",
        "process_role": "",
        "api_key_present": False,
        "request_prepared": False,
        "request_attempted": False,
        "response_received": False,
        "response_status_code": None,
        "last_stage": "",
        "state_writer": "",
        "state_written_at": "",
        "state_build_sha": "",
        "state_schema_version": STATE_SCHEMA_VERSION,
        "ui_audit_probe": "",
        "contexts": {},
    },
    "market_context": {
        "market_context_status": "NEUTRO",
        "market_context_reason": "Contexto neutro por padrao.",
        "market_context_updated_at": "",
        "market_context_score": 50.0,
        "market_context_impact": "Sem restricao adicional sobre sinais de cripto.",
        "market_context_regime": "indefinido",
        "watchlist_consistency": None,
        "btc_move_pct": None,
        "btc_volatility_pct": None,
    },
    "broker": {
        "provider": BROKER_PROVIDER,
        "mode": BROKER_MODE,
        "status": "paper",
        "last_sync_at": "",
        "last_error": "",
        "account_id": "",
        "requested_by": "",
        "configured_mode": BROKER_MODE,
        "effective_mode": BROKER_MODE,
        "base_url": "",
        "api_key_configured": False,
        "api_secret_configured": False,
        "execution_enabled": False,
        "can_submit_orders": False,
        "warning": "",
    },
    "security": {
        "real_mode_enabled": False,
        "real_mode_enabled_by": "",
        "real_mode_enabled_at": "",
    },
    "production": {
        "enabled": PRODUCTION_MODE,
        "alert_email_enabled": ALERT_EMAIL_ENABLED,
        "alert_provider": ALERT_EMAIL_PROVIDER,
        "heartbeat_age_seconds": None,
        "last_execution_at": "",
        "last_success_at": "",
        "feed_status": "UNKNOWN",
        "feed_status_legacy": "unknown",
        "broker_status": "paper",
        "consecutive_errors": 0,
        "health_level": "healthy",
        "health_reason": "healthy",
        "health_message": "Sistema saudavel. Broker em modo simulado (paper). Nenhuma ordem real sera enviada.",
        "last_health_at": "",
        "last_error": "",
        "last_error_at": "",
        "last_exception": "",
        "fallback_since_at": "",
        "fallback_age_minutes": 0,
        "last_alert_sent_at": "",
        "last_alert_type": "",
        "last_alert_subject": "",
        "last_alert_provider": "",
        "last_alert_error": "",
        "next_alert_eligible_at": "",
        "last_recovery_email_at": "",
    },
    "email_reporting": {
        "enabled": REPORT_EMAIL_ENABLED,
        "provider": REPORT_EMAIL_PROVIDER,
        "destination": REPORT_EMAIL_TO,
        "daily_enabled": REPORT_EMAIL_DAILY_ENABLED,
        "weekly_enabled": REPORT_EMAIL_WEEKLY_ENABLED,
        "ten_day_enabled": REPORT_EMAIL_10DAY_ENABLED,
        "final_enabled": REPORT_EMAIL_FINAL_ENABLED,
        "last_daily_report_email_date": "",
        "last_weekly_report_email_week": "",
        "last_10day_report_email_block": "",
        "last_final_report_email_ts": "",
        "last_email_delivery_status": "",
        "last_email_delivery_reason": "",
        "last_email_delivery_attempt_ts": "",
        "last_email_delivery_success_ts": "",
        "last_email_delivery_report_type": "",
        "last_email_delivery_provider": "",
        "last_email_delivery_subject": "",
    },
    "risk": {
        "daily_loss_limit_brl": DAILY_LOSS_LIMIT_BRL_DEFAULT,
        "daily_loss_day_key": "",
        "daily_realized_pnl_brl": 0.0,
        "daily_loss_consumed_brl": 0.0,
        "daily_loss_remaining_brl": DAILY_LOSS_LIMIT_BRL_DEFAULT,
        "daily_loss_block_active": False,
        "daily_loss_block_reason": "",
        "daily_loss_blocked_at": "",
        "daily_loss_blocked_value_brl": 0.0,
        "daily_loss_reset_at": "",
        "last_transition": "none",
        "last_updated_at": "",
    },
    "retention": {
        "enabled": RETENTION_ENABLED,
        "retention_days": RETENTION_DAYS,
        "run_interval_hours": RETENTION_RUN_INTERVAL_HOURS,
        "archive_trader_orders": RETENTION_ARCHIVE_TRADER_ORDERS,
        "last_run_at": "",
        "last_success_at": "",
        "last_error": "",
        "last_error_at": "",
        "last_summary": {},
        "archive_catalog": {
            "reports": [],
            "logs": [],
            "orders": [],
            "weekly_reports": [],
        },
        "weekly_reports_index": [],
    },
    "validation": {
        "validation_mode": "swing_10d",
        "validation_mode_label": VALIDATION_MODE_DISPLAY,
        "validation_started_at": "",
        "validation_day_number": 1,
        "validation_phase": "Coleta e observacao",
        "validation_status": "running",
        "final_validation_grade": "",
        "final_validation_reason": "",
        "final_validation_generated_at": "",
        "final_email_sent": False,
        "final_email_sent_at": "",
        "timeframe": "1d",
        "timeframe_label": "Diario (1D)",
        "period_label": "2y",
        "trading_mode": VALIDATION_TRADING_MODE,
        "live_trading_enabled": VALIDATION_LIVE_TRADING_ENABLED,
        "paper_only": True,
        "last_evaluated_at": "",
        "last_reset_at": "",
        "last_report": {},
        "rejection_top_reason": "",
        "rejection_top_layer": "",
        "rejection_top_strategy": "",
        "rejection_reason_breakdown": {},
        "rejection_layer_breakdown": {},
        "rejection_strategy_breakdown": {},
        "rejection_has_minimum_sample": False,
        "signal_counters": {
            "signals_total": 0,
            "signals_approved": 0,
            "signals_rejected": 0,
            "entries_against_trend": 0,
            "rejections": {
                "trend_not_confirmed": 0,
                "score_below_minimum": 0,
                "reversal_not_eligible": 0,
                "volatility_out_of_range": 0,
                "no_setup_eligible": 0,
                "breakout_not_confirmed": 0,
                "confidence_too_low": 0,
                "feed_quality_blocked": 0,
                "fallback_blocked": 0,
                "provider_unknown": 0,
                "context_blocked": 0,
                "daily_loss_guard": 0,
                "cooldown_active": 0,
                "duplicate_signal_blocked": 0,
                "position_limit_reached": 0,
                "schedule_blocked": 0,
            },
            "assets_observed": {},
            "assets_approved": {},
            "context_status_counts": {
                "FAVORAVEL": 0,
                "NEUTRO": 0,
                "DESFAVORAVEL": 0,
                "CRITICO": 0,
            },
            "context_blocked_signals": 0,
        },
        "last_signal_keys": {},
        "last_rejection_event_keys": {},
    },
    "trader": {
        "enabled": True,
        "profile": DEFAULT_TRADER_PROFILE,
        "ticket_value": VALIDATION_DEFAULT_ENTRY_AMOUNT_BRL,
        "holding_minutes": 60,
        "max_open_positions": VALIDATION_DEFAULT_MAX_OPEN_POSITIONS,
        "watchlist": list(SWING_VALIDATION_RECOMMENDED_WATCHLIST),
    },
}


def _ensure_csv(file_path, columns: list[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        pd.DataFrame(columns=columns).to_csv(file_path, index=False)


def ensure_storage() -> None:
    ensure_app_directories()
    if database_enabled():
        load_json_state("bot_state", lambda: deepcopy(DEFAULT_STATE), BOT_STATE_FILE)
        return

    if not BOT_STATE_FILE.exists():
        save_bot_state(deepcopy(DEFAULT_STATE))

    _ensure_csv(TRADER_ORDERS_FILE, TRADER_ORDERS_COLUMNS)
    _ensure_csv(TRADER_REPORTS_FILE, TRADER_REPORTS_COLUMNS)
    _ensure_csv(BOT_LOG_FILE, BOT_LOG_COLUMNS)


def _merge_missing_keys(current: dict, default: dict) -> dict:
    for key, value in default.items():
        if key not in current:
            current[key] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(current.get(key), dict):
            current[key] = _merge_missing_keys(current[key], value)
    return current


def load_bot_state() -> dict:
    ensure_storage()
    state = load_json_state("bot_state", lambda: deepcopy(DEFAULT_STATE), BOT_STATE_FILE)
    state = _merge_missing_keys(state, deepcopy(DEFAULT_STATE))
    production_state = state.get("production", {}) or {}
    production_state["enabled"] = PRODUCTION_MODE
    production_state["alert_email_enabled"] = ALERT_EMAIL_ENABLED
    production_state["alert_provider"] = ALERT_EMAIL_PROVIDER
    state["production"] = production_state
    email_reporting_state = state.get("email_reporting", {}) or {}
    email_reporting_state["enabled"] = REPORT_EMAIL_ENABLED
    email_reporting_state["provider"] = REPORT_EMAIL_PROVIDER
    email_reporting_state["destination"] = REPORT_EMAIL_TO
    email_reporting_state["daily_enabled"] = REPORT_EMAIL_DAILY_ENABLED
    email_reporting_state["weekly_enabled"] = REPORT_EMAIL_WEEKLY_ENABLED
    email_reporting_state["ten_day_enabled"] = REPORT_EMAIL_10DAY_ENABLED
    email_reporting_state["final_enabled"] = REPORT_EMAIL_FINAL_ENABLED
    state["email_reporting"] = email_reporting_state
    risk_state = state.get("risk", {}) or {}
    risk_state["daily_loss_limit_brl"] = max(
        1.0,
        float(risk_state.get("daily_loss_limit_brl") or DAILY_LOSS_LIMIT_BRL_DEFAULT),
    )
    state["risk"] = risk_state
    retention_state = state.get("retention", {}) or {}
    retention_state["enabled"] = RETENTION_ENABLED
    retention_state["retention_days"] = RETENTION_DAYS
    retention_state["run_interval_hours"] = RETENTION_RUN_INTERVAL_HOURS
    retention_state["archive_trader_orders"] = RETENTION_ARCHIVE_TRADER_ORDERS
    state["retention"] = retention_state
    state = _normalize_market_data_state(state)
    state = _apply_trader_watchlist_defaults(state)
    return _apply_validation_operating_defaults(state)


def save_bot_state(state: dict) -> None:
    ensure_app_directories()
    save_json_state("bot_state", state, BOT_STATE_FILE)


def reset_state() -> dict:
    state = deepcopy(DEFAULT_STATE)
    save_bot_state(state)
    return state


def append_csv_row(file_path, row: dict) -> None:
    ensure_storage()
    columns_map = {
        str(TRADER_ORDERS_FILE): TRADER_ORDERS_COLUMNS,
        str(TRADER_REPORTS_FILE): TRADER_REPORTS_COLUMNS,
        str(BOT_LOG_FILE): BOT_LOG_COLUMNS,
    }
    append_table_row(file_path, row, columns=columns_map.get(str(file_path)))


def read_storage_table(file_path, columns: list[str] | None = None) -> pd.DataFrame:
    ensure_storage()
    return read_table(file_path, columns=columns)


def replace_storage_table(file_path, rows: list[dict], columns: list[str] | None = None) -> None:
    ensure_storage()
    replace_table(file_path, rows, columns=columns)


def log_event(level: str, message: str) -> None:
    append_csv_row(
        BOT_LOG_FILE,
        {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
        },
    )


def resolve_market_data_views(state: dict) -> tuple[dict, dict]:
    market_state = dict(state.get("market_data", {}) or {})
    contexts = market_state.get("contexts", {}) or {}
    operational_state = dict(contexts.get("worker_cycle") or market_state or {})
    chart_state = dict(contexts.get("trader_chart") or {})
    return operational_state, chart_state


def update_worker_heartbeat(status: str = "online") -> None:
    state = load_bot_state()
    state["worker_status"] = status
    state["worker_heartbeat"] = datetime.utcnow().isoformat()
    save_bot_state(state)


def persist_worker_cycle_state(
    *,
    last_action: str,
    next_run_delta_seconds: int,
    worker_status: str = "online",
    market_data_payload: dict | None = None,
    runtime_started_at: str = "",
    process_role: str = "worker",
) -> dict:
    state = load_bot_state()
    now = datetime.utcnow()
    iso_now = now.isoformat()
    state["last_action"] = str(last_action or "")
    state["last_run_at"] = iso_now
    state["next_run_at"] = (now + timedelta(seconds=int(next_run_delta_seconds))).isoformat()
    state["worker_status"] = str(worker_status or "online")
    state["worker_heartbeat"] = iso_now

    payload = dict(market_data_payload or {})
    payload.setdefault("requested_by", "worker_cycle")
    payload["build_active"] = str(payload.get("build_active") or MARKET_DATA_BUILD_LABEL)
    payload["git_sha"] = str(payload.get("git_sha") or RAILWAY_GIT_COMMIT_SHA or "")
    payload["build_timestamp"] = str(payload.get("build_timestamp") or BUILD_TIMESTAMP or "")
    payload["runtime_started_at"] = str(payload.get("runtime_started_at") or runtime_started_at or "")
    payload["service_name"] = str(payload.get("service_name") or SERVICE_NAME or "")
    payload["process_role"] = str(payload.get("process_role") or process_role or "worker")
    payload["state_writer"] = "worker"
    payload["state_written_at"] = iso_now
    payload["state_build_sha"] = str(payload.get("state_build_sha") or payload.get("git_sha") or "")
    payload["state_schema_version"] = str(payload.get("state_schema_version") or STATE_SCHEMA_VERSION)
    payload["ui_audit_probe"] = str(payload.get("ui_audit_probe") or "worker_state_v2")
    payload["provider_effective"] = str(payload.get("provider_effective") or payload.get("provider") or "")
    if payload.get("requested_symbols") is None and payload.get("symbols") is not None:
        payload["requested_symbols"] = list(payload.get("symbols") or [])

    if payload:
        update_market_data_status(payload, state=state, save=False)

    save_bot_state(state)
    return state


def update_market_data_status(
    status_payload: dict | None,
    *,
    state: dict | None = None,
    save: bool = True,
) -> dict:
    current_state = state or load_bot_state()
    state_written_at = datetime.utcnow().isoformat()
    default_service_name = str(SERVICE_NAME or "").strip() or str(current_state.get("service_name") or "")
    market_state = current_state.get("market_data", {}) or {}
    payload = status_payload or {}
    context_name = str(payload.get("requested_by") or "runtime")
    contexts = market_state.get("contexts", {}) or {}
    context_state = dict(contexts.get(context_name, {}) or {})
    previous_provider = str(context_state.get("provider") or "").strip().lower()

    if payload.get("provider"):
        context_state["provider"] = str(payload.get("provider"))
        context_state["provider_effective"] = str(payload.get("provider"))
    if payload.get("configured_provider"):
        context_state["configured_provider"] = str(payload.get("configured_provider"))
    if payload.get("fallback_provider"):
        context_state["fallback_provider"] = str(payload.get("fallback_provider"))
    if payload.get("provider_chain") is not None:
        context_state["provider_chain"] = [str(item) for item in (payload.get("provider_chain") or []) if str(item)]
    if isinstance(payload.get("provider_breakdown"), dict):
        context_state["provider_breakdown"] = dict(payload.get("provider_breakdown") or {})
    if isinstance(payload.get("provider_diagnostics"), dict):
        context_state["provider_diagnostics"] = dict(payload.get("provider_diagnostics") or {})
    normalized_legacy_status = legacy_market_status(
        status=payload.get("status"),
        last_source=payload.get("last_source") or context_state.get("last_source"),
        source_breakdown=payload.get("source_breakdown") or context_state.get("source_breakdown"),
    )
    normalized_feed_status = classify_feed_status(
        status=payload.get("feed_status") or payload.get("status") or context_state.get("feed_status"),
        last_source=payload.get("last_source") or context_state.get("last_source"),
        source_breakdown=payload.get("source_breakdown") or context_state.get("source_breakdown"),
    )
    context_state["status"] = normalized_legacy_status
    context_state["status_legacy"] = normalized_legacy_status
    context_state["feed_status"] = normalized_feed_status
    if payload.get("last_sync_at"):
        context_state["last_sync_at"] = str(payload.get("last_sync_at"))
    if payload.get("last_source"):
        context_state["last_source"] = str(payload.get("last_source"))
    if isinstance(payload.get("source_breakdown"), dict):
        context_state["source_breakdown"] = dict(payload.get("source_breakdown") or {})
    if payload.get("symbols") is not None:
        context_state["symbols"] = [str(symbol).upper() for symbol in (payload.get("symbols") or [])]
    if payload.get("requested_symbols") is not None:
        context_state["requested_symbols"] = [str(symbol).upper() for symbol in (payload.get("requested_symbols") or [])]
    for key in ("live_symbols", "cached_symbols", "fallback_symbols", "unknown_symbols"):
        if payload.get(key) is not None:
            context_state[key] = [str(symbol).upper() for symbol in (payload.get(key) or [])]
    for key in (
        "build_active",
        "git_sha",
        "build_timestamp",
        "runtime_started_at",
        "service_name",
        "process_role",
        "last_stage",
        "state_writer",
        "state_written_at",
        "state_build_sha",
        "state_schema_version",
        "ui_audit_probe",
    ):
        if payload.get(key) is not None:
            context_state[key] = str(payload.get(key) or "")
    for key in ("api_key_present", "request_prepared", "request_attempted", "response_received"):
        if payload.get(key) is not None:
            context_state[key] = bool(payload.get(key))
    if payload.get("response_status_code") is not None:
        context_state["response_status_code"] = payload.get("response_status_code")
    if payload.get("provider_effective") is not None:
        context_state["provider_effective"] = str(payload.get("provider_effective") or "")
    context_state["requested_by"] = context_name
    context_state["service_name"] = str(context_state.get("service_name") or default_service_name or "")
    context_state["state_schema_version"] = str(
        context_state.get("state_schema_version") or STATE_SCHEMA_VERSION
    )
    context_state["state_written_at"] = str(context_state.get("state_written_at") or state_written_at)
    context_state["state_writer"] = str(context_state.get("state_writer") or "")
    context_state["state_build_sha"] = str(context_state.get("state_build_sha") or context_state.get("git_sha") or "")

    source_breakdown = context_state.get("source_breakdown", {}) or {}
    if int(source_breakdown.get("market", 0) or 0) > 0 or int(source_breakdown.get("cached", 0) or 0) > 0:
        context_state["last_success_at"] = context_state.get("last_sync_at", "")
        context_state["last_error"] = ""
        context_state["fallback_since_at"] = ""
    elif payload.get("last_error"):
        context_state["last_error"] = str(payload.get("last_error"))
        context_state["fallback_since_at"] = str(
            context_state.get("fallback_since_at") or context_state.get("last_sync_at") or ""
        )

    if str(context_state.get("feed_status") or "UNKNOWN").upper() != "FALLBACK":
        context_state["fallback_since_at"] = ""

    contexts[context_name] = context_state
    market_state["contexts"] = contexts

    should_promote_to_top_level = context_name == "worker_cycle" or not market_state.get("requested_by")
    if should_promote_to_top_level:
        for key in (
            "provider",
            "provider_effective",
            "configured_provider",
            "fallback_provider",
            "provider_chain",
            "provider_breakdown",
            "provider_diagnostics",
            "status",
            "status_legacy",
            "feed_status",
            "last_sync_at",
            "last_success_at",
            "last_error",
            "last_source",
            "fallback_since_at",
            "source_breakdown",
            "symbols",
            "requested_symbols",
            "live_symbols",
            "cached_symbols",
            "fallback_symbols",
            "unknown_symbols",
            "requested_by",
            "build_active",
            "git_sha",
            "build_timestamp",
            "runtime_started_at",
            "service_name",
            "process_role",
            "api_key_present",
            "request_prepared",
            "request_attempted",
            "response_received",
            "response_status_code",
            "last_stage",
            "state_writer",
            "state_written_at",
            "state_build_sha",
            "state_schema_version",
            "ui_audit_probe",
        ):
            if key in context_state:
                market_state[key] = context_state.get(key)

    current_state["market_data"] = market_state
    if save:
        save_bot_state(current_state)

    updated_provider = str(context_state.get("provider") or "").strip().lower()
    if previous_provider and updated_provider and previous_provider != updated_provider:
        provider_labels = {
            "twelvedata": "Twelve Data",
            "yahoo": "Yahoo",
            "synthetic": "Fallback sintetico",
            "mixed": "Twelve Data + Yahoo",
        }
        source_label = str(context_state.get("last_source") or "unknown").strip().lower()
        log_event(
            "INFO",
            (
                f"Provider de dados alterado em {context_name}: "
                f"{provider_labels.get(previous_provider, previous_provider.upper())} -> "
                f"{provider_labels.get(updated_provider, updated_provider.upper())} "
                f"(fonte: {source_label})."
            ),
        )

    td_diag = dict((context_state.get("provider_diagnostics", {}) or {}).get("twelvedata") or {})
    configured_provider = str(context_state.get("configured_provider") or "").strip().lower()
    if context_name == "worker_cycle" and configured_provider == "twelvedata" and td_diag:
        should_log_diag = str(context_state.get("feed_status") or "").strip().upper() == "FALLBACK" or bool(
            td_diag.get("used_live_data")
        )
        if should_log_diag:
            log_event(
                "INFO",
                (
                    "[twelvedata_request_diag] "
                    f"build={str(td_diag.get('build_label') or 'unknown')};"
                    f"service={str(td_diag.get('service_name') or 'unknown')};"
                    f"key_present={1 if bool(td_diag.get('api_key_present')) else 0};"
                    f"key_length={int(td_diag.get('api_key_length') or 0)};"
                    f"request_built={1 if bool(td_diag.get('request_built')) else 0};"
                    f"request_attempted={1 if bool(td_diag.get('request_attempted')) else 0};"
                    f"response_received={1 if bool(td_diag.get('response_received')) else 0};"
                    f"success_count={int(td_diag.get('success_count') or 0)};"
                    f"stage={str(td_diag.get('last_stage') or 'unknown')};"
                    f"host={str(td_diag.get('api_base_host') or td_diag.get('request_host') or 'unknown')};"
                    f"symbol={str(td_diag.get('sample_normalized_symbol') or td_diag.get('sample_symbol') or 'unknown')};"
                    f"error={str(td_diag.get('last_error') or 'none')}"
                ),
            )

    return context_state


def update_broker_status(status_payload: dict | None) -> dict:
    state = load_bot_state()
    broker_state = state.get("broker", {}) or {}
    payload = status_payload or {}

    for key in (
        "provider",
        "mode",
        "status",
        "last_sync_at",
        "last_error",
        "account_id",
        "requested_by",
        "configured_mode",
        "effective_mode",
        "base_url",
        "api_key_configured",
        "api_secret_configured",
        "execution_enabled",
        "can_submit_orders",
        "warning",
    ):
        if payload.get(key) is not None:
            broker_state[key] = payload.get(key)

    state["broker"] = broker_state
    save_bot_state(state)
    return broker_state


def update_market_context_status(status_payload: dict | None) -> dict:
    state = load_bot_state()
    market_context_state = state.get("market_context", {}) or {}
    payload = status_payload or {}

    for key, value in payload.items():
        if value is not None:
            market_context_state[key] = value

    state["market_context"] = market_context_state
    save_bot_state(state)
    return market_context_state


def update_production_status(status_payload: dict | None) -> dict:
    state = load_bot_state()
    production_state = state.get("production", {}) or {}
    payload = status_payload or {}

    for key, value in payload.items():
        if value is not None:
            production_state[key] = value

    state["production"] = production_state
    save_bot_state(state)
    return production_state


def update_email_reporting_status(status_payload: dict | None) -> dict:
    state = load_bot_state()
    email_reporting_state = state.get("email_reporting", {}) or {}
    payload = status_payload or {}

    for key, value in payload.items():
        if value is not None:
            email_reporting_state[key] = value

    state["email_reporting"] = email_reporting_state
    save_bot_state(state)
    return email_reporting_state


def update_risk_status(status_payload: dict | None) -> dict:
    state = load_bot_state()
    risk_state = state.get("risk", {}) or {}
    payload = status_payload or {}

    for key, value in payload.items():
        if value is not None:
            risk_state[key] = value

    risk_state["daily_loss_limit_brl"] = max(
        1.0,
        float(risk_state.get("daily_loss_limit_brl") or DAILY_LOSS_LIMIT_BRL_DEFAULT),
    )
    state["risk"] = risk_state
    save_bot_state(state)
    return risk_state


def update_retention_status(status_payload: dict | None) -> dict:
    state = load_bot_state()
    retention_state = state.get("retention", {}) or {}
    payload = status_payload or {}

    for key, value in payload.items():
        if value is not None:
            retention_state[key] = value

    state["retention"] = retention_state
    save_bot_state(state)
    return retention_state


def update_validation_status(status_payload: dict | None) -> dict:
    state = load_bot_state()
    validation_state = state.get("validation", {}) or {}
    payload = status_payload or {}

    for key, value in payload.items():
        if value is not None:
            validation_state[key] = value

    state["validation"] = validation_state
    save_bot_state(state)
    return validation_state
