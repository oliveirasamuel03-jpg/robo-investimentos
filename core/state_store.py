from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta

import pandas as pd

from core.no_setup_eligible_decomposition import default_no_setup_eligible_decomposition_state
from core.config import (
    ALERT_EMAIL_ENABLED,
    ALERT_EMAIL_PROVIDER,
    APP_SOURCE_COMMIT_SHA,
    BOS_PIVOT_TRACE_AUDIT_ENABLED,
    BOS_PIVOT_TRACE_SHADOW_ONLY,
    BUILD_TIMESTAMP,
    BOT_LOG_COLUMNS,
    BOT_LOG_FILE,
    BOT_STATE_FILE,
    BROKER_MODE,
    BROKER_PROVIDER,
    CALIBRATION_PREVIEW_ENABLED,
    DAILY_LOSS_LIMIT_BRL_DEFAULT,
    EXTERNAL_SIGNAL_ALLOWED_SOURCES,
    EXTERNAL_SIGNAL_ALLOWED_TIMEFRAMES,
    EXTERNAL_SIGNAL_DEDUPE_SECONDS,
    EXTERNAL_SIGNAL_MAX_AGE_SECONDS,
    EXTERNAL_SIGNAL_SECRET,
    EXTERNAL_SIGNAL_TEST_PANEL_ENABLED,
    EXTERNAL_SIGNAL_WEBHOOK_ENABLED,
    MARKET_DATA_FALLBACK_PROVIDER,
    MARKET_DATA_BUILD_LABEL,
    MARKET_DATA_PROVIDER,
    MULTITF_INTRADAY_CACHE_TTL_SECONDS,
    MULTITF_INTRADAY_FETCH_ENABLED,
    MULTITF_INTRADAY_MAX_CALLS_PER_CYCLE,
    MULTITF_INTRADAY_MAX_SYMBOLS,
    MULTITF_INTRADAY_PROVIDER_BUDGET_MODE,
    MULTITF_INTRADAY_REQUIRE_LIVE_FEED,
    MULTITF_INTRADAY_SHADOW_ONLY,
    MULTITF_INTRADAY_TIMEFRAMES,
    MULTITF_SWING_AUDIT_ENABLED,
    MULTITF_SWING_CACHE_TTL_SECONDS,
    MULTITF_SWING_MAX_SYMBOLS,
    MULTITF_SWING_REQUIRE_LIVE_FEED,
    MULTITF_SWING_SHADOW_ONLY,
    MULTITF_SWING_TIMEFRAMES,
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
    STRATEGY_DECISION_BRIDGE_TRACE_ENABLED,
    STRATEGY_DECISION_BRIDGE_TRACE_SHADOW_ONLY,
    RETENTION_ARCHIVE_TRADER_ORDERS,
    RETENTION_DAYS,
    RETENTION_ENABLED,
    RETENTION_RUN_INTERVAL_HOURS,
    LEGACY_MIXED_DEFAULT_WATCHLIST,
    LEGACY_VALIDATION_INITIAL_CAPITAL_BRL,
    MACRO_ALERT_EVENT_WINDOW_MINUTES,
    MACRO_ALERT_POST_EVENT_MINUTES,
    MACRO_ALERT_PRE_EVENT_MINUTES,
    MACRO_ALERTS_ENABLED,
    MACRO_ALERTS_FILE,
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
        normalized["source_commit_sha"] = str(normalized.get("source_commit_sha") or "")
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
        normalized["requested_interval"] = str(normalized.get("requested_interval") or "")
        normalized["effective_interval"] = str(normalized.get("effective_interval") or "")
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
        "requested_interval": "",
        "effective_interval": "",
        "live_symbols": [],
        "cached_symbols": [],
        "fallback_symbols": [],
        "unknown_symbols": [],
        "requested_by": "",
        "build_active": "",
        "git_sha": "",
        "source_commit_sha": "",
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
    "macro_alert": {
        "macro_alert_enabled": MACRO_ALERTS_ENABLED,
        "macro_alert_active": False,
        "macro_alert_level": "LOW",
        "macro_alert_currency": "",
        "macro_alert_title": "",
        "macro_alert_time": "",
        "macro_alert_window_status": "INACTIVE",
        "macro_alert_minutes_to_event": None,
        "macro_alert_reason": "Nenhum evento macro ativo.",
        "macro_alert_penalty": 0.0,
        "macro_alert_blocks_new_entries": False,
        "macro_alert_last_update_ts": "",
        "macro_alert_event_count": 0,
        "macro_alert_source": "none",
        "macro_alerts_file": MACRO_ALERTS_FILE,
        "macro_alert_pre_event_minutes": MACRO_ALERT_PRE_EVENT_MINUTES,
        "macro_alert_event_window_minutes": MACRO_ALERT_EVENT_WINDOW_MINUTES,
        "macro_alert_post_event_minutes": MACRO_ALERT_POST_EVENT_MINUTES,
    },
    "external_signal": {
        "enabled": EXTERNAL_SIGNAL_WEBHOOK_ENABLED,
        "webhook_configured": False,
        "allowed_sources": EXTERNAL_SIGNAL_ALLOWED_SOURCES,
        "allowed_timeframes": EXTERNAL_SIGNAL_ALLOWED_TIMEFRAMES,
        "max_age_seconds": EXTERNAL_SIGNAL_MAX_AGE_SECONDS,
        "dedupe_seconds": EXTERNAL_SIGNAL_DEDUPE_SECONDS,
        "test_panel_enabled": EXTERNAL_SIGNAL_TEST_PANEL_ENABLED,
        "audit_only": True,
        "last_ts": None,
        "last_received_at": None,
        "last_source": "",
        "last_strategy": "",
        "last_symbol": "",
        "last_side": "",
        "last_timeframe": "",
        "last_alert_price": None,
        "last_score": 0.0,
        "last_status": "DISABLED",
        "last_reason": "External signal webhook disabled.",
        "last_dedupe_key": "",
        "recent_events": [],
    },
    "calibration_preview": {
        "enabled": CALIBRATION_PREVIEW_ENABLED,
        "mode": "PREVIEW_ONLY",
        "near_approved_count": 0,
        "near_approved_rate": 0.0,
        "min_score_current": None,
        "preview_score_floor": None,
        "avg_score_gap": None,
        "best_score_seen": None,
        "top_asset": "",
        "top_setup": "",
        "safe_conditions_met_count": 0,
        "unsafe_conditions_count": 0,
        "recommendation": "observe_more",
        "reason": "No calibration preview data yet.",
        "near_approved_examples": [],
    },
    "strategy_bottleneck": {
        "enabled": True,
        "mode": "DIAGNOSTIC_ONLY",
        "dominant_bottleneck": "",
        "dominant_setup": "",
        "dominant_asset": "",
        "total_strategy_rejections": 0,
        "score_below_min_count": 0,
        "momentum_weak_count": 0,
        "secondary_confirmation_weak_count": 0,
        "rsi_out_of_range_count": 0,
        "trend_not_confirmed_count": 0,
        "volatility_filter_count": 0,
        "context_filter_count": 0,
        "feed_block_count": 0,
        "guard_block_count": 0,
        "unknown_count": 0,
        "top_assets_blocked": [],
        "top_setups_blocked": [],
        "top_filter_reasons": [],
        "closest_candidates": [],
        "recommendation": "observe_more",
        "reason": "No strategy bottleneck data yet.",
    },
    "strategy_structure_audit": {
        "structural_audit_enabled": True,
        "structural_audit_mode": "SHADOW_ONLY",
        "structural_audit_last_run_at": "",
        "structural_audit_candidates": 0,
        "structural_audit_top_setup": "",
        "structural_audit_top_symbol": "",
        "structural_audit_top_score": None,
        "structural_audit_top_gap": None,
        "structural_audit_primary_blocker": "",
        "structural_audit_secondary_blocker": "",
        "structural_audit_recommendation": "sem dados suficientes",
        "structural_audit_should_adjust_strategy": False,
        "structural_audit_setup_comparison": [],
        "structural_audit_total_candidates_by_setup": {},
        "structural_audit_near_candidates_by_setup": {},
        "structural_audit_primary_blockers_by_setup": {},
        "structural_audit_secondary_blockers_by_setup": {},
        "structural_audit_average_score_by_setup": {},
        "structural_audit_average_gap_by_setup": {},
        "structural_audit_best_candidate_by_setup": {},
        "structural_audit_timeframe_note": "Inconclusivo: sem amostra estrutural suficiente.",
        "structural_audit_rsi_momentum_note": "Inconclusivo: sem amostra estrutural suficiente.",
        "structural_audit_reversal_note": "Inconclusivo: sem amostra estrutural suficiente.",
        "structural_audit_recent_candidates": [],
        "structural_audit_reason": "No structural audit data yet.",
    },
    "market_structure_audit": {
        "market_structure_audit_enabled": True,
        "market_structure_audit_mode": "SHADOW_ONLY",
        "market_structure_audit_last_run_at": "",
        "market_structure_top_symbol": "",
        "market_structure_top_score": None,
        "market_structure_top_zone": "",
        "market_structure_top_recommendation": "sem dados suficientes",
        "market_structure_candidates_count": 0,
        "market_structure_best_candidates": [],
        "market_structure_setup_confluence": {},
        "market_structure_fib_summary": {},
        "market_structure_blockers_summary": {},
        "market_structure_regime_summary": {},
        "market_structure_data_sufficiency": "NO_DATA",
        "market_structure_minimum_sample_met": False,
        "market_structure_why_no_candidate": "No market structure audit data yet.",
    },
    "fib_alignment_audit": {
        "fib_alignment_enabled": True,
        "fib_alignment_mode": "SHADOW_ONLY",
        "fib_alignment_source": "video_pdf_inspired_checklist_v1",
        "fib_alignment_score": None,
        "fib_alignment_status": "insufficient_data",
        "fib_alignment_top_symbol": "",
        "fib_alignment_anchor_low_status": "insufficient",
        "fib_alignment_anchor_high_status": "insufficient",
        "fib_alignment_zone_status": "insufficient",
        "fib_alignment_pivot_status": "insufficient",
        "fib_alignment_bos_status": "insufficient",
        "fib_alignment_entry_confirmation_status": "insufficient",
        "fib_alignment_confluence_status": "insufficient",
        "fib_alignment_missing_evidence": ["no_data"],
        "fib_alignment_why_differs": "No Fibonacci video/PDF alignment audit data yet.",
        "fib_alignment_recommendation": "insufficient_data",
        "fib_alignment_checklist": [],
        "fib_alignment_last_run_at": "",
    },
    "multi_timeframe_intraday_fetcher": {
        "enabled": MULTITF_INTRADAY_FETCH_ENABLED,
        "mode": "SHADOW_ONLY",
        "generated_at": "",
        "provider_effective": "",
        "feed_status": "UNKNOWN",
        "timeframes_requested": [item.strip().lower() for item in MULTITF_INTRADAY_TIMEFRAMES.split(",") if item.strip()],
        "timeframes_available": [],
        "symbols_requested": [],
        "symbols_fetched": [],
        "cache_hits": 0,
        "cache_misses": 0,
        "provider_calls_attempted": 0,
        "provider_calls_skipped": 0,
        "provider_budget_guard_active": False,
        "provider_guard_reason": "No multi-timeframe intraday fetch data yet.",
        "estimated_provider_calls": 0,
        "last_success_at": "",
        "last_error": "",
        "intraday_data_quality": "NO_DATA",
        "intraday_fetch_recommendation": "observe_more",
        "diagnostics": [],
        "cache_ttl_seconds": MULTITF_INTRADAY_CACHE_TTL_SECONDS,
        "max_symbols": MULTITF_INTRADAY_MAX_SYMBOLS,
        "max_calls_per_cycle": MULTITF_INTRADAY_MAX_CALLS_PER_CYCLE,
        "require_live_feed": MULTITF_INTRADAY_REQUIRE_LIVE_FEED,
        "provider_budget_mode": MULTITF_INTRADAY_PROVIDER_BUDGET_MODE,
        "shadow_only": MULTITF_INTRADAY_SHADOW_ONLY,
    },
    "multi_timeframe_swing_audit": {
        "enabled": MULTITF_SWING_AUDIT_ENABLED,
        "mode": "SHADOW_ONLY",
        "generated_at": "",
        "provider_effective": "",
        "feed_status": "UNKNOWN",
        "timeframes_used": [item.strip().lower() for item in MULTITF_SWING_TIMEFRAMES.split(",") if item.strip()],
        "timeframe_source": "operational_cycle_resample",
        "timeframe_fallbacks": [],
        "symbols_analyzed": 0,
        "top_symbol": "",
        "top_alignment_score": None,
        "top_alignment_status": "INSUFFICIENT_DATA",
        "top_missing_confirmation": "",
        "top_recommendation": "insufficient_data",
        "dominant_conflict_reason": "No multi-timeframe swing audit data yet.",
        "candidates_count": 0,
        "strong_alignment_count": 0,
        "partial_alignment_count": 0,
        "conflict_count": 0,
        "insufficient_data_count": 0,
        "setup_support_count": 0,
        "recent_candidates": [],
        "estimated_provider_calls": 0,
        "cache_ttl_seconds": MULTITF_SWING_CACHE_TTL_SECONDS,
        "cache_status": "cycle_data_resample_only",
        "provider_guard": "not_evaluated",
        "require_live_feed": MULTITF_SWING_REQUIRE_LIVE_FEED,
        "shadow_only": MULTITF_SWING_SHADOW_ONLY,
        "max_symbols": MULTITF_SWING_MAX_SYMBOLS,
        "uses_real_intraday_data": False,
        "intraday_timeframes_available": [],
        "intraday_top_symbol": "",
        "intraday_missing_reason": "",
        "h4_data_quality": "missing",
        "h1_data_quality": "missing",
        "reason": "No multi-timeframe swing audit data yet.",
    },
    "bos_pivot_trace_audit": {
        "enabled": BOS_PIVOT_TRACE_AUDIT_ENABLED,
        "mode": "SHADOW_ONLY",
        "generated_at": "",
        "provider_effective": "",
        "feed_status": "UNKNOWN",
        "uses_real_intraday_data": False,
        "symbols_analyzed": 0,
        "timeframes_analyzed": ["4h", "1h"],
        "top_symbol": "",
        "top_timeframe": "",
        "top_pivot_state": "INSUFFICIENT_DATA",
        "top_bos_state": "INSUFFICIENT_DATA",
        "top_h4_bos_state": "INSUFFICIENT_DATA",
        "top_h1_bos_state": "INSUFFICIENT_DATA",
        "top_relationship": "INSUFFICIENT_DATA",
        "top_recommendation": "insufficient_data",
        "top_primary_missing_piece": "No BOS/Pivot trace audit data yet.",
        "dominant_missing_piece": "No BOS/Pivot trace audit data yet.",
        "h4_bos_missing_count": 0,
        "h1_bos_only_count": 0,
        "wick_only_bos_count": 0,
        "weak_close_bos_count": 0,
        "confirmed_bos_count": 0,
        "retest_pending_count": 0,
        "pivot_forming_count": 0,
        "pivot_confirmed_count": 0,
        "pivot_triggered_count": 0,
        "insufficient_data_count": 0,
        "should_keep_blocked_count": 0,
        "recent_candidates": [],
        "reason": "No BOS/Pivot trace audit data yet.",
        "shadow_only": BOS_PIVOT_TRACE_SHADOW_ONLY,
    },
    "strategy_decision_bridge_trace": {
        "enabled": STRATEGY_DECISION_BRIDGE_TRACE_ENABLED,
        "mode": "SHADOW_ONLY",
        "generated_at": "",
        "provider_effective": "",
        "feed_status": "UNKNOWN",
        "symbols_analyzed": 0,
        "top_symbol": "",
        "top_bridge_status": "INSUFFICIENT_TRACE_DATA",
        "top_real_blocker": "",
        "top_structure_status": "",
        "top_reconciliation_status": "UNKNOWN_MISMATCH",
        "fallback_scope_status": "UNKNOWN_SCOPE",
        "fallback_blocker_scope": "UNKNOWN",
        "current_feed_is_clean": False,
        "structure_confirmed_but_blocked_count": 0,
        "fallback_scope_mismatch_count": 0,
        "multi_tf_vs_bos_mismatch_count": 0,
        "real_strategy_authority_count": 0,
        "should_keep_blocked_count": 0,
        "recommendation": "observe_more",
        "recent_candidates": [],
        "reason": "No strategy decision bridge trace data yet.",
        "shadow_only": STRATEGY_DECISION_BRIDGE_TRACE_SHADOW_ONLY,
    },
    "feed_scope_reconciliation": {
        "enabled": True,
        "mode": "DIAGNOSTIC_ONLY",
        "generated_at": "",
        "provider_effective": "",
        "current_feed_status": "UNKNOWN",
        "current_cycle_feed_status": "UNKNOWN",
        "current_cycle_provider": "",
        "current_live_count": 0,
        "current_cycle_live_count": 0,
        "current_fallback_count": 0,
        "current_cycle_fallback_count": 0,
        "current_cycle_unknown_count": 0,
        "visual_feed_status": "UNKNOWN",
        "visual_chart_feed_status": "UNKNOWN",
        "worker_feed_status": "UNKNOWN",
        "accumulated_fallback_count": 0,
        "accumulated_strategy_count": 0,
        "historical_fallback_count": 0,
        "candidate_fallback_flags": {},
        "dominant_rejection_current": "",
        "dominant_rejection_accumulated": "",
        "fallback_scope_status": "UNKNOWN_SCOPE",
        "fallback_blocker_scope": "UNKNOWN",
        "current_feed_is_clean": False,
        "recommendation": "observe_more",
        "notes": "No feed scope reconciliation data yet.",
    },
    "no_setup_eligible_decomposition": default_no_setup_eligible_decomposition_state(),
    "shadow_decision_simulator": {
        "shadow_decision_simulator_enabled": True,
        "shadow_decision_mode": "SHADOW_ONLY",
        "shadow_entry_policy": "conservative_v1",
        "shadow_decision_last_run_at": "",
        "preview_near_approved_count": 0,
        "shadow_candidates_received_count": 0,
        "shadow_candidates_unique_count": 0,
        "shadow_candidates_ignored_count": 0,
        "shadow_candidates_classified_count": 0,
        "shadow_candidates_analyzed_count": 0,
        "shadow_raw_near_approved_count": 0,
        "shadow_counts_scope": "current_cycle_and_accumulated_recent",
        "shadow_current_cycle_candidates_count": 0,
        "shadow_accumulated_candidates_count": 0,
        "shadow_current_cycle_received_count": 0,
        "shadow_current_cycle_new_unique_count": 0,
        "shadow_current_cycle_duplicate_count": 0,
        "shadow_current_cycle_already_analyzed_count": 0,
        "shadow_current_cycle_analyzed_count": 0,
        "shadow_current_cycle_analyzed_new_count": 0,
        "shadow_current_cycle_classified_count": 0,
        "shadow_current_cycle_classified_new_count": 0,
        "shadow_current_cycle_raw_near_approved_count": 0,
        "shadow_current_cycle_safe_near_approved_count": 0,
        "shadow_current_cycle_marginal_near_approved_count": 0,
        "shadow_current_cycle_unsafe_count": 0,
        "shadow_current_cycle_unsafe_new_count": 0,
        "shadow_current_cycle_ignored_count": 0,
        "shadow_current_cycle_primary_blocked_count": 0,
        "shadow_current_cycle_primary_blocked_new_count": 0,
        "shadow_current_cycle_secondary_blocked_count": 0,
        "shadow_current_cycle_secondary_blocked_new_count": 0,
        "shadow_accumulated_received_count": 0,
        "shadow_accumulated_raw_received_count": 0,
        "shadow_accumulated_unique_candidates_count": 0,
        "shadow_accumulated_analyzed_count": 0,
        "shadow_accumulated_analyzed_unique_count": 0,
        "shadow_accumulated_classified_unique_count": 0,
        "shadow_accumulated_raw_near_approved_count": 0,
        "shadow_accumulated_unsafe_count": 0,
        "shadow_accumulated_unsafe_unique_count": 0,
        "shadow_accumulated_primary_blocked_count": 0,
        "shadow_accumulated_secondary_blocked_count": 0,
        "shadow_raw_to_unique_ratio": 0.0,
        "shadow_duplicate_ratio": 0.0,
        "shadow_counter_health_status": "healthy",
        "shadow_near_approved_count": 0,
        "shadow_safe_near_approved_count": 0,
        "shadow_marginal_near_approved_count": 0,
        "shadow_marginal_count": 0,
        "shadow_unsafe_count": 0,
        "shadow_unsafe_rejection_count": 0,
        "shadow_primary_blocked_count": 0,
        "shadow_secondary_blocked_count": 0,
        "shadow_structure_missing_count": 0,
        "shadow_confirmation_missing_count": 0,
        "shadow_ignored_count": 0,
        "shadow_ignored_reason": "",
        "shadow_counter_warning": False,
        "shadow_counter_warning_reason": "",
        "shadow_scope_warning": False,
        "shadow_scope_warning_reason": "",
        "shadow_would_enter_count": 0,
        "shadow_pending_count": 0,
        "shadow_would_win_count": 0,
        "shadow_would_lose_count": 0,
        "shadow_best_symbol": "",
        "shadow_best_strategy": "",
        "shadow_best_candidate_score": None,
        "shadow_dominant_block_reason": "",
        "shadow_dominant_block_reason_current": "",
        "shadow_dominant_block_reason_accumulated": "",
        "dominant_exclusion_current_scope": "",
        "dominant_exclusion_accumulated_scope": "",
        "fallback_blocker_scope": "UNKNOWN",
        "fallback_current_count": 0,
        "fallback_accumulated_count": 0,
        "fallback_scope_status": "UNKNOWN_SCOPE",
        "fallback_scope_note": "",
        "shadow_policy_recommendation": "observe_more",
        "shadow_current_cycle_candidates": [],
        "shadow_accumulated_recent_candidates": [],
        "shadow_recent_candidates": [],
        "shadow_outcome_summary": {},
        "shadow_reason": "No shadow decision simulator data yet.",
        "shadow_stop_pct": 0.025,
        "shadow_take_profit_pct": 0.04,
        "shadow_max_hold_cycles": 24,
    },
    "phase2_fine_tune": {
        "fine_tune_enabled": True,
        "fine_tune_reason": "Relaxamento conservador de confirmacao secundaria marginal em PAPER.",
        "fine_tune_target": "trend_pullback_breakout_secondary_breakout_confirmation",
        "fine_tune_before": "breakout_20 >= breakout_min",
        "fine_tune_after": "breakout_20 >= breakout_min - 0.005, apenas com score minimo preservado e guards seguros",
        "fine_tune_applied_count": 0,
        "fine_tune_blocked_count": 0,
        "fine_tune_last_guard_reason": "",
    },
    "phase2_1_fine_tune": {
        "phase2_1_fine_tune_enabled": True,
        "phase2_1_fine_tune_target": "trend_pullback_breakout_multi_minor_confirmation",
        "phase2_1_fine_tune_reason": "Relaxamento conservador de multiplas falhas pequenas de momentum/confirmacao secundaria em PAPER.",
        "phase2_1_fine_tune_applied_count": 0,
        "phase2_1_fine_tune_blocked_count": 0,
        "phase2_1_fine_tune_last_guard": "",
        "phase2_1_fine_tune_last_decision": "",
        "phase2_1_fine_tune_score_gap": None,
        "phase2_1_fine_tune_allowed_reasons": [],
        "phase2_1_fine_tune_blocked_reasons": [],
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
        "last_cycle_rejection_summary": {},
        "current_cycle_rejection_reason": "",
        "accumulated_rejection_reason": "",
        "fallback_rejection_current_cycle_count": 0,
        "fallback_rejection_accumulated_count": 0,
        "strategy_rejection_current_cycle_count": 0,
        "strategy_rejection_accumulated_count": 0,
        "guard_rejection_current_cycle_count": 0,
        "guard_rejection_accumulated_count": 0,
        "feed_rejection_consistency": {},
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
                "macro_alert_guard": 0,
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
    macro_state = state.get("macro_alert", {}) or {}
    macro_state["macro_alert_enabled"] = MACRO_ALERTS_ENABLED
    macro_state["macro_alerts_file"] = MACRO_ALERTS_FILE
    macro_state["macro_alert_pre_event_minutes"] = MACRO_ALERT_PRE_EVENT_MINUTES
    macro_state["macro_alert_event_window_minutes"] = MACRO_ALERT_EVENT_WINDOW_MINUTES
    macro_state["macro_alert_post_event_minutes"] = MACRO_ALERT_POST_EVENT_MINUTES
    state["macro_alert"] = macro_state
    external_signal_state = state.get("external_signal", {}) or {}
    external_signal_state["enabled"] = EXTERNAL_SIGNAL_WEBHOOK_ENABLED
    external_signal_state["webhook_configured"] = bool(
        EXTERNAL_SIGNAL_WEBHOOK_ENABLED and EXTERNAL_SIGNAL_SECRET and EXTERNAL_SIGNAL_ALLOWED_SOURCES
    )
    external_signal_state["allowed_sources"] = EXTERNAL_SIGNAL_ALLOWED_SOURCES
    external_signal_state["allowed_timeframes"] = EXTERNAL_SIGNAL_ALLOWED_TIMEFRAMES
    external_signal_state["max_age_seconds"] = EXTERNAL_SIGNAL_MAX_AGE_SECONDS
    external_signal_state["dedupe_seconds"] = EXTERNAL_SIGNAL_DEDUPE_SECONDS
    external_signal_state["test_panel_enabled"] = EXTERNAL_SIGNAL_TEST_PANEL_ENABLED
    external_signal_state["audit_only"] = True
    external_signal_state["recent_events"] = [
        event
        for event in list(external_signal_state.get("recent_events", []) or [])
        if isinstance(event, dict)
    ][-20:]
    if not external_signal_state.get("last_status"):
        external_signal_state["last_status"] = "DISABLED"
    if not external_signal_state.get("last_reason"):
        external_signal_state["last_reason"] = (
            "External signal webhook disabled."
            if not EXTERNAL_SIGNAL_WEBHOOK_ENABLED
            else "No external signal received yet."
        )
    state["external_signal"] = external_signal_state
    calibration_preview_state = state.get("calibration_preview", {}) or {}
    calibration_preview_state["enabled"] = CALIBRATION_PREVIEW_ENABLED
    calibration_preview_state["mode"] = str(calibration_preview_state.get("mode") or "PREVIEW_ONLY")
    calibration_preview_state["near_approved_examples"] = [
        event
        for event in list(calibration_preview_state.get("near_approved_examples", []) or [])
        if isinstance(event, dict)
    ][-20:]
    if not calibration_preview_state.get("recommendation"):
        calibration_preview_state["recommendation"] = "observe_more"
    if not calibration_preview_state.get("reason"):
        calibration_preview_state["reason"] = "No calibration preview data yet."
    state["calibration_preview"] = calibration_preview_state
    strategy_bottleneck_state = state.get("strategy_bottleneck", {}) or {}
    strategy_bottleneck_state["enabled"] = bool(strategy_bottleneck_state.get("enabled", True))
    strategy_bottleneck_state["mode"] = str(strategy_bottleneck_state.get("mode") or "DIAGNOSTIC_ONLY")
    for key in ("top_assets_blocked", "top_setups_blocked", "top_filter_reasons"):
        strategy_bottleneck_state[key] = [
            item
            for item in list(strategy_bottleneck_state.get(key, []) or [])
            if isinstance(item, dict)
        ][:5]
    strategy_bottleneck_state["closest_candidates"] = [
        item
        for item in list(strategy_bottleneck_state.get("closest_candidates", []) or [])
        if isinstance(item, dict)
    ][:10]
    if not strategy_bottleneck_state.get("recommendation"):
        strategy_bottleneck_state["recommendation"] = "observe_more"
    if not strategy_bottleneck_state.get("reason"):
        strategy_bottleneck_state["reason"] = "No strategy bottleneck data yet."
    state["strategy_bottleneck"] = strategy_bottleneck_state
    strategy_structure_state = state.get("strategy_structure_audit", {}) or {}
    strategy_structure_state["structural_audit_enabled"] = bool(
        strategy_structure_state.get("structural_audit_enabled", True)
    )
    strategy_structure_state["structural_audit_mode"] = str(
        strategy_structure_state.get("structural_audit_mode") or "SHADOW_ONLY"
    )
    for key, fallback in (
        ("structural_audit_last_run_at", ""),
        ("structural_audit_top_setup", ""),
        ("structural_audit_top_symbol", ""),
        ("structural_audit_primary_blocker", ""),
        ("structural_audit_secondary_blocker", ""),
        ("structural_audit_recommendation", "sem dados suficientes"),
        ("structural_audit_timeframe_note", "Inconclusivo: sem amostra estrutural suficiente."),
        ("structural_audit_rsi_momentum_note", "Inconclusivo: sem amostra estrutural suficiente."),
        ("structural_audit_reversal_note", "Inconclusivo: sem amostra estrutural suficiente."),
        ("structural_audit_reason", "No structural audit data yet."),
    ):
        strategy_structure_state[key] = str(strategy_structure_state.get(key) or fallback)
    strategy_structure_state["structural_audit_candidates"] = int(
        strategy_structure_state.get("structural_audit_candidates", 0) or 0
    )
    strategy_structure_state["structural_audit_should_adjust_strategy"] = bool(
        strategy_structure_state.get("structural_audit_should_adjust_strategy", False)
    )
    for key in ("structural_audit_top_score", "structural_audit_top_gap"):
        value = strategy_structure_state.get(key)
        try:
            strategy_structure_state[key] = None if value in (None, "") else float(value or 0.0)
        except (TypeError, ValueError):
            strategy_structure_state[key] = None
    strategy_structure_state["structural_audit_setup_comparison"] = [
        item
        for item in list(strategy_structure_state.get("structural_audit_setup_comparison", []) or [])
        if isinstance(item, dict)
    ][:10]
    for key in (
        "structural_audit_total_candidates_by_setup",
        "structural_audit_near_candidates_by_setup",
        "structural_audit_primary_blockers_by_setup",
        "structural_audit_secondary_blockers_by_setup",
        "structural_audit_average_score_by_setup",
        "structural_audit_average_gap_by_setup",
        "structural_audit_best_candidate_by_setup",
    ):
        value = strategy_structure_state.get(key, {}) or {}
        strategy_structure_state[key] = dict(value) if isinstance(value, dict) else {}
    strategy_structure_state["structural_audit_recent_candidates"] = [
        item
        for item in list(strategy_structure_state.get("structural_audit_recent_candidates", []) or [])
        if isinstance(item, dict)
    ][:15]
    state["strategy_structure_audit"] = strategy_structure_state
    market_structure_state = state.get("market_structure_audit", {}) or {}
    market_structure_state["market_structure_audit_enabled"] = bool(
        market_structure_state.get("market_structure_audit_enabled", True)
    )
    market_structure_state["market_structure_audit_mode"] = str(
        market_structure_state.get("market_structure_audit_mode") or "SHADOW_ONLY"
    )
    for key, fallback in (
        ("market_structure_audit_last_run_at", ""),
        ("market_structure_top_symbol", ""),
        ("market_structure_top_zone", ""),
        ("market_structure_top_recommendation", "sem dados suficientes"),
        ("market_structure_data_sufficiency", "NO_DATA"),
        ("market_structure_why_no_candidate", "No market structure audit data yet."),
    ):
        market_structure_state[key] = str(market_structure_state.get(key) or fallback)
    market_structure_state["market_structure_candidates_count"] = int(
        market_structure_state.get("market_structure_candidates_count", 0) or 0
    )
    market_structure_state["market_structure_minimum_sample_met"] = bool(
        market_structure_state.get("market_structure_minimum_sample_met", False)
    )
    top_score = market_structure_state.get("market_structure_top_score")
    try:
        market_structure_state["market_structure_top_score"] = None if top_score in (None, "") else float(top_score or 0.0)
    except (TypeError, ValueError):
        market_structure_state["market_structure_top_score"] = None
    market_structure_state["market_structure_best_candidates"] = [
        item
        for item in list(market_structure_state.get("market_structure_best_candidates", []) or [])
        if isinstance(item, dict)
    ][:10]
    for key in (
        "market_structure_setup_confluence",
        "market_structure_fib_summary",
        "market_structure_blockers_summary",
        "market_structure_regime_summary",
    ):
        value = market_structure_state.get(key, {}) or {}
        market_structure_state[key] = dict(value) if isinstance(value, dict) else {}
    state["market_structure_audit"] = market_structure_state
    fib_alignment_state = state.get("fib_alignment_audit", {}) or {}
    fib_alignment_state["fib_alignment_enabled"] = bool(fib_alignment_state.get("fib_alignment_enabled", True))
    for key, fallback in (
        ("fib_alignment_mode", "SHADOW_ONLY"),
        ("fib_alignment_source", "video_pdf_inspired_checklist_v1"),
        ("fib_alignment_status", "insufficient_data"),
        ("fib_alignment_top_symbol", ""),
        ("fib_alignment_anchor_low_status", "insufficient"),
        ("fib_alignment_anchor_high_status", "insufficient"),
        ("fib_alignment_zone_status", "insufficient"),
        ("fib_alignment_pivot_status", "insufficient"),
        ("fib_alignment_bos_status", "insufficient"),
        ("fib_alignment_entry_confirmation_status", "insufficient"),
        ("fib_alignment_confluence_status", "insufficient"),
        ("fib_alignment_why_differs", "No Fibonacci video/PDF alignment audit data yet."),
        ("fib_alignment_recommendation", "insufficient_data"),
        ("fib_alignment_last_run_at", ""),
    ):
        fib_alignment_state[key] = str(fib_alignment_state.get(key) or fallback)
    alignment_score = fib_alignment_state.get("fib_alignment_score")
    try:
        fib_alignment_state["fib_alignment_score"] = None if alignment_score in (None, "") else float(alignment_score or 0.0)
    except (TypeError, ValueError):
        fib_alignment_state["fib_alignment_score"] = None
    fib_alignment_state["fib_alignment_missing_evidence"] = [
        str(item)
        for item in list(fib_alignment_state.get("fib_alignment_missing_evidence", []) or [])
        if str(item).strip()
    ][:8]
    fib_alignment_state["fib_alignment_checklist"] = [
        item
        for item in list(fib_alignment_state.get("fib_alignment_checklist", []) or [])
        if isinstance(item, dict)
    ][:12]
    state["fib_alignment_audit"] = fib_alignment_state
    intraday_fetch_state = state.get("multi_timeframe_intraday_fetcher", {}) or {}
    intraday_fetch_state["enabled"] = MULTITF_INTRADAY_FETCH_ENABLED
    intraday_fetch_state["shadow_only"] = MULTITF_INTRADAY_SHADOW_ONLY
    intraday_fetch_state["require_live_feed"] = MULTITF_INTRADAY_REQUIRE_LIVE_FEED
    for key, fallback in (
        ("mode", "SHADOW_ONLY"),
        ("generated_at", ""),
        ("provider_effective", ""),
        ("feed_status", "UNKNOWN"),
        ("provider_guard_reason", "No multi-timeframe intraday fetch data yet."),
        ("last_success_at", ""),
        ("last_error", ""),
        ("intraday_data_quality", "NO_DATA"),
        ("intraday_fetch_recommendation", "observe_more"),
        ("provider_budget_mode", MULTITF_INTRADAY_PROVIDER_BUDGET_MODE),
    ):
        intraday_fetch_state[key] = str(intraday_fetch_state.get(key) or fallback)
    for key in (
        "cache_hits",
        "cache_misses",
        "provider_calls_attempted",
        "provider_calls_skipped",
        "estimated_provider_calls",
        "cache_ttl_seconds",
        "max_symbols",
        "max_calls_per_cycle",
    ):
        try:
            default_value = 0
            if key == "cache_ttl_seconds":
                default_value = MULTITF_INTRADAY_CACHE_TTL_SECONDS
            elif key == "max_symbols":
                default_value = MULTITF_INTRADAY_MAX_SYMBOLS
            elif key == "max_calls_per_cycle":
                default_value = MULTITF_INTRADAY_MAX_CALLS_PER_CYCLE
            intraday_fetch_state[key] = int(intraday_fetch_state.get(key, default_value) or default_value)
        except (TypeError, ValueError):
            intraday_fetch_state[key] = 0
    intraday_fetch_state["provider_budget_guard_active"] = bool(
        intraday_fetch_state.get("provider_budget_guard_active", False)
    )
    configured_intraday_tfs = [item.strip().lower() for item in MULTITF_INTRADAY_TIMEFRAMES.split(",") if item.strip()]
    for key, fallback in (
        ("timeframes_requested", configured_intraday_tfs),
        ("timeframes_available", []),
        ("symbols_requested", []),
        ("symbols_fetched", []),
    ):
        intraday_fetch_state[key] = [
            str(item).strip()
            for item in list(intraday_fetch_state.get(key, fallback) or fallback)
            if str(item).strip()
        ][:10]
    intraday_fetch_state["diagnostics"] = [
        item
        for item in list(intraday_fetch_state.get("diagnostics", []) or [])
        if isinstance(item, dict)
    ][:30]
    state["multi_timeframe_intraday_fetcher"] = intraday_fetch_state
    multi_tf_state = state.get("multi_timeframe_swing_audit", {}) or {}
    multi_tf_state["enabled"] = MULTITF_SWING_AUDIT_ENABLED
    multi_tf_state["shadow_only"] = MULTITF_SWING_SHADOW_ONLY
    multi_tf_state["require_live_feed"] = MULTITF_SWING_REQUIRE_LIVE_FEED
    for key, fallback in (
        ("mode", "SHADOW_ONLY"),
        ("generated_at", ""),
        ("provider_effective", ""),
        ("feed_status", "UNKNOWN"),
        ("timeframe_source", "operational_cycle_resample"),
        ("top_symbol", ""),
        ("top_alignment_status", "INSUFFICIENT_DATA"),
        ("top_missing_confirmation", ""),
        ("top_recommendation", "insufficient_data"),
        ("dominant_conflict_reason", "No multi-timeframe swing audit data yet."),
        ("cache_status", "cycle_data_resample_only"),
        ("provider_guard", "not_evaluated"),
        ("intraday_top_symbol", ""),
        ("intraday_missing_reason", ""),
        ("h4_data_quality", "missing"),
        ("h1_data_quality", "missing"),
        ("bos_pivot_trace_relationship", "INSUFFICIENT_DATA"),
        ("bos_pivot_top_pivot_state", "INSUFFICIENT_DATA"),
        ("bos_pivot_top_bos_state", "INSUFFICIENT_DATA"),
        ("bos_pivot_dominant_missing_piece", ""),
        ("reason", "No multi-timeframe swing audit data yet."),
    ):
        multi_tf_state[key] = str(multi_tf_state.get(key) or fallback)
    multi_tf_state["uses_real_intraday_data"] = bool(multi_tf_state.get("uses_real_intraday_data", False))
    for key in (
        "symbols_analyzed",
        "candidates_count",
        "strong_alignment_count",
        "partial_alignment_count",
        "conflict_count",
        "insufficient_data_count",
        "setup_support_count",
        "estimated_provider_calls",
        "cache_ttl_seconds",
        "max_symbols",
    ):
        try:
            default_value = MULTITF_SWING_CACHE_TTL_SECONDS if key == "cache_ttl_seconds" else 0
            if key == "max_symbols":
                default_value = MULTITF_SWING_MAX_SYMBOLS
            multi_tf_state[key] = int(multi_tf_state.get(key, default_value) or default_value)
        except (TypeError, ValueError):
            multi_tf_state[key] = MULTITF_SWING_CACHE_TTL_SECONDS if key == "cache_ttl_seconds" else 0
    top_score = multi_tf_state.get("top_alignment_score")
    try:
        multi_tf_state["top_alignment_score"] = None if top_score in (None, "") else float(top_score or 0.0)
    except (TypeError, ValueError):
        multi_tf_state["top_alignment_score"] = None
    configured_timeframes = [item.strip().lower() for item in MULTITF_SWING_TIMEFRAMES.split(",") if item.strip()]
    timeframes = [
        str(item).strip().lower()
        for item in list(multi_tf_state.get("timeframes_used", configured_timeframes) or configured_timeframes)
        if str(item).strip()
    ]
    multi_tf_state["timeframes_used"] = timeframes or configured_timeframes or ["1d", "4h", "1h"]
    multi_tf_state["timeframe_fallbacks"] = [
        str(item)
        for item in list(multi_tf_state.get("timeframe_fallbacks", []) or [])
        if str(item).strip()
    ][:12]
    multi_tf_state["intraday_timeframes_available"] = [
        str(item)
        for item in list(multi_tf_state.get("intraday_timeframes_available", []) or [])
        if str(item).strip()
    ][:5]
    multi_tf_state["recent_candidates"] = [
        item
        for item in list(multi_tf_state.get("recent_candidates", []) or [])
        if isinstance(item, dict)
    ][:10]
    state["multi_timeframe_swing_audit"] = multi_tf_state
    bos_pivot_state = state.get("bos_pivot_trace_audit", {}) or {}
    bos_pivot_state["enabled"] = BOS_PIVOT_TRACE_AUDIT_ENABLED
    bos_pivot_state["shadow_only"] = BOS_PIVOT_TRACE_SHADOW_ONLY
    for key, fallback in (
        ("mode", "SHADOW_ONLY"),
        ("generated_at", ""),
        ("provider_effective", ""),
        ("feed_status", "UNKNOWN"),
        ("top_symbol", ""),
        ("top_timeframe", ""),
        ("top_pivot_state", "INSUFFICIENT_DATA"),
        ("top_bos_state", "INSUFFICIENT_DATA"),
        ("top_h4_bos_state", "INSUFFICIENT_DATA"),
        ("top_h1_bos_state", "INSUFFICIENT_DATA"),
        ("top_relationship", "INSUFFICIENT_DATA"),
        ("top_recommendation", "insufficient_data"),
        ("top_primary_missing_piece", "No BOS/Pivot trace audit data yet."),
        ("dominant_missing_piece", "No BOS/Pivot trace audit data yet."),
        ("reason", "No BOS/Pivot trace audit data yet."),
    ):
        bos_pivot_state[key] = str(bos_pivot_state.get(key) or fallback)
    bos_pivot_state["uses_real_intraday_data"] = bool(
        bos_pivot_state.get("uses_real_intraday_data", False)
    )
    for key in (
        "symbols_analyzed",
        "h4_bos_missing_count",
        "h1_bos_only_count",
        "wick_only_bos_count",
        "weak_close_bos_count",
        "confirmed_bos_count",
        "retest_pending_count",
        "pivot_forming_count",
        "pivot_confirmed_count",
        "pivot_triggered_count",
        "insufficient_data_count",
        "should_keep_blocked_count",
    ):
        try:
            bos_pivot_state[key] = int(bos_pivot_state.get(key, 0) or 0)
        except (TypeError, ValueError):
            bos_pivot_state[key] = 0
    bos_pivot_state["timeframes_analyzed"] = [
        str(item).strip().lower()
        for item in list(bos_pivot_state.get("timeframes_analyzed", ["4h", "1h"]) or ["4h", "1h"])
        if str(item).strip()
    ][:4]
    bos_pivot_state["recent_candidates"] = [
        item
        for item in list(bos_pivot_state.get("recent_candidates", []) or [])
        if isinstance(item, dict)
    ][:12]
    state["bos_pivot_trace_audit"] = bos_pivot_state
    bridge_state = state.get("strategy_decision_bridge_trace", {}) or {}
    bridge_state["enabled"] = STRATEGY_DECISION_BRIDGE_TRACE_ENABLED
    bridge_state["shadow_only"] = STRATEGY_DECISION_BRIDGE_TRACE_SHADOW_ONLY
    for key, fallback in (
        ("mode", "SHADOW_ONLY"),
        ("generated_at", ""),
        ("provider_effective", ""),
        ("feed_status", "UNKNOWN"),
        ("top_symbol", ""),
        ("top_bridge_status", "INSUFFICIENT_TRACE_DATA"),
        ("top_real_blocker", ""),
        ("top_structure_status", ""),
        ("top_reconciliation_status", "UNKNOWN_MISMATCH"),
        ("fallback_scope_status", "UNKNOWN_SCOPE"),
        ("fallback_blocker_scope", "UNKNOWN"),
        ("recommendation", "observe_more"),
        ("reason", "No strategy decision bridge trace data yet."),
    ):
        bridge_state[key] = str(bridge_state.get(key) or fallback)
    bridge_state["current_feed_is_clean"] = bool(bridge_state.get("current_feed_is_clean", False))
    for key in (
        "symbols_analyzed",
        "structure_confirmed_but_blocked_count",
        "fallback_scope_mismatch_count",
        "multi_tf_vs_bos_mismatch_count",
        "real_strategy_authority_count",
        "should_keep_blocked_count",
    ):
        try:
            bridge_state[key] = int(bridge_state.get(key, 0) or 0)
        except (TypeError, ValueError):
            bridge_state[key] = 0
    bridge_state["recent_candidates"] = [
        item
        for item in list(bridge_state.get("recent_candidates", []) or [])
        if isinstance(item, dict)
    ][:12]
    state["strategy_decision_bridge_trace"] = bridge_state
    feed_scope_state = state.get("feed_scope_reconciliation", {}) or {}
    feed_scope_state["enabled"] = bool(feed_scope_state.get("enabled", True))
    for key, fallback in (
        ("mode", "DIAGNOSTIC_ONLY"),
        ("generated_at", ""),
        ("provider_effective", ""),
        ("current_feed_status", "UNKNOWN"),
        ("current_cycle_feed_status", "UNKNOWN"),
        ("current_cycle_provider", ""),
        ("visual_feed_status", "UNKNOWN"),
        ("visual_chart_feed_status", "UNKNOWN"),
        ("worker_feed_status", "UNKNOWN"),
        ("dominant_rejection_current", ""),
        ("dominant_rejection_accumulated", ""),
        ("fallback_scope_status", "UNKNOWN_SCOPE"),
        ("fallback_blocker_scope", "UNKNOWN"),
        ("recommendation", "observe_more"),
        ("notes", "No feed scope reconciliation data yet."),
    ):
        feed_scope_state[key] = str(feed_scope_state.get(key) or fallback)
    for key in (
        "current_live_count",
        "current_cycle_live_count",
        "current_fallback_count",
        "current_cycle_fallback_count",
        "current_cycle_unknown_count",
        "accumulated_fallback_count",
        "accumulated_strategy_count",
        "historical_fallback_count",
    ):
        try:
            feed_scope_state[key] = int(feed_scope_state.get(key, 0) or 0)
        except (TypeError, ValueError):
            feed_scope_state[key] = 0
    feed_scope_state["current_feed_is_clean"] = bool(feed_scope_state.get("current_feed_is_clean", False))
    feed_scope_state["candidate_fallback_flags"] = dict(feed_scope_state.get("candidate_fallback_flags", {}) or {})
    state["feed_scope_reconciliation"] = feed_scope_state
    no_setup_state = state.get("no_setup_eligible_decomposition", {}) or {}
    no_setup_defaults = default_no_setup_eligible_decomposition_state()
    for key, fallback in no_setup_defaults.items():
        if key not in no_setup_state:
            no_setup_state[key] = fallback
    for key, fallback in (
        ("mode", "DIAGNOSTIC_ONLY"),
        ("safety_mode", "SHADOW_ONLY"),
        ("status", "INSUFFICIENT_DATA"),
        ("generated_at", ""),
        ("target_setup", "trend_pullback_breakout"),
        ("top_symbol", ""),
        ("top_setup", "trend_pullback_breakout"),
        ("top_reason_bucket", "INSUFFICIENT_DATA_FOR_DECOMPOSITION"),
        ("top_real_blocker", ""),
        ("top_secondary_blocker", ""),
        ("fallback_blocker_scope", "UNKNOWN"),
        ("recommendation", "insufficient_data"),
        ("notes", "No NO_SETUP_ELIGIBLE decomposition data yet."),
    ):
        no_setup_state[key] = str(no_setup_state.get(key) or fallback)
    no_setup_state["enabled"] = bool(no_setup_state.get("enabled", True))
    no_setup_state["should_keep_blocked"] = True
    no_setup_state["shadow_only"] = bool(no_setup_state.get("shadow_only", True))
    no_setup_state["current_feed_is_clean"] = bool(no_setup_state.get("current_feed_is_clean", False))
    for key in (
        "total_candidates_checked",
        "no_setup_eligible_count",
        "structure_confirmed_count",
        "structure_confirmed_but_no_setup_count",
        "near_approved_no_setup_count",
    ):
        try:
            no_setup_state[key] = int(no_setup_state.get(key, 0) or 0)
        except (TypeError, ValueError):
            no_setup_state[key] = 0
    for key in ("top_score", "top_min_score", "top_score_gap"):
        value = no_setup_state.get(key)
        try:
            no_setup_state[key] = None if value in (None, "") else float(value)
        except (TypeError, ValueError):
            no_setup_state[key] = None
    no_setup_state["candidates"] = [
        item
        for item in list(no_setup_state.get("candidates", []) or [])
        if isinstance(item, dict)
    ][:10]
    state["no_setup_eligible_decomposition"] = no_setup_state
    shadow_state = state.get("shadow_decision_simulator", {}) or {}
    shadow_state["shadow_decision_simulator_enabled"] = bool(
        shadow_state.get("shadow_decision_simulator_enabled", True)
    )
    for key, fallback in (
        ("shadow_decision_mode", "SHADOW_ONLY"),
        ("shadow_entry_policy", "conservative_v1"),
        ("shadow_decision_last_run_at", ""),
        ("shadow_best_symbol", ""),
        ("shadow_best_strategy", ""),
        ("shadow_dominant_block_reason", ""),
        ("shadow_dominant_block_reason_current", ""),
        ("shadow_dominant_block_reason_accumulated", ""),
        ("dominant_exclusion_current_scope", ""),
        ("dominant_exclusion_accumulated_scope", ""),
        ("fallback_blocker_scope", "UNKNOWN"),
        ("fallback_scope_status", "UNKNOWN_SCOPE"),
        ("fallback_scope_note", ""),
        ("shadow_policy_recommendation", "observe_more"),
        ("shadow_ignored_reason", ""),
        ("shadow_counts_scope", "current_cycle_and_accumulated_recent"),
        ("shadow_counter_health_status", "healthy"),
        ("shadow_counter_warning_reason", ""),
        ("shadow_scope_warning_reason", ""),
        ("shadow_reason", "No shadow decision simulator data yet."),
    ):
        shadow_state[key] = str(shadow_state.get(key) or fallback)
    shadow_state["shadow_counter_warning"] = bool(shadow_state.get("shadow_counter_warning", False))
    shadow_state["shadow_scope_warning"] = bool(shadow_state.get("shadow_scope_warning", False))
    for key in (
        "preview_near_approved_count",
        "shadow_candidates_received_count",
        "shadow_candidates_unique_count",
        "shadow_candidates_ignored_count",
        "shadow_candidates_classified_count",
        "shadow_candidates_analyzed_count",
        "shadow_raw_near_approved_count",
        "shadow_current_cycle_candidates_count",
        "shadow_accumulated_candidates_count",
        "shadow_current_cycle_received_count",
        "shadow_current_cycle_new_unique_count",
        "shadow_current_cycle_duplicate_count",
        "shadow_current_cycle_already_analyzed_count",
        "shadow_current_cycle_analyzed_count",
        "shadow_current_cycle_analyzed_new_count",
        "shadow_current_cycle_classified_count",
        "shadow_current_cycle_classified_new_count",
        "shadow_current_cycle_raw_near_approved_count",
        "shadow_current_cycle_safe_near_approved_count",
        "shadow_current_cycle_marginal_near_approved_count",
        "shadow_current_cycle_unsafe_count",
        "shadow_current_cycle_unsafe_new_count",
        "shadow_current_cycle_ignored_count",
        "shadow_current_cycle_primary_blocked_count",
        "shadow_current_cycle_primary_blocked_new_count",
        "shadow_current_cycle_secondary_blocked_count",
        "shadow_current_cycle_secondary_blocked_new_count",
        "shadow_accumulated_received_count",
        "shadow_accumulated_raw_received_count",
        "shadow_accumulated_unique_candidates_count",
        "shadow_accumulated_analyzed_count",
        "shadow_accumulated_analyzed_unique_count",
        "shadow_accumulated_classified_unique_count",
        "shadow_accumulated_raw_near_approved_count",
        "shadow_accumulated_unsafe_count",
        "shadow_accumulated_unsafe_unique_count",
        "shadow_accumulated_primary_blocked_count",
        "shadow_accumulated_secondary_blocked_count",
        "shadow_near_approved_count",
        "shadow_safe_near_approved_count",
        "shadow_marginal_near_approved_count",
        "shadow_marginal_count",
        "shadow_unsafe_count",
        "shadow_unsafe_rejection_count",
        "shadow_primary_blocked_count",
        "shadow_secondary_blocked_count",
        "shadow_structure_missing_count",
        "shadow_confirmation_missing_count",
        "shadow_ignored_count",
        "shadow_would_enter_count",
        "shadow_pending_count",
        "shadow_would_win_count",
        "shadow_would_lose_count",
        "fallback_current_count",
        "fallback_accumulated_count",
        "shadow_max_hold_cycles",
    ):
        try:
            shadow_state[key] = int(shadow_state.get(key, 0) or 0)
        except (TypeError, ValueError):
            shadow_state[key] = 0
    if shadow_state["shadow_accumulated_raw_received_count"] <= 0 and shadow_state["shadow_accumulated_received_count"] > 0:
        shadow_state["shadow_accumulated_raw_received_count"] = shadow_state["shadow_accumulated_received_count"]
    if shadow_state["shadow_accumulated_unique_candidates_count"] <= 0 and shadow_state["shadow_accumulated_candidates_count"] > 0:
        shadow_state["shadow_accumulated_unique_candidates_count"] = shadow_state["shadow_accumulated_candidates_count"]
    if shadow_state["shadow_accumulated_analyzed_unique_count"] <= 0 and shadow_state["shadow_accumulated_analyzed_count"] > 0:
        shadow_state["shadow_accumulated_analyzed_unique_count"] = shadow_state["shadow_accumulated_analyzed_count"]
    if shadow_state["shadow_accumulated_unsafe_unique_count"] <= 0 and shadow_state["shadow_accumulated_unsafe_count"] > 0:
        shadow_state["shadow_accumulated_unsafe_unique_count"] = shadow_state["shadow_accumulated_unsafe_count"]
    for key in (
        "shadow_best_candidate_score",
        "shadow_stop_pct",
        "shadow_take_profit_pct",
        "shadow_raw_to_unique_ratio",
        "shadow_duplicate_ratio",
    ):
        value = shadow_state.get(key)
        try:
            shadow_state[key] = None if value in (None, "") else float(value or 0.0)
        except (TypeError, ValueError):
            shadow_state[key] = 0.0 if key in {"shadow_raw_to_unique_ratio", "shadow_duplicate_ratio"} else None
    shadow_state["shadow_current_cycle_candidates"] = [
        item
        for item in list(shadow_state.get("shadow_current_cycle_candidates", []) or [])
        if isinstance(item, dict)
    ][:30]
    shadow_state["shadow_accumulated_recent_candidates"] = [
        item
        for item in list(shadow_state.get("shadow_accumulated_recent_candidates", []) or [])
        if isinstance(item, dict)
    ][:30]
    shadow_state["shadow_recent_candidates"] = [
        item
        for item in list(shadow_state.get("shadow_recent_candidates", []) or [])
        if isinstance(item, dict)
    ][:30]
    if not shadow_state["shadow_accumulated_recent_candidates"]:
        shadow_state["shadow_accumulated_recent_candidates"] = list(shadow_state["shadow_recent_candidates"])
    outcome_summary = shadow_state.get("shadow_outcome_summary", {}) or {}
    shadow_state["shadow_outcome_summary"] = dict(outcome_summary) if isinstance(outcome_summary, dict) else {}
    state["shadow_decision_simulator"] = shadow_state
    phase2_fine_tune_state = state.get("phase2_fine_tune", {}) or {}
    phase2_fine_tune_state["fine_tune_enabled"] = bool(phase2_fine_tune_state.get("fine_tune_enabled", True))
    for key, fallback in (
        ("fine_tune_reason", "Relaxamento conservador de confirmacao secundaria marginal em PAPER."),
        ("fine_tune_target", "trend_pullback_breakout_secondary_breakout_confirmation"),
        ("fine_tune_before", "breakout_20 >= breakout_min"),
        ("fine_tune_after", "breakout_20 >= breakout_min - 0.005, apenas com score minimo preservado e guards seguros"),
        ("fine_tune_last_guard_reason", ""),
    ):
        phase2_fine_tune_state[key] = str(phase2_fine_tune_state.get(key) or fallback)
    phase2_fine_tune_state["fine_tune_applied_count"] = int(phase2_fine_tune_state.get("fine_tune_applied_count", 0) or 0)
    phase2_fine_tune_state["fine_tune_blocked_count"] = int(phase2_fine_tune_state.get("fine_tune_blocked_count", 0) or 0)
    state["phase2_fine_tune"] = phase2_fine_tune_state
    phase2_1_state = state.get("phase2_1_fine_tune", {}) or {}
    phase2_1_state["phase2_1_fine_tune_enabled"] = bool(phase2_1_state.get("phase2_1_fine_tune_enabled", True))
    for key, fallback in (
        ("phase2_1_fine_tune_target", "trend_pullback_breakout_multi_minor_confirmation"),
        ("phase2_1_fine_tune_reason", "Relaxamento conservador de multiplas falhas pequenas de momentum/confirmacao secundaria em PAPER."),
        ("phase2_1_fine_tune_last_guard", ""),
        ("phase2_1_fine_tune_last_decision", ""),
    ):
        phase2_1_state[key] = str(phase2_1_state.get(key) or fallback)
    phase2_1_state["phase2_1_fine_tune_applied_count"] = int(
        phase2_1_state.get("phase2_1_fine_tune_applied_count", 0) or 0
    )
    phase2_1_state["phase2_1_fine_tune_blocked_count"] = int(
        phase2_1_state.get("phase2_1_fine_tune_blocked_count", 0) or 0
    )
    score_gap = phase2_1_state.get("phase2_1_fine_tune_score_gap")
    try:
        phase2_1_state["phase2_1_fine_tune_score_gap"] = None if score_gap in (None, "") else float(score_gap or 0.0)
    except (TypeError, ValueError):
        phase2_1_state["phase2_1_fine_tune_score_gap"] = None
    phase2_1_state["phase2_1_fine_tune_allowed_reasons"] = [
        str(item) for item in list(phase2_1_state.get("phase2_1_fine_tune_allowed_reasons", []) or [])
    ][:10]
    phase2_1_state["phase2_1_fine_tune_blocked_reasons"] = [
        str(item) for item in list(phase2_1_state.get("phase2_1_fine_tune_blocked_reasons", []) or [])
    ][:10]
    state["phase2_1_fine_tune"] = phase2_1_state
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
    payload["source_commit_sha"] = str(payload.get("source_commit_sha") or APP_SOURCE_COMMIT_SHA or "")
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
    if payload.get("requested_interval") is not None:
        context_state["requested_interval"] = str(payload.get("requested_interval") or "")
    if payload.get("effective_interval") is not None:
        context_state["effective_interval"] = str(payload.get("effective_interval") or "")
    for key in ("live_symbols", "cached_symbols", "fallback_symbols", "unknown_symbols"):
        if payload.get(key) is not None:
            context_state[key] = [str(symbol).upper() for symbol in (payload.get(key) or [])]
    for key in (
        "build_active",
        "git_sha",
        "source_commit_sha",
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
            "requested_interval",
            "effective_interval",
            "live_symbols",
            "cached_symbols",
            "fallback_symbols",
            "unknown_symbols",
            "requested_by",
            "build_active",
            "git_sha",
            "source_commit_sha",
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


def update_macro_alert_status(status_payload: dict | None) -> dict:
    state = load_bot_state()
    macro_state = state.get("macro_alert", {}) or {}
    payload = status_payload or {}

    for key, value in payload.items():
        if value is not None:
            macro_state[key] = value

    macro_state["macro_alert_enabled"] = MACRO_ALERTS_ENABLED
    macro_state["macro_alerts_file"] = MACRO_ALERTS_FILE
    macro_state["macro_alert_pre_event_minutes"] = MACRO_ALERT_PRE_EVENT_MINUTES
    macro_state["macro_alert_event_window_minutes"] = MACRO_ALERT_EVENT_WINDOW_MINUTES
    macro_state["macro_alert_post_event_minutes"] = MACRO_ALERT_POST_EVENT_MINUTES
    state["macro_alert"] = macro_state
    save_bot_state(state)
    return macro_state


def update_external_signal_status(status_payload: dict | None) -> dict:
    state = load_bot_state()
    external_signal_state = state.get("external_signal", {}) or {}
    payload = status_payload or {}

    for key, value in payload.items():
        if value is not None:
            external_signal_state[key] = value

    external_signal_state["enabled"] = EXTERNAL_SIGNAL_WEBHOOK_ENABLED
    external_signal_state["webhook_configured"] = bool(
        EXTERNAL_SIGNAL_WEBHOOK_ENABLED and EXTERNAL_SIGNAL_SECRET and EXTERNAL_SIGNAL_ALLOWED_SOURCES
    )
    external_signal_state["allowed_sources"] = EXTERNAL_SIGNAL_ALLOWED_SOURCES
    external_signal_state["allowed_timeframes"] = EXTERNAL_SIGNAL_ALLOWED_TIMEFRAMES
    external_signal_state["max_age_seconds"] = EXTERNAL_SIGNAL_MAX_AGE_SECONDS
    external_signal_state["dedupe_seconds"] = EXTERNAL_SIGNAL_DEDUPE_SECONDS
    external_signal_state["test_panel_enabled"] = EXTERNAL_SIGNAL_TEST_PANEL_ENABLED
    external_signal_state["audit_only"] = True
    external_signal_state["recent_events"] = [
        event
        for event in list(external_signal_state.get("recent_events", []) or [])
        if isinstance(event, dict)
    ][-20:]
    state["external_signal"] = external_signal_state
    save_bot_state(state)
    return external_signal_state


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
