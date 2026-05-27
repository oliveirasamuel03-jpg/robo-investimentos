"""Provider budget and visual fallback observability for FASE 2.6B.2.

This module is intentionally diagnostic only. It never changes trading
decisions, scores, thresholds, broker/provider configuration, positions,
orders, PnL, or official paper-trading execution.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Mapping


PHASE = "2.6B.2"
MODE = "OBSERVABILITY_ONLY"
DIAGNOSTIC_MODE = "DIAGNOSTIC_ONLY"
SAFETY_MODE = "SHADOW_ONLY"


def _env_float(name: str, default: float) -> float:
    try:
        raw = str(os.getenv(name, "") or "").strip()
        return float(raw) if raw else float(default)
    except (TypeError, ValueError):
        return float(default)


TWELVEDATA_DAILY_CREDIT_LIMIT_ESTIMATE = _env_float("TWELVEDATA_DAILY_CREDIT_LIMIT", 800)
TWELVEDATA_MINUTE_LIMIT_ESTIMATE = _env_float("TWELVEDATA_MINUTE_LIMIT", 8)

PROVIDER_BUDGET_VISUAL_FALLBACK_MARKERS = (
    "[provider_budget_visual_fallback_summary]",
    "[provider_budget_visual_fallback_budget]",
    "[provider_budget_visual_fallback_scope]",
    "[provider_budget_visual_fallback_cache]",
    "[provider_budget_visual_fallback_safety]",
)

BLOCKED_ACTIONS = [
    "start_real_money",
    "approve_trade_from_provider_budget",
    "approve_trade_from_fallback",
    "lower_global_min_signal_score_now",
    "change_threshold_now",
    "change_score_now",
    "change_broker_now",
    "change_provider_now",
    "increase_capital_now",
    "increase_ticket_now",
    "increase_max_open_positions_now",
    "advance_to_phase_2_6c",
    "convert_visual_fallback_to_signal",
    "convert_worker_fallback_to_signal",
    "convert_shadow_to_real_signal",
    "use_fallback_as_reliable_strategy_data",
]

SAFE_RECOMMENDATIONS = {
    "observe_provider_budget",
    "separate_worker_and_visual_fallback",
    "mark_worker_fallback_not_reliable",
    "mark_visual_only_fallback",
    "watch_daily_budget",
    "watch_minute_burst_risk",
    "keep_paper_no_2_6c",
    "insufficient_data",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _safe_text(value: Any, default: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else default


def _log_text(value: Any, default: str = "unknown") -> str:
    text = _safe_text(value, default)
    return text.replace("\n", " ").replace("\r", " ").replace(";", ",").replace("|", ",")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_number(*values: Any) -> float | None:
    for value in values:
        parsed = _safe_float(value, None)
        if parsed is not None:
            return parsed
    return None


def _first_number_with_source(*items: tuple[Any, str]) -> tuple[float | None, str]:
    for value, source in items:
        parsed = _safe_float(value, None)
        if parsed is not None:
            return parsed, source
    return None, "unknown"


def _is_twelvedata_provider(*providers: Any) -> bool:
    aliases = {"twelvedata", "twelve_data", "twelve-data"}
    return any(_safe_text(provider, "").lower() in aliases for provider in providers)


def _feed_status(value: Any) -> str:
    raw = _safe_text(value, "UNKNOWN").upper()
    if raw in {"LIVE", "DELAYED", "FALLBACK", "UNKNOWN"}:
        return raw
    if raw == "HEALTHY":
        return "LIVE"
    if raw in {"DEGRADED", "CACHED"}:
        return "DELAYED"
    if raw == "ERROR":
        return "FALLBACK"
    return "UNKNOWN"


def _source_breakdown(payload: dict[str, Any]) -> dict[str, int]:
    raw = _as_dict(payload.get("source_breakdown"))
    return {
        "market": _safe_int(raw.get("market")),
        "cached": _safe_int(raw.get("cached")),
        "fallback": _safe_int(raw.get("fallback")),
        "unknown": _safe_int(raw.get("unknown")),
    }


def _fallback_count(payload: dict[str, Any]) -> int:
    symbols = _as_list(payload.get("fallback_symbols"))
    if symbols:
        return len(symbols)
    return _safe_int(_source_breakdown(payload).get("fallback"))


def _provider(payload: dict[str, Any]) -> str:
    return _safe_text(payload.get("provider_effective") or payload.get("provider"), "unknown").lower()


def _twelvedata_diag(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _as_dict(payload.get("provider_diagnostics"))
    return _as_dict(diagnostics.get("twelvedata"))


def _has_429(payload: dict[str, Any], td_diag: dict[str, Any]) -> bool:
    response_status = _safe_int(payload.get("response_status_code"), 0)
    http_statuses = [_safe_int(item, 0) for item in _as_list(td_diag.get("http_statuses"))]
    payload_codes = [str(item or "").strip() for item in _as_list(td_diag.get("payload_codes"))]
    last_error = f"{payload.get('last_error') or ''} {td_diag.get('last_error') or ''}".lower()
    return (
        response_status == 429
        or 429 in http_statuses
        or "429" in payload_codes
        or "429" in last_error
        or ("limit" in last_error and "twelve" in last_error)
    )


def _budget_status(used: float | None, limit: float | None, has_429: bool) -> tuple[str, float | None]:
    if has_429:
        usage_pct = None if used is None or not limit else round(float(used) / max(float(limit), 1.0), 4)
        return "LIMIT_EXCEEDED_OR_429", usage_pct
    if used is None or not limit:
        return "UNKNOWN", None
    usage_pct = round(float(used) / max(float(limit), 1.0), 4)
    if usage_pct >= 0.90:
        return "DAILY_BUDGET_CRITICAL", usage_pct
    if usage_pct >= 0.75:
        return "DAILY_BUDGET_HIGH", usage_pct
    if usage_pct >= 0.50:
        return "DAILY_BUDGET_WATCH", usage_pct
    return "DAILY_BUDGET_OK", usage_pct


def _minute_status(
    *,
    minute_average: float | None,
    minute_maximum: float | None,
    minute_limit: float | None,
    estimated_calls: int,
) -> tuple[str, bool]:
    if minute_limit is None:
        return "UNKNOWN", False
    limit = max(float(minute_limit), 1.0)
    if minute_maximum is not None and float(minute_maximum) >= limit:
        return "MINUTE_BURST_RISK", True
    if estimated_calls >= limit:
        return "MINUTE_BURST_RISK", True
    if minute_maximum is not None and float(minute_maximum) >= limit * 0.75:
        return "MINUTE_PRESSURE_WATCH", False
    if minute_average is not None and float(minute_average) >= limit * 0.75:
        return "MINUTE_PRESSURE_WATCH", False
    if minute_average is None and minute_maximum is None:
        return "UNKNOWN", False
    return "MINUTE_OK", False


def _cache_status(cache_hits: int, cache_misses: int) -> str:
    if cache_hits > 0 and cache_misses <= 0:
        return "CACHE_REUSED"
    if cache_hits > 0 and cache_misses > 0:
        return "PARTIAL_CACHE_REUSE"
    if cache_misses > 0:
        return "CACHE_MISSES_PRESENT"
    return "UNKNOWN"


def _recommendation(
    *,
    worker_fallback: bool,
    visual_only_fallback: bool,
    daily_status: str,
    minute_status: str,
    has_429: bool,
) -> str:
    if worker_fallback:
        return "mark_worker_fallback_not_reliable"
    if visual_only_fallback:
        return "mark_visual_only_fallback"
    if has_429 or daily_status in {"LIMIT_EXCEEDED_OR_429", "DAILY_BUDGET_CRITICAL", "DAILY_BUDGET_HIGH"}:
        return "watch_daily_budget"
    if minute_status == "MINUTE_BURST_RISK":
        return "watch_minute_burst_risk"
    return "observe_provider_budget"


def _alerts(
    *,
    worker_fallback: bool,
    visual_only_fallback: bool,
    daily_status: str,
    minute_status: str,
    has_429: bool,
    daily_used: float | None,
    daily_limit: float | None,
    minute_maximum: float | None,
    minute_limit: float | None,
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    if has_429:
        alerts.append(
            {
                "id": "twelvedata_429_observed",
                "severity": "HIGH",
                "message": "Twelve Data retornou 429/limite; tratar como risco de cota, nao como autorizacao operacional.",
                "operational_authority": False,
            }
        )
    if daily_status in {"DAILY_BUDGET_WATCH", "DAILY_BUDGET_HIGH", "DAILY_BUDGET_CRITICAL", "LIMIT_EXCEEDED_OR_429"}:
        used_label = "unknown" if daily_used is None else f"{float(daily_used):.0f}"
        limit_label = "unknown" if daily_limit is None else f"{float(daily_limit):.0f}"
        alerts.append(
            {
                "id": "daily_budget_pressure",
                "severity": "HIGH" if daily_status in {"DAILY_BUDGET_CRITICAL", "LIMIT_EXCEEDED_OR_429"} else "MEDIUM",
                "message": f"Cota diaria Twelve Data em atencao: {used_label}/{limit_label} creditos estimados.",
                "operational_authority": False,
            }
        )
    if minute_status == "MINUTE_BURST_RISK":
        max_label = "unknown" if minute_maximum is None else f"{float(minute_maximum):.0f}"
        limit_label = "unknown" if minute_limit is None else f"{float(minute_limit):.0f}"
        alerts.append(
            {
                "id": "minute_burst_risk",
                "severity": "MEDIUM",
                "message": f"Limite por minuto pressionado: max={max_label}/{limit_label}. Risco de rajada, nao falha confirmada.",
                "operational_authority": False,
            }
        )
    if worker_fallback:
        alerts.append(
            {
                "id": "worker_operational_fallback",
                "severity": "HIGH",
                "message": "Fallback operacional do worker: leitura estrategica nao confiavel neste ciclo.",
                "operational_authority": False,
            }
        )
    if visual_only_fallback:
        alerts.append(
            {
                "id": "visual_only_fallback",
                "severity": "MEDIUM",
                "message": "Fallback apenas visual no grafico/Trader; separar da leitura operacional do worker.",
                "operational_authority": False,
            }
        )
    return alerts


def default_provider_budget_visual_fallback_state(
    reason: str = "No provider budget visual fallback data yet.",
) -> dict[str, Any]:
    return {
        "enabled": True,
        "phase": PHASE,
        "mode": MODE,
        "diagnostic_mode": DIAGNOSTIC_MODE,
        "safety_mode": SAFETY_MODE,
        "generated_at": "",
        "provider_configured": "unknown",
        "provider_effective_worker": "unknown",
        "provider_effective_visual": "unknown",
        "worker_feed_status": "UNKNOWN",
        "visual_feed_status": "UNKNOWN",
        "worker_fallback_operational": False,
        "visual_fallback_active": False,
        "visual_only_fallback": False,
        "worker_strategy_reading_reliable": True,
        "fallback_scope": "UNKNOWN",
        "fallback_scope_status": "UNKNOWN",
        "fallback_blocker_scope": "UNKNOWN",
        "daily_budget_limit": None,
        "daily_budget_source": "unknown",
        "daily_credit_limit_estimate": TWELVEDATA_DAILY_CREDIT_LIMIT_ESTIMATE,
        "daily_credits_used_estimate": None,
        "daily_credit_usage_pct": None,
        "daily_budget_status": "UNKNOWN",
        "minute_limit": None,
        "minute_limit_source": "unknown",
        "minute_limit_estimate": TWELVEDATA_MINUTE_LIMIT_ESTIMATE,
        "minutely_average": None,
        "minutely_maximum": None,
        "minute_limit_status": "UNKNOWN",
        "minute_burst_risk": False,
        "risk_429": False,
        "cache_hits": 0,
        "cache_misses": 0,
        "stale_cache_hits": 0,
        "cache_status": "UNKNOWN",
        "estimated_provider_calls": 0,
        "provider_calls_attempted": 0,
        "provider_calls_skipped": 0,
        "requested_by_worker": "worker_cycle",
        "requested_by_visual": "",
        "budget_block_reason": "none",
        "provider_budget_status": "UNKNOWN",
        "ui_alerts": [],
        "blocked_actions": list(BLOCKED_ACTIONS),
        "recommendation": "insufficient_data",
        "provider_budget_recommendation": "insufficient_data",
        "notes": reason,
        "paper_required": True,
        "observability_only": True,
        "diagnostic_only": True,
        "shadow_only": True,
        "should_continue_paper": True,
        "should_start_real_money": False,
        "should_change_threshold_now": False,
        "should_change_score_now": False,
        "should_change_broker_now": False,
        "should_change_provider_now": False,
        "should_change_profile_now": False,
        "should_apply_micro_adjustment_now": False,
        "should_advance_2_6c_now": False,
        "trade_authority": False,
        "score_authority": False,
        "broker_authority": False,
        "provider_authority": False,
        "threshold_authority": False,
        "execution_authority": False,
        "can_approve_trade": False,
        "can_change_provider": False,
        "can_change_threshold": False,
        "can_change_score": False,
        "can_change_broker": False,
        "can_advance_2_6c": False,
    }


def build_provider_budget_visual_fallback_audit(
    *,
    market_data_status: dict[str, Any] | None = None,
    visual_chart_status: dict[str, Any] | None = None,
    feed_scope_reconciliation: dict[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    if not enabled:
        disabled = default_provider_budget_visual_fallback_state("Provider budget visual fallback audit disabled.")
        disabled["enabled"] = False
        return disabled

    worker_status = _as_dict(market_data_status)
    visual_status = _as_dict(visual_chart_status)
    feed_scope = _as_dict(feed_scope_reconciliation)
    td_diag = _twelvedata_diag(worker_status)

    worker_feed = _feed_status(worker_status.get("feed_status") or worker_status.get("status"))
    visual_feed = _feed_status(visual_status.get("feed_status") or visual_status.get("status"))
    worker_fallback_count = _fallback_count(worker_status)
    visual_fallback_count = _fallback_count(visual_status)
    worker_fallback = bool(worker_feed == "FALLBACK" or worker_fallback_count > 0)
    visual_fallback = bool(visual_feed == "FALLBACK" or visual_fallback_count > 0)
    visual_only_fallback = bool(visual_fallback and not worker_fallback)

    configured_provider = _safe_text(worker_status.get("configured_provider"), "unknown").lower()
    worker_provider = _provider(worker_status)
    visual_provider = _provider(visual_status) if visual_status else "unknown"
    twelvedata_worker = _is_twelvedata_provider(worker_provider, configured_provider)
    has_429 = _has_429(worker_status, td_diag)

    daily_limit, daily_limit_source = _first_number_with_source(
        (worker_status.get("twelvedata_daily_credit_limit"), "configured"),
        (worker_status.get("daily_credit_limit"), "configured"),
        (td_diag.get("daily_credit_limit"), "configured"),
        (td_diag.get("credit_limit"), "configured"),
    ) or float(TWELVEDATA_DAILY_CREDIT_LIMIT_ESTIMATE)
    if daily_limit is None and twelvedata_worker:
        daily_limit = float(TWELVEDATA_DAILY_CREDIT_LIMIT_ESTIMATE)
        daily_limit_source = "estimated"
    daily_used = _first_number(
        worker_status.get("twelvedata_daily_credits_used"),
        worker_status.get("daily_credits_used"),
        worker_status.get("credits_used"),
        td_diag.get("daily_credits_used"),
        td_diag.get("credits_used"),
        td_diag.get("api_credits_used"),
    )
    daily_status, daily_usage_pct = _budget_status(daily_used, daily_limit, has_429)
    daily_budget_source = "measured" if daily_used is not None else daily_limit_source
    if daily_status == "UNKNOWN" and twelvedata_worker and daily_limit is not None:
        daily_status = "DAILY_BUDGET_CONFIGURED_ONLY"

    minute_limit, minute_limit_source = _first_number_with_source(
        (worker_status.get("twelvedata_minutely_limit"), "configured"),
        (worker_status.get("minutely_limit"), "configured"),
        (td_diag.get("minutely_limit"), "configured"),
        (td_diag.get("minute_limit"), "configured"),
    ) or float(TWELVEDATA_MINUTE_LIMIT_ESTIMATE)
    if minute_limit is None and twelvedata_worker:
        minute_limit = float(TWELVEDATA_MINUTE_LIMIT_ESTIMATE)
        minute_limit_source = "estimated"
    minute_average = _first_number(
        worker_status.get("twelvedata_minutely_average"),
        worker_status.get("minutely_average"),
        td_diag.get("minutely_average"),
    )
    minute_maximum = _first_number(
        worker_status.get("twelvedata_minutely_maximum"),
        worker_status.get("minutely_maximum"),
        td_diag.get("minutely_maximum"),
    )
    provider_calls_attempted = _safe_int(
        worker_status.get("provider_calls_attempted")
        or worker_status.get("estimated_provider_calls")
        or td_diag.get("request_attempted_count")
    )
    estimated_provider_calls = _safe_int(worker_status.get("estimated_provider_calls"), provider_calls_attempted)
    minute_status, minute_burst = _minute_status(
        minute_average=minute_average,
        minute_maximum=minute_maximum,
        minute_limit=minute_limit,
        estimated_calls=estimated_provider_calls or provider_calls_attempted,
    )
    minute_limit_source = "measured" if minute_average is not None or minute_maximum is not None else minute_limit_source
    if minute_status == "UNKNOWN" and twelvedata_worker and minute_limit is not None:
        minute_status = "MINUTE_LIMIT_CONFIGURED_ONLY"

    cache_hits = _safe_int(worker_status.get("cache_hits"))
    cache_misses = _safe_int(worker_status.get("cache_misses"))
    stale_cache_hits = _safe_int(worker_status.get("stale_cache_hits"))
    cache_status = _cache_status(cache_hits + stale_cache_hits, cache_misses)

    if worker_fallback:
        fallback_scope = "WORKER_OPERATIONAL_FALLBACK"
    elif visual_only_fallback:
        fallback_scope = "VISUAL_CHART_ONLY"
    elif worker_feed in {"LIVE", "DELAYED"} and visual_feed in {"LIVE", "DELAYED", "UNKNOWN"}:
        fallback_scope = "NONE"
    else:
        fallback_scope = _safe_text(feed_scope.get("fallback_blocker_scope"), "UNKNOWN")

    if worker_fallback:
        budget_block_reason = "worker_operational_fallback"
        provider_budget_status = "WORKER_OPERATIONAL_FALLBACK"
    elif has_429:
        budget_block_reason = "twelvedata_429_or_limit_observed"
        provider_budget_status = "PROVIDER_RATE_LIMIT_OBSERVED"
    elif minute_status == "MINUTE_BURST_RISK":
        budget_block_reason = "minute_burst_risk"
        provider_budget_status = "MINUTE_BURST_RISK"
    elif daily_status in {"DAILY_BUDGET_WATCH", "DAILY_BUDGET_HIGH", "DAILY_BUDGET_CRITICAL", "LIMIT_EXCEEDED_OR_429"}:
        budget_block_reason = "daily_budget_pressure"
        provider_budget_status = daily_status
    elif visual_only_fallback:
        budget_block_reason = "visual_only_fallback"
        provider_budget_status = "VISUAL_FALLBACK_ONLY"
    elif not twelvedata_worker and daily_status == "UNKNOWN" and minute_status == "UNKNOWN":
        budget_block_reason = "insufficient_provider_budget_data"
        provider_budget_status = "UNKNOWN_PROVIDER_BUDGET"
    else:
        budget_block_reason = "none"
        provider_budget_status = "OK"

    recommendation = _recommendation(
        worker_fallback=worker_fallback,
        visual_only_fallback=visual_only_fallback,
        daily_status=daily_status,
        minute_status=minute_status,
        has_429=has_429,
    )
    if recommendation not in SAFE_RECOMMENDATIONS:
        recommendation = "insufficient_data"
    if provider_budget_status == "UNKNOWN_PROVIDER_BUDGET":
        recommendation = "insufficient_data"

    alerts = _alerts(
        worker_fallback=worker_fallback,
        visual_only_fallback=visual_only_fallback,
        daily_status=daily_status,
        minute_status=minute_status,
        has_429=has_429,
        daily_used=daily_used,
        daily_limit=daily_limit,
        minute_maximum=minute_maximum,
        minute_limit=minute_limit,
    )

    notes = (
        "Worker fallback operational makes strategic reading unreliable."
        if worker_fallback
        else (
            "Visual chart fallback is separated from the operational worker feed."
            if visual_only_fallback
            else "Provider budget and feed scopes are observational only."
        )
    )

    audit = {
        "enabled": True,
        "phase": PHASE,
        "mode": MODE,
        "diagnostic_mode": DIAGNOSTIC_MODE,
        "safety_mode": SAFETY_MODE,
        "generated_at": _utc_now_iso(),
        "provider_configured": configured_provider,
        "provider_effective_worker": worker_provider,
        "provider_effective_visual": visual_provider,
        "worker_feed_status": worker_feed,
        "visual_feed_status": visual_feed,
        "worker_fallback_operational": worker_fallback,
        "visual_fallback_active": visual_fallback,
        "visual_only_fallback": visual_only_fallback,
        "worker_strategy_reading_reliable": not worker_fallback,
        "fallback_scope": fallback_scope,
        "fallback_scope_status": fallback_scope,
        "fallback_blocker_scope": fallback_scope,
        "daily_budget_limit": daily_limit,
        "daily_budget_source": daily_budget_source,
        "daily_credit_limit_estimate": daily_limit,
        "daily_credits_used_estimate": daily_used,
        "daily_credit_usage_pct": daily_usage_pct,
        "daily_budget_status": daily_status,
        "minute_limit": minute_limit,
        "minute_limit_source": minute_limit_source,
        "minute_limit_estimate": minute_limit,
        "minutely_average": minute_average,
        "minutely_maximum": minute_maximum,
        "minute_limit_status": minute_status,
        "minute_burst_risk": minute_burst,
        "risk_429": has_429,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "stale_cache_hits": stale_cache_hits,
        "cache_status": cache_status,
        "estimated_provider_calls": estimated_provider_calls,
        "provider_calls_attempted": provider_calls_attempted,
        "provider_calls_skipped": _safe_int(worker_status.get("provider_calls_skipped")),
        "requested_by_worker": _safe_text(worker_status.get("requested_by"), "worker_cycle"),
        "requested_by_visual": _safe_text(visual_status.get("requested_by"), ""),
        "budget_block_reason": budget_block_reason,
        "provider_budget_status": provider_budget_status,
        "ui_alerts": alerts,
        "blocked_actions": list(BLOCKED_ACTIONS),
        "recommendation": recommendation,
        "provider_budget_recommendation": recommendation,
        "notes": notes,
        "paper_required": True,
        "observability_only": True,
        "diagnostic_only": True,
        "shadow_only": True,
        "should_continue_paper": True,
        "should_start_real_money": False,
        "should_change_threshold_now": False,
        "should_change_score_now": False,
        "should_change_broker_now": False,
        "should_change_provider_now": False,
        "should_change_profile_now": False,
        "should_apply_micro_adjustment_now": False,
        "should_advance_2_6c_now": False,
        "trade_authority": False,
        "score_authority": False,
        "broker_authority": False,
        "provider_authority": False,
        "threshold_authority": False,
        "execution_authority": False,
        "can_approve_trade": False,
        "can_change_provider": False,
        "can_change_threshold": False,
        "can_change_score": False,
        "can_change_broker": False,
        "can_advance_2_6c": False,
    }
    return audit


def build_provider_budget_visual_fallback_log_lines(audit_payload: Mapping[str, Any] | None) -> list[str]:
    audit = _as_dict(audit_payload)
    if not audit:
        return []
    alerts = [item for item in _as_list(audit.get("ui_alerts")) if isinstance(item, Mapping)]
    blocked_actions = [str(item) for item in _as_list(audit.get("blocked_actions"))]
    base = (
        f'phase="{PHASE}";'
        f"mode={_log_text(audit.get('mode'), MODE)};"
        f"diagnostic_mode={_log_text(audit.get('diagnostic_mode'), DIAGNOSTIC_MODE)};"
        f"safety_mode={_log_text(audit.get('safety_mode'), SAFETY_MODE)};"
        "observability_only=true;"
        "paper_required=true"
    )
    return [
        (
            f"{PROVIDER_BUDGET_VISUAL_FALLBACK_MARKERS[0]} "
            f"{base};"
            f"status={_log_text(audit.get('provider_budget_status'), 'UNKNOWN')};"
            f"worker_feed={_log_text(audit.get('worker_feed_status'), 'UNKNOWN')};"
            f"visual_feed={_log_text(audit.get('visual_feed_status'), 'UNKNOWN')};"
            f"provider_worker={_log_text(audit.get('provider_effective_worker'), 'unknown')};"
            f"provider_visual={_log_text(audit.get('provider_effective_visual'), 'unknown')};"
            f"recommendation={_log_text(audit.get('recommendation'), 'insufficient_data')}"
        ),
        (
            f"{PROVIDER_BUDGET_VISUAL_FALLBACK_MARKERS[1]} "
            f"{base};"
            f"daily_used={_log_text(audit.get('daily_credits_used_estimate'), 'unknown')};"
            f"daily_limit={_log_text(audit.get('daily_credit_limit_estimate'), str(TWELVEDATA_DAILY_CREDIT_LIMIT_ESTIMATE))};"
            f"daily_budget_limit={_log_text(audit.get('daily_budget_limit'), str(TWELVEDATA_DAILY_CREDIT_LIMIT_ESTIMATE))};"
            f"daily_budget_source={_log_text(audit.get('daily_budget_source'), 'unknown')};"
            f"daily_status={_log_text(audit.get('daily_budget_status'), 'UNKNOWN')};"
            f"minute_avg={_log_text(audit.get('minutely_average'), 'unknown')};"
            f"minute_max={_log_text(audit.get('minutely_maximum'), 'unknown')};"
            f"minute_limit={_log_text(audit.get('minute_limit'), str(TWELVEDATA_MINUTE_LIMIT_ESTIMATE))};"
            f"minute_limit_source={_log_text(audit.get('minute_limit_source'), 'unknown')};"
            f"minute_status={_log_text(audit.get('minute_limit_status'), 'UNKNOWN')};"
            f"risk_429={str(bool(audit.get('risk_429', False))).lower()}"
        ),
        (
            f"{PROVIDER_BUDGET_VISUAL_FALLBACK_MARKERS[2]} "
            f"{base};"
            f"fallback_scope={_log_text(audit.get('fallback_scope'), 'UNKNOWN')};"
            f"worker_fallback_operational={str(bool(audit.get('worker_fallback_operational', False))).lower()};"
            f"visual_fallback_active={str(bool(audit.get('visual_fallback_active', False))).lower()};"
            f"visual_only_fallback={str(bool(audit.get('visual_only_fallback', False))).lower()};"
            f"worker_strategy_reading_reliable={str(bool(audit.get('worker_strategy_reading_reliable', True))).lower()};"
            f"alerts_count={len(alerts)}"
        ),
        (
            f"{PROVIDER_BUDGET_VISUAL_FALLBACK_MARKERS[3]} "
            f"{base};"
            f"cache_hits={int(audit.get('cache_hits', 0) or 0)};"
            f"cache_misses={int(audit.get('cache_misses', 0) or 0)};"
            f"stale_cache_hits={int(audit.get('stale_cache_hits', 0) or 0)};"
            f"cache_status={_log_text(audit.get('cache_status'), 'UNKNOWN')};"
            f"calls_attempted={int(audit.get('provider_calls_attempted', 0) or 0)};"
            f"estimated_calls={int(audit.get('estimated_provider_calls', 0) or 0)};"
            f"calls_skipped={int(audit.get('provider_calls_skipped', 0) or 0)}"
        ),
        (
            f"{PROVIDER_BUDGET_VISUAL_FALLBACK_MARKERS[4]} "
            f"{base};"
            "should_continue_paper=true;"
            "should_start_real_money=false;"
            "should_change_threshold_now=false;"
            "should_change_score_now=false;"
            "should_change_broker_now=false;"
            "should_change_provider_now=false;"
            "should_change_profile_now=false;"
            "should_apply_micro_adjustment_now=false;"
            "should_advance_2_6c_now=false;"
            "trade_authority=false;"
            "score_authority=false;"
            "broker_authority=false;"
            "provider_authority=false;"
            "threshold_authority=false;"
            "execution_authority=false;"
            "can_approve_trade=false;"
            "can_change_provider=false;"
            "can_change_threshold=false;"
            "can_change_score=false;"
            "can_change_broker=false;"
            "can_advance_2_6c=false;"
            f"blocked_actions_count={len(blocked_actions)}"
        ),
    ]


__all__ = [
    "BLOCKED_ACTIONS",
    "PROVIDER_BUDGET_VISUAL_FALLBACK_MARKERS",
    "build_provider_budget_visual_fallback_audit",
    "build_provider_budget_visual_fallback_log_lines",
    "default_provider_budget_visual_fallback_state",
]
