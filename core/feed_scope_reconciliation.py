from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


MODE = "DIAGNOSTIC_ONLY"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


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


def _feed_quality_snapshot(status: dict[str, Any]) -> dict[str, Any]:
    payload = dict(status or {})
    breakdown = dict(payload.get("source_breakdown", {}) or {})
    requested = [str(item).upper() for item in list(payload.get("requested_symbols") or payload.get("symbols") or []) if str(item)]
    live_symbols = [str(item).upper() for item in list(payload.get("live_symbols") or []) if str(item)]
    fallback_symbols = [str(item).upper() for item in list(payload.get("fallback_symbols") or []) if str(item)]
    unknown_symbols = [str(item).upper() for item in list(payload.get("unknown_symbols") or []) if str(item)]
    live_count = len(live_symbols) if live_symbols else _safe_int(breakdown.get("market"))
    fallback_count = len(fallback_symbols) if fallback_symbols else _safe_int(breakdown.get("fallback"))
    unknown_count = len(unknown_symbols) if unknown_symbols else _safe_int(breakdown.get("unknown"))
    feed_status = _feed_status(payload.get("feed_status") or payload.get("status"))
    if feed_status == "UNKNOWN":
        if fallback_count > 0 and live_count == 0:
            feed_status = "FALLBACK"
        elif live_count > 0 and fallback_count == 0:
            feed_status = "LIVE"
        elif live_count > 0 or _safe_int(breakdown.get("cached")) > 0:
            feed_status = "DELAYED"
    return {
        "feed_status": feed_status,
        "provider_effective": _safe_text(payload.get("provider_effective") or payload.get("provider")),
        "live_count": live_count,
        "fallback_count": fallback_count,
        "unknown_count": unknown_count,
        "total_symbols": len(requested) or sum(_safe_int(value) for value in breakdown.values()),
    }


def _fallback_like(value: Any) -> bool:
    text = _safe_text(value).lower()
    return "fallback" in text or text in {"synthetic", "sintetico"}


def _candidate_feed_status(row: dict[str, Any]) -> str:
    for key in ("feed_status", "data_source", "source", "entry_data_source"):
        if row.get(key):
            raw = _safe_text(row.get(key)).lower()
            if raw in {"market", "live"}:
                return "LIVE"
            if raw in {"cached", "delayed", "stale"}:
                return "DELAYED"
            if _fallback_like(raw):
                return "FALLBACK"
    reasons = row.get("reasons") or row.get("rejection_reasons") or row.get("risk_blockers") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    if any(_fallback_like(item) for item in list(reasons or [])):
        return "FALLBACK"
    return "UNKNOWN"


def default_feed_scope_reconciliation_state(reason: str = "No feed scope reconciliation data yet.") -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": MODE,
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
        "notes": reason,
    }


def _count_candidate_fallbacks(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if isinstance(row, dict) and _candidate_feed_status(row) == "FALLBACK")


def build_feed_scope_reconciliation(
    *,
    market_data_status: dict[str, Any] | None = None,
    visual_chart_status: dict[str, Any] | None = None,
    validation_state: dict[str, Any] | None = None,
    shadow_decision_simulator: dict[str, Any] | None = None,
    signals: list[dict[str, Any]] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    if not enabled:
        disabled = default_feed_scope_reconciliation_state("Feed scope reconciliation disabled.")
        disabled["enabled"] = False
        return disabled

    worker_status = dict(market_data_status or {})
    visual_status = dict(visual_chart_status or {})
    validation = dict(validation_state or {})
    shadow = dict(shadow_decision_simulator or {})
    worker_quality = _feed_quality_snapshot(worker_status)
    visual_quality = _feed_quality_snapshot(visual_status)

    feed_status = _feed_status(worker_quality.get("feed_status") or worker_status.get("feed_status") or worker_status.get("status"))
    visual_feed = _feed_status(visual_quality.get("feed_status") or visual_status.get("feed_status") or visual_status.get("status"))
    live_count = _safe_int(worker_quality.get("live_count"))
    fallback_count = _safe_int(worker_quality.get("fallback_count"))
    unknown_count = _safe_int(worker_quality.get("unknown_count"))
    provider = _safe_text(worker_quality.get("provider_effective") or worker_status.get("provider_effective") or worker_status.get("provider"))
    accumulated_fallback = max(
        _safe_int(validation.get("fallback_rejection_accumulated_count")),
        _safe_int(shadow.get("fallback_accumulated_count")),
    )
    current_rejection_fallback = _safe_int(validation.get("fallback_rejection_current_cycle_count"))
    accumulated_strategy = _safe_int(validation.get("strategy_rejection_accumulated_count"))
    dominant_current = _safe_text(validation.get("current_cycle_rejection_reason") or shadow.get("dominant_exclusion_current_scope"))
    dominant_accumulated = _safe_text(validation.get("accumulated_rejection_reason") or shadow.get("shadow_dominant_block_reason_accumulated") or shadow.get("shadow_dominant_block_reason"))

    current_rows = [dict(row or {}) for row in list(signals or []) if isinstance(row, dict)]
    current_rows += [dict(row or {}) for row in list(shadow.get("shadow_current_cycle_candidates", []) or []) if isinstance(row, dict)]
    accumulated_rows = []
    accumulated_rows += [
        dict(row or {})
        for row in list(shadow.get("shadow_accumulated_recent_candidates", shadow.get("shadow_recent_candidates", [])) or [])
        if isinstance(row, dict)
    ]
    current_candidate_fallback = _count_candidate_fallbacks(current_rows)
    accumulated_candidate_fallback = _count_candidate_fallbacks(accumulated_rows)
    visual_fallback = _safe_int(visual_quality.get("fallback_count")) > 0 or visual_feed == "FALLBACK"
    current_feed_is_clean = bool(feed_status == "LIVE" and fallback_count == 0 and live_count > 0)

    historical_fallback = int(
        max(0, accumulated_fallback - current_rejection_fallback)
        + (1 if "fallback" in dominant_accumulated.lower() and current_feed_is_clean else 0)
    )

    if not current_feed_is_clean and (feed_status == "FALLBACK" or fallback_count > 0 or current_rejection_fallback > 0):
        fallback_scope_status = "CURRENT_CYCLE_FALLBACK"
        fallback_blocker_scope = "CURRENT_CYCLE"
        recommendation = "check_current_feed"
        notes = "Current worker feed has fallback or current-cycle fallback rejection."
    elif current_feed_is_clean and accumulated_fallback > 0:
        fallback_scope_status = "ACCUMULATED_ONLY_FALLBACK"
        fallback_blocker_scope = "ACCUMULATED"
        recommendation = "accumulated_fallback_only"
        notes = "Fallback is accumulated/historical and does not represent the current clean worker feed."
    elif current_feed_is_clean and accumulated_candidate_fallback > 0:
        fallback_scope_status = "CANDIDATE_LEVEL_OLD_FALLBACK"
        fallback_blocker_scope = "CANDIDATE_OLD"
        recommendation = "candidate_old_fallback_only"
        notes = "Fallback appears only on old accumulated candidate rows."
    elif current_feed_is_clean and visual_fallback:
        fallback_scope_status = "VISUAL_ONLY_FALLBACK"
        fallback_blocker_scope = "VISUAL_CHART_ONLY"
        recommendation = "visual_only_fallback"
        notes = "Fallback appears only in the visual chart context, not in the worker feed."
    elif current_feed_is_clean and historical_fallback > 0:
        fallback_scope_status = "HISTORICAL_ONLY_FALLBACK"
        fallback_blocker_scope = "HISTORICAL"
        recommendation = "historical_fallback_only"
        notes = "Fallback evidence is historical and does not represent the current clean worker feed."
    elif current_feed_is_clean:
        fallback_scope_status = "NO_CURRENT_FALLBACK"
        fallback_blocker_scope = "NONE"
        recommendation = "observe_current_feed_clean"
        notes = "Current worker feed is clean; fallback is not a current-cycle blocker."
    elif historical_fallback > 0:
        fallback_scope_status = "HISTORICAL_ONLY_FALLBACK"
        fallback_blocker_scope = "HISTORICAL"
        recommendation = "historical_fallback_only"
        notes = "Fallback evidence is historical, with no clean current-cycle fallback scope."
    else:
        fallback_scope_status = "UNKNOWN_SCOPE"
        fallback_blocker_scope = "UNKNOWN"
        recommendation = "observe_more"
        notes = "Fallback scope is not clear enough yet."

    return {
        "enabled": True,
        "mode": MODE,
        "generated_at": _utc_now_iso(),
        "provider_effective": provider,
        "current_feed_status": feed_status,
        "current_cycle_feed_status": feed_status,
        "current_cycle_provider": provider,
        "current_live_count": live_count,
        "current_cycle_live_count": live_count,
        "current_fallback_count": fallback_count,
        "current_cycle_fallback_count": fallback_count,
        "current_cycle_unknown_count": unknown_count,
        "visual_feed_status": visual_feed,
        "visual_chart_feed_status": visual_feed,
        "worker_feed_status": feed_status,
        "accumulated_fallback_count": accumulated_fallback,
        "accumulated_strategy_count": accumulated_strategy,
        "historical_fallback_count": historical_fallback,
        "candidate_fallback_flags": {
            "current_cycle_candidate_fallback_count": current_candidate_fallback,
            "accumulated_candidate_fallback_count": accumulated_candidate_fallback,
            "visual_chart_fallback": bool(visual_fallback),
        },
        "dominant_rejection_current": dominant_current,
        "dominant_rejection_accumulated": dominant_accumulated,
        "fallback_scope_status": fallback_scope_status,
        "fallback_blocker_scope": fallback_blocker_scope,
        "current_feed_is_clean": current_feed_is_clean,
        "recommendation": recommendation,
        "notes": notes,
    }
