from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import (
    BASE_DIR,
    MACRO_ALERT_EVENT_WINDOW_MINUTES,
    MACRO_ALERT_POST_EVENT_MINUTES,
    MACRO_ALERT_PRE_EVENT_MINUTES,
    MACRO_ALERTS_ENABLED,
    MACRO_ALERTS_FILE,
)

WINDOW_INACTIVE = "INACTIVE"
WINDOW_PRE_EVENT = "PRE_EVENT"
WINDOW_IN_EVENT = "IN_EVENT"
WINDOW_POST_EVENT = "POST_EVENT"

IMPACT_WEIGHTS = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}

PENALTY_BY_LEVEL_AND_WINDOW = {
    ("MEDIUM", WINDOW_PRE_EVENT): 0.01,
    ("MEDIUM", WINDOW_IN_EVENT): 0.02,
    ("MEDIUM", WINDOW_POST_EVENT): 0.01,
    ("HIGH", WINDOW_PRE_EVENT): 0.04,
    ("HIGH", WINDOW_IN_EVENT): 0.08,
    ("HIGH", WINDOW_POST_EVENT): 0.03,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def _safe_text(value: object) -> str:
    return str(value or "").strip()


def _parse_datetime(value: object) -> datetime | None:
    raw = _safe_text(value)
    if not raw:
        return None
    try:
        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_impact_level(value: object) -> str:
    raw = _safe_text(value).upper()
    aliases = {
        "3": "HIGH",
        "HIGH": "HIGH",
        "ALTO": "HIGH",
        "2": "MEDIUM",
        "MEDIUM": "MEDIUM",
        "MEDIO": "MEDIUM",
        "1": "LOW",
        "LOW": "LOW",
        "BAIXO": "LOW",
    }
    return aliases.get(raw, "LOW")


def default_macro_alert_state(*, enabled: bool | None = None, reason: str = "") -> dict[str, Any]:
    is_enabled = MACRO_ALERTS_ENABLED if enabled is None else bool(enabled)
    return {
        "macro_alert_enabled": bool(is_enabled),
        "macro_alert_active": False,
        "macro_alert_level": "LOW",
        "macro_alert_currency": "",
        "macro_alert_title": "",
        "macro_alert_time": "",
        "macro_alert_window_status": WINDOW_INACTIVE,
        "macro_alert_minutes_to_event": None,
        "macro_alert_reason": reason or "Nenhum evento macro ativo.",
        "macro_alert_penalty": 0.0,
        "macro_alert_blocks_new_entries": False,
        "macro_alert_last_update_ts": utc_now_iso(),
        "macro_alert_event_count": 0,
        "macro_alert_source": "none",
    }


def _event_candidates_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item or {}) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        events = payload.get("events")
        if isinstance(events, list):
            return [dict(item or {}) for item in events if isinstance(item, dict)]
        return [dict(payload)]
    return []


def load_macro_events(file_path: str | None = None) -> list[dict[str, Any]]:
    raw_path = _safe_text(file_path or MACRO_ALERTS_FILE)
    if not raw_path:
        return []

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return _event_candidates_from_payload(payload)


def _event_window_status(
    event_time: datetime,
    *,
    now: datetime,
    pre_minutes: int,
    event_window_minutes: int,
    post_minutes: int,
) -> tuple[str, float]:
    minutes_to_event = (event_time - now).total_seconds() / 60.0
    if 0 < minutes_to_event <= pre_minutes:
        return WINDOW_PRE_EVENT, minutes_to_event
    if -event_window_minutes <= minutes_to_event <= 0:
        return WINDOW_IN_EVENT, minutes_to_event
    if -(event_window_minutes + post_minutes) <= minutes_to_event < -event_window_minutes:
        return WINDOW_POST_EVENT, minutes_to_event
    return WINDOW_INACTIVE, minutes_to_event


def _macro_reason(title: str, currency: str, level: str, window_status: str) -> str:
    label = {
        WINDOW_PRE_EVENT: "Evento macro se aproximando",
        WINDOW_IN_EVENT: "Evento macro em andamento",
        WINDOW_POST_EVENT: "Janela de estabilizacao pos-evento",
    }.get(window_status, "Nenhum evento macro ativo")
    parts = [label]
    if level:
        parts.append(f"impacto {level}")
    if currency:
        parts.append(str(currency).upper())
    if title:
        parts.append(title)
    return " | ".join(parts)


def evaluate_macro_alert(
    events: list[dict[str, Any]] | None = None,
    *,
    now: datetime | None = None,
    enabled: bool | None = None,
    pre_minutes: int | None = None,
    event_window_minutes: int | None = None,
    post_minutes: int | None = None,
) -> dict[str, Any]:
    current_time = (now or utc_now()).astimezone(timezone.utc)
    is_enabled = MACRO_ALERTS_ENABLED if enabled is None else bool(enabled)
    if not is_enabled:
        return default_macro_alert_state(enabled=False, reason="Alertas macro desativados.")

    source_events = list(events if events is not None else load_macro_events())
    pre = max(0, int(MACRO_ALERT_PRE_EVENT_MINUTES if pre_minutes is None else pre_minutes))
    event_window = max(1, int(MACRO_ALERT_EVENT_WINDOW_MINUTES if event_window_minutes is None else event_window_minutes))
    post = max(0, int(MACRO_ALERT_POST_EVENT_MINUTES if post_minutes is None else post_minutes))

    active_candidates: list[dict[str, Any]] = []
    for raw_event in source_events:
        event = dict(raw_event or {})
        if event.get("enabled") is False:
            continue
        event_time = _parse_datetime(event.get("datetime") or event.get("time") or event.get("event_time"))
        if event_time is None:
            continue
        window_status, minutes_to_event = _event_window_status(
            event_time,
            now=current_time,
            pre_minutes=pre,
            event_window_minutes=event_window,
            post_minutes=post,
        )
        if window_status == WINDOW_INACTIVE:
            continue
        level = _normalize_impact_level(event.get("impact_level"))
        active_candidates.append(
            {
                "event": event,
                "event_time": event_time,
                "window_status": window_status,
                "minutes_to_event": minutes_to_event,
                "impact_level": level,
            }
        )

    if not active_candidates:
        state = default_macro_alert_state(enabled=True, reason="Nenhum evento macro dentro da janela de risco.")
        state["macro_alert_event_count"] = len(source_events)
        state["macro_alert_source"] = "manual"
        return state

    active_candidates.sort(
        key=lambda item: (
            IMPACT_WEIGHTS.get(str(item["impact_level"]), 0),
            -abs(float(item["minutes_to_event"])),
        ),
        reverse=True,
    )
    selected = active_candidates[0]
    event = dict(selected["event"])
    level = str(selected["impact_level"])
    window_status = str(selected["window_status"])
    title = _safe_text(event.get("title") or event.get("name"))
    currency = _safe_text(event.get("currency")).upper()
    penalty = float(PENALTY_BY_LEVEL_AND_WINDOW.get((level, window_status), 0.0))
    blocks_new_entries = bool(level == "HIGH" and window_status == WINDOW_IN_EVENT)

    return {
        "macro_alert_enabled": True,
        "macro_alert_active": True,
        "macro_alert_level": level,
        "macro_alert_currency": currency,
        "macro_alert_title": title,
        "macro_alert_time": selected["event_time"].isoformat(),
        "macro_alert_window_status": window_status,
        "macro_alert_minutes_to_event": round(float(selected["minutes_to_event"]), 2),
        "macro_alert_reason": _macro_reason(title, currency, level, window_status),
        "macro_alert_penalty": round(penalty, 4),
        "macro_alert_blocks_new_entries": blocks_new_entries,
        "macro_alert_last_update_ts": current_time.isoformat(),
        "macro_alert_event_count": len(source_events),
        "macro_alert_source": "manual",
    }


def evaluate_configured_macro_alert(*, now: datetime | None = None) -> dict[str, Any]:
    return evaluate_macro_alert(now=now)


def macro_alert_operational_effect(macro_alert: dict[str, Any] | None) -> str:
    alert = dict(macro_alert or default_macro_alert_state())
    if not bool(alert.get("macro_alert_active")):
        return "Sem restricao macro ativa. O robo segue usando apenas os filtros internos."

    level = str(alert.get("macro_alert_level") or "LOW").upper()
    window_status = str(alert.get("macro_alert_window_status") or WINDOW_INACTIVE).upper()
    penalty = float(alert.get("macro_alert_penalty", 0.0) or 0.0)
    if level == "HIGH" and window_status == WINDOW_IN_EVENT:
        return (
            "Alerta macro alto em janela de evento: setups frageis ficam bloqueados, "
            "fallback bloqueia entrada e tendencia forte exige confirmacao maior."
        )
    if level == "HIGH":
        return (
            f"Alerta macro alto: confianca reduzida em {penalty:.2f}, setups frageis restritos "
            "e tendencia exige margem adicional."
        )
    if level == "MEDIUM":
        return f"Alerta macro medio: confianca reduzida em {penalty:.2f} para setups mais frageis."
    return "Alerta macro baixo: apenas observacao operacional, sem bloqueio adicional."


def _setup_family(signal_context: dict[str, Any]) -> str:
    trend_aligned = str(signal_context.get("trend_alignment") or "").lower() == "aligned"
    trend_ok = bool(signal_context.get("trend_ok")) and trend_aligned
    if bool(signal_context.get("breakout_ok")) and bool(signal_context.get("momentum_ok")) and trend_aligned:
        return "BREAKOUT_CONFIRMED"
    if trend_ok:
        return "TREND_FOLLOWING_SWING"
    return "REVERSAL_V_PATTERN"


def apply_macro_risk_filter(signal_context: dict[str, Any], macro_alert: dict[str, Any] | None) -> dict[str, Any]:
    alert = dict(macro_alert or default_macro_alert_state())
    score = float(signal_context.get("score", 0.0) or 0.0)
    minimum = float(signal_context.get("effective_min_signal_score", signal_context.get("base_min_signal_score", 0.0)) or 0.0)
    if not bool(alert.get("macro_alert_active")):
        return {
            "adjusted_score": score,
            "effective_min_signal_score": minimum,
            "blocked": False,
            "reason_code": "",
            "reason": "Nenhum alerta macro ativo.",
            "setup_family": _setup_family(signal_context),
            "penalty": 0.0,
        }

    level = str(alert.get("macro_alert_level") or "LOW").upper()
    window_status = str(alert.get("macro_alert_window_status") or WINDOW_INACTIVE).upper()
    setup_family = _setup_family(signal_context)
    penalty = float(alert.get("macro_alert_penalty", 0.0) or 0.0)
    adjusted_score = max(0.0, round(score - penalty, 4))
    extra_requirement = 0.0
    blocked = False
    reason = str(alert.get("macro_alert_reason") or "Alerta macro ativo.")

    if level == "HIGH":
        if setup_family == "TREND_FOLLOWING_SWING":
            extra_requirement = 0.02 if window_status != WINDOW_IN_EVENT else 0.03
        elif setup_family == "BREAKOUT_CONFIRMED":
            extra_requirement = 0.04
        else:
            extra_requirement = 0.06

        required_score = minimum + extra_requirement
        strong_trend_setup = setup_family == "TREND_FOLLOWING_SWING" and adjusted_score >= required_score + 0.03
        fragile_setup = setup_family != "TREND_FOLLOWING_SWING" or adjusted_score < required_score
        source_label = str(signal_context.get("data_source") or "").strip().lower()
        fallback_source = source_label in {"fallback", "synthetic", "unknown"} or "fallback" in source_label
        event_block_window = bool(alert.get("macro_alert_blocks_new_entries")) and not strong_trend_setup
        if event_block_window or fallback_source or fragile_setup:
            blocked = True
    elif level == "MEDIUM":
        if setup_family != "TREND_FOLLOWING_SWING":
            extra_requirement = 0.02

    return {
        "adjusted_score": adjusted_score,
        "effective_min_signal_score": round(minimum + extra_requirement, 4),
        "blocked": blocked,
        "reason_code": "macro_alert_guard" if blocked or penalty > 0 else "",
        "reason": reason,
        "setup_family": setup_family,
        "penalty": round(penalty, 4),
    }
