from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any


DEFAULT_STRATEGY_NAME = "trend_pullback_breakout"

REJECTION_REASON_SPECS: dict[str, dict[str, str]] = {
    "score_below_minimum": {
        "human_reason": "Score ajustado ficou abaixo do minimo exigido.",
        "layer": "strategy",
        "strategy_name": DEFAULT_STRATEGY_NAME,
    },
    "trend_not_confirmed": {
        "human_reason": "Tendencia principal nao foi confirmada para entrada.",
        "layer": "strategy",
        "strategy_name": DEFAULT_STRATEGY_NAME,
    },
    "breakout_not_confirmed": {
        "human_reason": "Breakout minimo nao foi confirmado no setup atual.",
        "layer": "strategy",
        "strategy_name": DEFAULT_STRATEGY_NAME,
    },
    "reversal_not_eligible": {
        "human_reason": "RSI ficou fora da faixa elegivel do setup.",
        "layer": "strategy",
        "strategy_name": DEFAULT_STRATEGY_NAME,
    },
    "volatility_out_of_range": {
        "human_reason": "Volatilidade ficou fora da faixa operacional permitida.",
        "layer": "strategy",
        "strategy_name": DEFAULT_STRATEGY_NAME,
    },
    "context_blocked": {
        "human_reason": "Contexto de mercado bloqueou o sinal no ciclo.",
        "layer": "context",
        "strategy_name": "context_filter",
    },
    "feed_quality_blocked": {
        "human_reason": "Qualidade do feed nao foi considerada confiavel para entrada.",
        "layer": "feed",
        "strategy_name": "feed_guard",
    },
    "fallback_blocked": {
        "human_reason": "Ativo caiu em fallback sintetico e o sinal foi bloqueado.",
        "layer": "feed",
        "strategy_name": "feed_guard",
    },
    "provider_unknown": {
        "human_reason": "Origem do dado nao foi identificada com seguranca.",
        "layer": "feed",
        "strategy_name": "feed_guard",
    },
    "cooldown_active": {
        "human_reason": "Ativo ainda esta em cooldown de reentrada.",
        "layer": "guard",
        "strategy_name": "runtime_guard",
    },
    "duplicate_signal_blocked": {
        "human_reason": "Ativo ja possui posicao aberta ou sinal equivalente em andamento.",
        "layer": "guard",
        "strategy_name": "runtime_guard",
    },
    "position_limit_reached": {
        "human_reason": "Limite de posicoes abertas foi atingido neste ciclo.",
        "layer": "guard",
        "strategy_name": "runtime_guard",
    },
    "schedule_blocked": {
        "human_reason": "Novas entradas estavam bloqueadas pelo estado operacional do robo.",
        "layer": "guard",
        "strategy_name": "runtime_guard",
    },
    "no_setup_eligible": {
        "human_reason": "Nao houve estrutura elegivel para o setup principal.",
        "layer": "strategy",
        "strategy_name": DEFAULT_STRATEGY_NAME,
    },
    "confidence_too_low": {
        "human_reason": "Momentum ou confirmacao secundaria ficaram fracos demais.",
        "layer": "strategy",
        "strategy_name": DEFAULT_STRATEGY_NAME,
    },
    "strategy_conflict": {
        "human_reason": "O sinal entrou em conflito com outra regra da estrategia.",
        "layer": "strategy",
        "strategy_name": DEFAULT_STRATEGY_NAME,
    },
    "daily_loss_guard": {
        "human_reason": "Trava de perda diaria bloqueou novas entradas.",
        "layer": "guard",
        "strategy_name": "risk_guard",
    },
    "macro_alert_guard": {
        "human_reason": "Alerta macro de risco restringiu ou bloqueou o setup.",
        "layer": "guard",
        "strategy_name": "macro_risk_guard",
    },
    "other_safe_reason": {
        "human_reason": "Bloqueio seguro adicional aplicado pelo runtime.",
        "layer": "runtime",
        "strategy_name": "runtime_guard",
    },
}

LAYER_LABELS = {
    "strategy": "Estrategia",
    "context": "Contexto",
    "feed": "Feed",
    "guard": "Guards",
    "runtime": "Runtime",
}

DOMINANT_LAYER_MESSAGES = {
    "strategy": "rejeicao majoritariamente estrategica",
    "context": "rejeicao majoritariamente por contexto",
    "feed": "rejeicao majoritariamente operacional",
    "guard": "rejeicao majoritariamente por guards",
    "runtime": "rejeicao majoritariamente operacional",
}

NON_SETUP_STRATEGY_NAMES = {"context_filter", "feed_guard", "runtime_guard", "risk_guard", "macro_risk_guard"}

RAW_REASON_MAP = {
    "score_below_minimum": "score_below_minimum",
    "weak_score": "score_below_minimum",
    "trend_not_confirmed": "trend_not_confirmed",
    "against_trend": "trend_not_confirmed",
    "breakout_not_confirmed": "breakout_not_confirmed",
    "reversal_not_eligible": "reversal_not_eligible",
    "rsi_filter": "reversal_not_eligible",
    "volatility_out_of_range": "volatility_out_of_range",
    "atr_filter": "volatility_out_of_range",
    "context_blocked": "context_blocked",
    "cooldown_active": "cooldown_active",
    "duplicate_signal_blocked": "duplicate_signal_blocked",
    "position_limit_reached": "position_limit_reached",
    "schedule_blocked": "schedule_blocked",
    "no_setup_eligible": "no_setup_eligible",
    "structure_filter": "no_setup_eligible",
    "confidence_too_low": "confidence_too_low",
    "momentum_filter": "confidence_too_low",
    "strategy_conflict": "strategy_conflict",
    "daily_loss_guard": "daily_loss_guard",
    "macro_alert_guard": "macro_alert_guard",
    "other_safe_reason": "other_safe_reason",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def rejection_reason_label(reason_code: str | None) -> str:
    code = _safe_text(reason_code)
    if not code:
        return "Nao registrado"
    spec = REJECTION_REASON_SPECS.get(code, {})
    return str(spec.get("human_reason") or code.replace("_", " ").capitalize())


def rejection_layer_label(layer: str | None) -> str:
    normalized = _safe_text(layer).lower()
    if not normalized:
        return "Nao registrado"
    return str(LAYER_LABELS.get(normalized, normalized.capitalize()))


def rejection_dominant_message(layer: str | None) -> str:
    normalized = _safe_text(layer).lower()
    return str(DOMINANT_LAYER_MESSAGES.get(normalized, "rejeicao majoritariamente operacional"))


def _canonical_reason_code(raw_reason: str, *, signal: dict[str, Any] | None = None) -> str:
    normalized = _safe_text(raw_reason).lower()
    if normalized == "feed_unreliable":
        data_source = _safe_text((signal or {}).get("data_source")).lower()
        if data_source == "fallback":
            return "fallback_blocked"
        if data_source == "unknown":
            return "provider_unknown"
        return "feed_quality_blocked"
    return RAW_REASON_MAP.get(normalized, normalized or "other_safe_reason")


def build_rejection_event(
    reason_code: str,
    *,
    symbol: str = "",
    timestamp: str = "",
    layer: str | None = None,
    strategy_name: str | None = None,
    human_reason: str | None = None,
    event_key: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_code = _safe_text(reason_code).lower() or "other_safe_reason"
    spec = dict(REJECTION_REASON_SPECS.get(normalized_code, {}) or {})
    resolved_layer = _safe_text(layer) or _safe_text(spec.get("layer")) or "runtime"
    resolved_strategy = _safe_text(strategy_name) or _safe_text(spec.get("strategy_name")) or "runtime_guard"
    resolved_human = _safe_text(human_reason) or _safe_text(spec.get("human_reason")) or "Bloqueio seguro sem detalhe."
    return {
        "reason_code": normalized_code,
        "human_reason": resolved_human,
        "layer": resolved_layer,
        "strategy_name": resolved_strategy,
        "symbol": _safe_text(symbol).upper(),
        "timestamp": _safe_text(timestamp) or _utc_now_iso(),
        "event_key": _safe_text(event_key),
        "metadata": dict(metadata or {}),
    }


def build_signal_rejection_events(signal: dict[str, Any] | None) -> list[dict[str, Any]]:
    payload = dict(signal or {})
    reasons = payload.get("rejection_reasons", []) or []
    if isinstance(reasons, str):
        reasons = [chunk.strip() for chunk in reasons.split(",")]

    symbol = _safe_text(payload.get("asset")).upper()
    timestamp = _safe_text(payload.get("signal_timestamp") or payload.get("timestamp") or payload.get("datetime"))
    signal_key = _safe_text(payload.get("signal_key"))
    strategy_name = _safe_text(payload.get("strategy_name")) or DEFAULT_STRATEGY_NAME

    events: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    for raw_reason in reasons:
        canonical = _canonical_reason_code(str(raw_reason), signal=payload)
        if canonical in seen_codes:
            continue
        seen_codes.add(canonical)
        event_key = f"{signal_key or timestamp or symbol}|{canonical}"
        events.append(
            build_rejection_event(
                canonical,
                symbol=symbol,
                timestamp=timestamp,
                strategy_name=strategy_name,
                event_key=event_key,
                metadata={
                    "raw_reason": _safe_text(raw_reason),
                    "signal_key": signal_key,
                    "data_source": _safe_text(payload.get("data_source")),
                    "context_status": _safe_text(payload.get("context_status")),
                },
            )
        )
    return events


def summarize_rejection_events(
    rejection_events: list[dict[str, Any]] | None,
    *,
    rejected_signal_count: int = 0,
) -> dict[str, Any]:
    events = [dict(item or {}) for item in (rejection_events or []) if isinstance(item, dict)]
    reason_counter: Counter[str] = Counter()
    layer_counter: Counter[str] = Counter()
    strategy_counter: Counter[str] = Counter()

    for event in events:
        reason_code = _safe_text(event.get("reason_code")).lower() or "other_safe_reason"
        layer = _safe_text(event.get("layer")).lower() or "runtime"
        strategy_name = _safe_text(event.get("strategy_name")) or "runtime_guard"
        reason_counter[reason_code] += 1
        layer_counter[layer] += 1
        strategy_counter[strategy_name] += 1

    total_events = int(sum(reason_counter.values()))
    top_reason = reason_counter.most_common(1)[0][0] if reason_counter else ""
    top_layer = layer_counter.most_common(1)[0][0] if layer_counter else ""
    preferred_strategies = [
        (name, count) for name, count in strategy_counter.items() if str(name) not in NON_SETUP_STRATEGY_NAMES
    ]
    if preferred_strategies:
        preferred_strategies.sort(key=lambda item: int(item[1] or 0), reverse=True)
        top_strategy = str(preferred_strategies[0][0])
    else:
        top_strategy = strategy_counter.most_common(1)[0][0] if strategy_counter else ""

    top_reasons: list[dict[str, Any]] = []
    for reason_code, count in reason_counter.most_common(5):
        pct = (float(count) / float(total_events)) if total_events > 0 else 0.0
        spec = REJECTION_REASON_SPECS.get(reason_code, {})
        top_reasons.append(
            {
                "reason_code": reason_code,
                "human_reason": rejection_reason_label(reason_code),
                "count": int(count),
                "pct": round(pct, 4),
                "layer": _safe_text(spec.get("layer")) or "",
                "strategy_name": _safe_text(spec.get("strategy_name")) or "",
            }
        )

    effective_signal_sample = max(int(rejected_signal_count or 0), total_events)
    return {
        "total_rejected_signals": int(rejected_signal_count or 0),
        "total_rejection_events": total_events,
        "rejected_by_reason": dict(reason_counter),
        "rejected_by_layer": dict(layer_counter),
        "rejected_by_strategy": dict(strategy_counter),
        "top_rejection_reason": top_reason,
        "top_rejection_layer": top_layer,
        "top_rejection_strategy": top_strategy,
        "top_reasons": top_reasons,
        "has_minimum_sample": bool(effective_signal_sample >= 8),
        "dominant_message": rejection_dominant_message(top_layer) if top_layer else "",
    }


def update_validation_rejection_state(
    validation_state: dict[str, Any],
    cycle_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(cycle_result, dict):
        return validation_state

    cycle_validation = dict(cycle_result.get("validation_cycle", {}) or {})
    raw_events = [dict(item or {}) for item in (cycle_validation.get("rejection_events") or []) if isinstance(item, dict)]

    seen_keys = dict(validation_state.get("last_rejection_event_keys", {}) or {})
    reason_breakdown = Counter(dict(validation_state.get("rejection_reason_breakdown", {}) or {}))
    layer_breakdown = Counter(dict(validation_state.get("rejection_layer_breakdown", {}) or {}))
    strategy_breakdown = Counter(dict(validation_state.get("rejection_strategy_breakdown", {}) or {}))

    for event in raw_events:
        event_key = _safe_text(event.get("event_key"))
        if event_key and event_key in seen_keys:
            continue
        if event_key:
            seen_keys[event_key] = _safe_text(event.get("timestamp")) or _utc_now_iso()
        reason_code = _safe_text(event.get("reason_code")).lower() or "other_safe_reason"
        layer = _safe_text(event.get("layer")).lower() or "runtime"
        strategy_name = _safe_text(event.get("strategy_name")) or "runtime_guard"
        reason_breakdown[reason_code] += 1
        layer_breakdown[layer] += 1
        strategy_breakdown[strategy_name] += 1

    top_reason = reason_breakdown.most_common(1)[0][0] if reason_breakdown else ""
    top_layer = layer_breakdown.most_common(1)[0][0] if layer_breakdown else ""
    preferred_strategies = [
        (name, count) for name, count in strategy_breakdown.items() if str(name) not in NON_SETUP_STRATEGY_NAMES
    ]
    if preferred_strategies:
        preferred_strategies.sort(key=lambda item: int(item[1] or 0), reverse=True)
        top_strategy = str(preferred_strategies[0][0])
    else:
        top_strategy = strategy_breakdown.most_common(1)[0][0] if strategy_breakdown else ""
    total_rejection_events = int(sum(reason_breakdown.values()))
    rejected_signal_count = int((validation_state.get("signal_counters", {}) or {}).get("signals_rejected", 0) or 0)

    validation_state["rejection_top_reason"] = top_reason
    validation_state["rejection_top_layer"] = top_layer
    validation_state["rejection_top_strategy"] = top_strategy
    validation_state["rejection_reason_breakdown"] = dict(reason_breakdown)
    validation_state["rejection_layer_breakdown"] = dict(layer_breakdown)
    validation_state["rejection_strategy_breakdown"] = dict(strategy_breakdown)
    validation_state["rejection_has_minimum_sample"] = bool(
        max(rejected_signal_count, total_rejection_events) >= 8
    )
    validation_state["last_rejection_event_keys"] = seen_keys
    return validation_state
