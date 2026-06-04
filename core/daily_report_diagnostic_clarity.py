"""Daily report diagnostic clarity for FASE 2.6B.3.

This module is report-only. It derives an executive diagnostic block from
existing state/report fields and never mutates state, trading decisions,
scores, thresholds, broker/provider configuration, orders, PnL, history, or
positions.
"""

from __future__ import annotations

from typing import Any, Mapping


PHASE = "2.6B.3"
MODE = "REPORT_ONLY"
OBSERVABILITY_MODE = "OBSERVABILITY_ONLY"
DIAGNOSTIC_MODE = "DIAGNOSTIC_ONLY"

PROHIBITED_ACTIONS = [
    "não operar dinheiro real",
    "não avançar FASE 2.6C",
    "não aplicar microajuste",
    "não alterar threshold",
    "não alterar score",
    "não alterar min_signal_score",
    "não alterar broker",
    "não alterar provider",
    "não alterar capital",
    "não alterar ticket",
    "não alterar max_open_positions",
    "não converter Fibonacci, BOS, pivô, H1/4H, MTF, webhook ou diagnóstico shadow em gatilho",
]

NEXT_PHASE_PRECONDITIONS = [
    "worker LIVE por janela suficiente",
    "fallback operacional 0",
    "provider efetivo confiável",
    "contexto fora de CRÍTICO",
    "sinais near-approved seguros",
    "ausência de FEED_BLOCK dominante",
    "score/RSI/confirmação secundária coerentes",
    "PAPER preservado",
    "revisão humana antes de qualquer PR operacional",
]

NO_COMPARISON_TEXT = "Comparativo indisponível no runtime atual."


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_text(value: Any, fallback: str = "unknown") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else fallback


def _upper(value: Any, fallback: str = "UNKNOWN") -> str:
    return _safe_text(value, fallback).upper()


def _lower(value: Any, fallback: str = "unknown") -> str:
    return _safe_text(value, fallback).lower()


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "sim", "y"}:
        return True
    if text in {"false", "0", "no", "nao", "não", "n"}:
        return False
    return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _first_text(*values: Any, fallback: str = "unknown") -> str:
    for value in values:
        text = str(value if value is not None else "").strip()
        if text:
            return text
    return fallback


def _contains_any(value: Any, tokens: tuple[str, ...]) -> bool:
    haystack = _safe_text(value, "").upper()
    return any(token in haystack for token in tokens)


def _extract_provider_budget(validation_report: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    report = _as_dict(validation_report)
    state_payload = _as_dict(state)
    audit = _as_dict(report.get("provider_budget_visual_fallback")) or _as_dict(
        state_payload.get("provider_budget_visual_fallback")
    )
    market_state = _as_dict(state_payload.get("market_data"))
    feed_scope = _as_dict(report.get("feed_scope_reconciliation")) or _as_dict(
        state_payload.get("feed_scope_reconciliation")
    )

    worker_feed = _upper(
        _first_text(
            audit.get("worker_feed_status"),
            feed_scope.get("current_feed_status"),
            market_state.get("feed_status"),
            fallback="UNKNOWN",
        )
    )
    visual_feed = _upper(
        _first_text(
            audit.get("visual_feed_status"),
            market_state.get("visual_feed_status"),
            fallback=worker_feed,
        )
    )
    worker_provider = _lower(
        _first_text(
            audit.get("provider_effective_worker"),
            market_state.get("provider_effective"),
            market_state.get("provider"),
            fallback="unknown",
        )
    )
    visual_provider = _lower(
        _first_text(
            audit.get("provider_effective_visual"),
            market_state.get("visual_provider_effective"),
            market_state.get("visual_provider"),
            fallback=worker_provider,
        )
    )
    current_fallback_count = _int(
        _first_text(
            feed_scope.get("current_fallback_count"),
            market_state.get("fallback_count"),
            fallback="0",
        )
    )
    worker_fallback = (
        _bool(audit.get("worker_fallback_operational"), False)
        or worker_feed == "FALLBACK"
        or current_fallback_count > 0
    )
    visual_only_fallback = (
        _bool(audit.get("visual_only_fallback"), False)
        or (visual_feed == "FALLBACK" and not worker_fallback)
    )
    risk_429 = _bool(audit.get("risk_429"), False) or _contains_any(
        " ".join(
            [
                _safe_text(audit.get("provider_budget_status"), ""),
                _safe_text(audit.get("budget_block_reason"), ""),
                _safe_text(market_state.get("last_error"), ""),
            ]
        ),
        ("429", "RATE_LIMIT", "LIMIT_EXCEEDED"),
    )

    return {
        "worker_feed": worker_feed,
        "visual_feed": visual_feed,
        "worker_provider": worker_provider,
        "visual_provider": visual_provider,
        "worker_fallback": worker_fallback,
        "visual_only_fallback": visual_only_fallback,
        "current_fallback_count": current_fallback_count,
        "risk_429": risk_429,
        "daily_limit": audit.get("daily_budget_limit") or audit.get("daily_credit_limit_estimate") or "unknown",
        "daily_source": _safe_text(audit.get("daily_budget_source"), "unknown"),
        "daily_budget": _safe_text(audit.get("daily_budget_status"), "UNKNOWN"),
        "minute_limit": audit.get("minute_limit") or audit.get("minute_limit_estimate") or "unknown",
        "minute_source": _safe_text(audit.get("minute_limit_source"), "unknown"),
        "minute_status": _safe_text(audit.get("minute_limit_status"), "UNKNOWN"),
        "provider_budget_status": _safe_text(audit.get("provider_budget_status"), "UNKNOWN"),
        "recommendation": _safe_text(
            audit.get("provider_budget_recommendation") or audit.get("recommendation"),
            "insufficient_data",
        ),
    }


def _feed_problem(provider_budget: Mapping[str, Any], validation_report: Mapping[str, Any]) -> bool:
    feed_rejection = _as_dict(validation_report.get("feed_rejection_consistency"))
    feed_scope = _as_dict(validation_report.get("feed_scope_reconciliation"))
    tokens = " ".join(
        [
            _safe_text(provider_budget.get("provider_budget_status"), ""),
            _safe_text(feed_rejection.get("top_layer"), ""),
            _safe_text(feed_rejection.get("dominant_rejection_scope"), ""),
            _safe_text(feed_scope.get("fallback_scope_status"), ""),
            _safe_text(feed_scope.get("fallback_blocker_scope"), ""),
        ]
    )
    return (
        bool(provider_budget.get("worker_fallback"))
        or bool(provider_budget.get("risk_429"))
        or _contains_any(tokens, ("FEED_BLOCK", "FEED_OR_PROVIDER", "WORKER_OPERATIONAL_FALLBACK"))
    )


def _context_status(validation_report: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    context_state = _as_dict(_as_dict(state).get("market_context"))
    study = _as_dict(_as_dict(validation_report).get("controlled_micro_adjustment_study"))
    return _upper(
        _first_text(
            validation_report.get("context_status"),
            study.get("market_context_status"),
            context_state.get("market_context_status"),
            context_state.get("context_status"),
            fallback="UNKNOWN",
        )
    )


def _main_blocker(validation_report: Mapping[str, Any]) -> str:
    bottleneck = _as_dict(validation_report.get("strategy_bottleneck"))
    taxonomy = _as_dict(validation_report.get("setup_blocker_taxonomy_audit"))
    study = _as_dict(validation_report.get("controlled_micro_adjustment_study"))
    rejection = _as_dict(validation_report.get("rejection_quality"))
    return _upper(
        _first_text(
            bottleneck.get("dominant_bottleneck"),
            taxonomy.get("normalized_primary_reason"),
            taxonomy.get("official_primary_blocker"),
            study.get("dominant_bottleneck"),
            rejection.get("top_reason"),
            fallback="UNKNOWN",
        )
    )


def _operational_status(
    *,
    provider_budget: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    state: Mapping[str, Any],
) -> str:
    production = _as_dict(_as_dict(state).get("production"))
    worker_status = _lower(_as_dict(state).get("worker_status"), "unknown")
    health_level = _upper(production.get("health_level"), "HEALTHY")
    consecutive_errors = _int(production.get("consecutive_errors"), 0)

    if worker_status in {"offline", "stopped", "error"} or health_level in {"ERROR", "CRITICAL", "DOWN"}:
        return "ERRO OPERACIONAL"
    if _feed_problem(provider_budget, validation_report):
        return "BLOQUEADO POR FEED"
    if bool(provider_budget.get("visual_only_fallback")) or consecutive_errors > 0:
        return "ATENÇÃO"
    return "SAUDÁVEL"


def _strategic_status(
    *,
    provider_budget: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    state: Mapping[str, Any],
) -> str:
    if _feed_problem(provider_budget, validation_report):
        return "BLOQUEADO POR FEED"

    context = _context_status(validation_report, state)
    if context in {"CRITICO", "CRÍTICO", "DESFAVORAVEL", "DESFAVORÁVEL"}:
        return "BLOQUEADO POR CONTEXTO"

    blocker = _main_blocker(validation_report)
    if "SCORE_BELOW" in blocker or blocker in {"SCORE_BELOW_MIN", "SCORE_BELOW_MINIMUM"}:
        return "BLOQUEADO POR SCORE"
    if _contains_any(
        blocker,
        (
            "SECONDARY_CONFIRMATION",
            "BREAKOUT",
            "BOS",
            "NO_SETUP",
            "PULLBACK",
            "CONFIRMATION",
        ),
    ):
        return "BLOQUEADO POR CONFIRMAÇÃO"
    if _contains_any(blocker, ("INSUFFICIENT", "UNKNOWN")):
        return "BLOQUEADO POR DADOS INSUFICIENTES"
    return "LIBERADO PARA OBSERVAÇÃO EM PAPER"


def _reading_reliability(provider_budget: Mapping[str, Any], validation_report: Mapping[str, Any]) -> str:
    if _feed_problem(provider_budget, validation_report) or _lower(provider_budget.get("worker_provider")) == "synthetic":
        return "BAIXA"
    if provider_budget.get("worker_feed") == "LIVE" and bool(provider_budget.get("visual_only_fallback")):
        return "MÉDIA"
    if provider_budget.get("worker_feed") == "LIVE" and not bool(provider_budget.get("worker_fallback")):
        return "ALTA"
    return "MÉDIA"


def _recommended_action(strategic_status: str, main_blocker: str, provider_budget: Mapping[str, Any]) -> str:
    if strategic_status == "BLOQUEADO POR FEED":
        return "Manter PAPER e tratar leitura de feed/provider; nenhuma decisão operacional autorizada."
    if strategic_status == "BLOQUEADO POR CONTEXTO":
        return "Observar mais; contexto não autoriza microajuste, dinheiro real ou FASE 2.6C."
    if strategic_status == "BLOQUEADO POR SCORE":
        return "Observar score em PAPER; não alterar threshold, score ou min_signal_score."
    if strategic_status == "BLOQUEADO POR CONFIRMAÇÃO":
        return "Estudar confirmação em diagnóstico; manter bloqueado até critério real validado."
    if strategic_status == "BLOQUEADO POR DADOS INSUFICIENTES":
        return "Coletar mais dados em PAPER; não inferir comparativo ou autorização operacional."
    if bool(provider_budget.get("visual_only_fallback")):
        return "Separar fallback visual do feed operacional; seguir apenas em observação PAPER."
    return f"Observar em PAPER sem mudança operacional; blocker principal={main_blocker}."


def _comparison_notes(
    current: Mapping[str, Any],
    previous_report: Mapping[str, Any] | None,
    state: Mapping[str, Any],
) -> list[str]:
    previous = _as_dict(previous_report)
    if not previous:
        previous = _as_dict(_as_dict(state).get("previous_daily_report"))
    if not previous:
        previous = _as_dict(_as_dict(state).get("daily_report_previous"))
    if not previous:
        return [NO_COMPARISON_TEXT]

    notes: list[str] = []
    for key, label in (
        ("operational_status", "Status operacional"),
        ("strategic_status", "Status estratégico"),
        ("reading_reliability", "Confiabilidade"),
        ("main_blocker", "Blocker principal"),
    ):
        before = _safe_text(previous.get(key), "unknown")
        after = _safe_text(current.get(key), "unknown")
        if before != after:
            notes.append(f"{label}: {before} -> {after}")
    return notes or ["Sem mudança relevante registrada no comparativo disponível."]


def build_atlas_daily_decision(
    validation_report: Mapping[str, Any] | None,
    *,
    state: Mapping[str, Any] | None = None,
    previous_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the read-only "Decisão Atlas do Dia" payload."""

    report = _as_dict(validation_report)
    state_payload = _as_dict(state)
    provider_budget = _extract_provider_budget(report, state_payload)
    main_blocker = _main_blocker(report)
    operational_status = _operational_status(
        provider_budget=provider_budget,
        validation_report=report,
        state=state_payload,
    )
    strategic_status = _strategic_status(
        provider_budget=provider_budget,
        validation_report=report,
        state=state_payload,
    )
    reliability = _reading_reliability(provider_budget, report)

    payload: dict[str, Any] = {
        "enabled": True,
        "phase": PHASE,
        "mode": MODE,
        "observability_mode": OBSERVABILITY_MODE,
        "diagnostic_mode": DIAGNOSTIC_MODE,
        "operational_status": operational_status,
        "strategic_status": strategic_status,
        "reading_reliability": reliability,
        "worker_feed": provider_budget["worker_feed"],
        "visual_feed": provider_budget["visual_feed"],
        "worker_provider": provider_budget["worker_provider"],
        "visual_provider": provider_budget["visual_provider"],
        "worker_fallback": bool(provider_budget["worker_fallback"]),
        "visual_only_fallback": bool(provider_budget["visual_only_fallback"]),
        "current_fallback_count": int(provider_budget["current_fallback_count"]),
        "risk_429": bool(provider_budget["risk_429"]),
        "main_blocker": main_blocker,
        "recommended_action": _recommended_action(strategic_status, main_blocker, provider_budget),
        "prohibited_actions": list(PROHIBITED_ACTIONS),
        "next_phase_preconditions": list(NEXT_PHASE_PRECONDITIONS),
        "provider_budget_summary": dict(provider_budget),
        "paper_required": True,
        "report_only": True,
        "observability_only": True,
        "diagnostic_only": True,
        "trade_authority": False,
        "score_authority": False,
        "broker_authority": False,
        "provider_authority": False,
        "threshold_authority": False,
        "execution_authority": False,
        "can_advance_2_6c": False,
        "notes": (
            "REPORT_ONLY / OBSERVABILITY_ONLY / DIAGNOSTIC_ONLY. "
            "Este bloco explica o estado do robô e não altera decisão operacional."
        ),
    }
    payload["what_changed_since_yesterday"] = _comparison_notes(payload, previous_report, state_payload)
    return payload


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _format_section_items(items: list[Any]) -> list[str]:
    return [f"- {_safe_text(item, 'unknown')}" for item in items]


def format_atlas_daily_decision_block(payload: Mapping[str, Any] | None) -> str:
    """Format the executive diagnostic block for the plain-text daily email."""

    data = _as_dict(payload)
    provider_budget = _as_dict(data.get("provider_budget_summary"))
    lines = [
        "Decisão Atlas do Dia",
        f"- Modo: {MODE} / {OBSERVABILITY_MODE} / {DIAGNOSTIC_MODE}",
        f"- Status operacional: {_safe_text(data.get('operational_status'), 'UNKNOWN')}",
        f"- Status estratégico: {_safe_text(data.get('strategic_status'), 'UNKNOWN')}",
        f"- Confiabilidade da leitura: {_safe_text(data.get('reading_reliability'), 'UNKNOWN')}",
        f"- Feed operacional do worker: {_safe_text(data.get('worker_feed'), 'UNKNOWN')}",
        f"- Feed visual do gráfico: {_safe_text(data.get('visual_feed'), 'UNKNOWN')}",
        f"- Provider efetivo do worker: {_safe_text(data.get('worker_provider'), 'unknown')}",
        f"- Provider visual: {_safe_text(data.get('visual_provider'), 'unknown')}",
        f"- Fallback operacional do worker: {_format_bool(data.get('worker_fallback'))}",
        f"- Fallback apenas visual: {_format_bool(data.get('visual_only_fallback'))}",
        f"- Principal motivo de bloqueio: {_safe_text(data.get('main_blocker'), 'UNKNOWN')}",
        f"- Ação recomendada: {_safe_text(data.get('recommended_action'), 'Observar em PAPER.')}",
        "",
        "O que mudou desde ontem",
        *_format_section_items(list(data.get("what_changed_since_yesterday", []) or [NO_COMPARISON_TEXT])),
        "",
        "O que está proibido hoje",
        *_format_section_items(list(data.get("prohibited_actions", []) or PROHIBITED_ACTIONS)),
        "",
        "Pré-condições para próxima fase",
        *_format_section_items(list(data.get("next_phase_preconditions", []) or NEXT_PHASE_PRECONDITIONS)),
        "",
        "Resumo de Provider Budget",
        f"- worker_feed={_safe_text(provider_budget.get('worker_feed'), _safe_text(data.get('worker_feed'), 'UNKNOWN'))}",
        f"- visual_feed={_safe_text(provider_budget.get('visual_feed'), _safe_text(data.get('visual_feed'), 'UNKNOWN'))}",
        f"- worker_provider={_safe_text(provider_budget.get('worker_provider'), _safe_text(data.get('worker_provider'), 'unknown'))}",
        f"- visual_provider={_safe_text(provider_budget.get('visual_provider'), _safe_text(data.get('visual_provider'), 'unknown'))}",
        f"- daily_limit={_safe_text(provider_budget.get('daily_limit'), 'unknown')}",
        f"- daily_source={_safe_text(provider_budget.get('daily_source'), 'unknown')}",
        f"- daily_budget={_safe_text(provider_budget.get('daily_budget'), 'UNKNOWN')}",
        f"- minute_limit={_safe_text(provider_budget.get('minute_limit'), 'unknown')}",
        f"- minute_source={_safe_text(provider_budget.get('minute_source'), 'unknown')}",
        f"- minute_status={_safe_text(provider_budget.get('minute_status'), 'UNKNOWN')}",
        f"- worker_fallback={_format_bool(data.get('worker_fallback'))}",
        f"- visual_only_fallback={_format_bool(data.get('visual_only_fallback'))}",
        f"- risk_429={_format_bool(data.get('risk_429'))}",
        f"- recommendation={_safe_text(provider_budget.get('recommendation'), 'insufficient_data')}",
        "- estimated/configured não é medição real.",
        "- visual fallback não deve ser confundido com fallback operacional.",
        "- worker FALLBACK torna leitura estratégica não confiável.",
        "",
        "Safety",
        "- PAPER TRADING obrigatório; nenhuma ordem real autorizada.",
        "- trade_authority=false; score_authority=false; broker_authority=false; "
        "provider_authority=false; threshold_authority=false; execution_authority=false; can_advance_2_6c=false.",
    ]
    return "\n".join(lines)


__all__ = [
    "MODE",
    "OBSERVABILITY_MODE",
    "DIAGNOSTIC_MODE",
    "NO_COMPARISON_TEXT",
    "PROHIBITED_ACTIONS",
    "NEXT_PHASE_PRECONDITIONS",
    "build_atlas_daily_decision",
    "format_atlas_daily_decision_block",
]
