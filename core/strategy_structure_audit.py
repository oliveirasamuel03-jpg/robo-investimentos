from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

SETUPS = ("trend_pullback_breakout", "breakout_confirmed", "reversal_v_pattern")
MODE = "SHADOW_ONLY"
MAX_RECENT_CANDIDATES = 15

RISK_REASONS = {
    "fallback_blocked",
    "feed_quality_blocked",
    "provider_unknown",
    "context_blocked",
    "daily_loss_guard",
    "macro_alert_guard",
    "position_limit_reached",
    "schedule_blocked",
}
PRIMARY_REASONS = {
    "trend_not_confirmed",
    "reversal_not_eligible",
    "volatility_out_of_range",
    "no_setup_eligible",
}
SECONDARY_REASONS = {"breakout_not_confirmed", "confidence_too_low", "score_below_minimum"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value: Any) -> bool:
    return bool(value)


def _reason_values(signal: dict[str, Any]) -> list[str]:
    return [str(reason) for reason in list(signal.get("rejection_reasons", []) or []) if str(reason)]


def _score_gap(signal: dict[str, Any]) -> float:
    score = _as_float(signal.get("score"))
    min_score = _as_float(signal.get("effective_min_signal_score"), _as_float(signal.get("base_min_signal_score"), 0.0))
    return round(max(0.0, min_score - score), 6)


def _base_blockers(signal: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    reasons = set(_reason_values(signal))
    primary = sorted(reasons.intersection(PRIMARY_REASONS))
    secondary = sorted(reasons.intersection(SECONDARY_REASONS))
    risk = sorted(reasons.intersection(RISK_REASONS))
    if str(signal.get("data_source") or "").lower() != "market":
        risk.append("feed_not_live")
    if str(signal.get("context_status") or "").upper() == "CRITICO":
        risk.append("context_critical")
    if bool(signal.get("macro_alert_active", False)):
        risk.append("macro_alert_active")
    return sorted(set(primary)), sorted(set(secondary)), sorted(set(risk))


def _passed_filters(signal: dict[str, Any]) -> list[str]:
    labels = []
    for key in ("trend_ok", "score_ok", "rsi_ok", "atr_ok", "pullback_ok", "breakout_ok", "momentum_short_ok", "momentum_medium_ok"):
        if _as_bool(signal.get(key)):
            labels.append(key)
    return labels


def _failed_filters(signal: dict[str, Any]) -> list[str]:
    labels = []
    for key in ("trend_ok", "score_ok", "rsi_ok", "atr_ok", "pullback_ok", "breakout_ok", "momentum_short_ok", "momentum_medium_ok"):
        if not _as_bool(signal.get(key)):
            labels.append(key)
    return labels


def _evaluate_trend_pullback(signal: dict[str, Any]) -> dict[str, Any]:
    primary, secondary, risk = _base_blockers(signal)
    gap = _score_gap(signal)
    strict = not primary and not secondary and not risk and gap <= 0.0
    relaxed = not primary and not risk and gap <= 0.04 and len(secondary) <= 3
    recommendation = "ajustar trend_pullback_breakout" if relaxed and not strict else "manter estrategia"
    if primary:
        recommendation = "revisar timeframe" if "trend_not_confirmed" in primary else "sem dados suficientes"
    return _setup_payload("trend_pullback_breakout", signal, strict, relaxed, primary, secondary, risk, recommendation)


def _evaluate_breakout_confirmed(signal: dict[str, Any]) -> dict[str, Any]:
    primary, secondary, risk = _base_blockers(signal)
    breakout = _as_float(signal.get("breakout_20"))
    momentum = _as_float(signal.get("momentum"))
    momentum_8 = _as_float(signal.get("momentum_8"))
    rsi = _as_float(signal.get("rsi"), 50.0)
    gap = _score_gap(signal)
    local_primary = list(primary)
    local_secondary = list(secondary)
    if str(signal.get("trend_alignment") or "").lower() == "countertrend":
        local_primary.append("countertrend")
    if not (38.0 <= rsi <= 72.0):
        local_primary.append("rsi_not_breakout_friendly")
    if breakout < 0.0:
        local_secondary.append("breakout_confirmation_missing")
    if momentum < 0.0 or momentum_8 < -0.005:
        local_secondary.append("breakout_momentum_weak")
    strict = not local_primary and not risk and breakout >= 0.0 and momentum >= 0.0 and momentum_8 >= -0.005 and gap <= 0.0
    relaxed = not local_primary and not risk and breakout >= -0.01 and momentum >= -0.005 and momentum_8 >= -0.015 and gap <= 0.04
    recommendation = "criar setup complementar" if relaxed and not strict else "manter estrategia"
    return _setup_payload("breakout_confirmed", signal, strict, relaxed, local_primary, local_secondary, risk, recommendation)


def _evaluate_reversal_v_pattern(signal: dict[str, Any]) -> dict[str, Any]:
    primary, secondary, risk = _base_blockers(signal)
    rsi = _as_float(signal.get("rsi"), 50.0)
    momentum = _as_float(signal.get("momentum"))
    momentum_8 = _as_float(signal.get("momentum_8"))
    gap = _score_gap(signal)
    local_primary = [reason for reason in primary if reason != "reversal_not_eligible"]
    local_secondary = list(secondary)
    if not (34.0 <= rsi <= 52.0):
        local_primary.append("rsi_not_reversal_zone")
    elif not (38.0 <= rsi <= 48.0):
        local_secondary.append("rsi_marginal_reversal_zone")
    if momentum < -0.012 or momentum_8 < -0.025:
        local_secondary.append("reversal_momentum_not_confirmed")
    strict = not local_primary and not risk and 38.0 <= rsi <= 48.0 and momentum >= -0.005 and gap <= 0.0
    relaxed = not local_primary and not risk and 34.0 <= rsi <= 52.0 and momentum >= -0.012 and momentum_8 >= -0.025 and gap <= 0.06
    recommendation = "criar setup complementar" if relaxed and not strict else "sem dados suficientes"
    if risk:
        recommendation = "manter estrategia"
    return _setup_payload("reversal_v_pattern", signal, strict, relaxed, local_primary, local_secondary, risk, recommendation)


def _setup_payload(
    setup_name: str,
    signal: dict[str, Any],
    strict: bool,
    relaxed: bool,
    primary: list[str],
    secondary: list[str],
    risk: list[str],
    recommendation: str,
) -> dict[str, Any]:
    gap = _score_gap(signal)
    return {
        "setup_name": setup_name,
        "symbol": str(signal.get("asset") or ""),
        "would_pass_strict": bool(strict),
        "would_pass_relaxed_shadow": bool(relaxed),
        "score": _as_float(signal.get("score")),
        "min_score": _as_float(signal.get("effective_min_signal_score"), _as_float(signal.get("base_min_signal_score"), 0.0)),
        "score_gap": gap,
        "failed_filters": _failed_filters(signal),
        "passed_filters": _passed_filters(signal),
        "primary_blockers": sorted(set(primary)),
        "secondary_blockers": sorted(set(secondary)),
        "risk_blockers": sorted(set(risk)),
        "context_status": str(signal.get("context_status") or "NEUTRO"),
        "feed_status": "LIVE" if str(signal.get("data_source") or "").lower() == "market" else "FALLBACK",
        "recommendation": recommendation,
    }


def evaluate_signal_setups(signal: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _evaluate_trend_pullback(signal),
        _evaluate_breakout_confirmed(signal),
        _evaluate_reversal_v_pattern(signal),
    ]


def default_strategy_structure_audit_state(reason: str = "No structural audit data yet.") -> dict[str, Any]:
    return {
        "structural_audit_enabled": True,
        "structural_audit_mode": MODE,
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
        "structural_audit_reason": reason,
    }


def build_strategy_structure_audit(signals: list[dict[str, Any]] | None, *, enabled: bool = True) -> dict[str, Any]:
    if not enabled:
        return default_strategy_structure_audit_state("Structural audit disabled.")

    rows: list[dict[str, Any]] = []
    for signal in list(signals or []):
        rows.extend(evaluate_signal_setups(dict(signal or {})))

    if not rows:
        state = default_strategy_structure_audit_state("No signal sample available for structural audit yet.")
        state["structural_audit_last_run_at"] = _utc_now_iso()
        return state

    comparison = _build_setup_comparison(rows)
    comparison_maps = _comparison_maps(comparison)
    relaxed_candidates = [row for row in rows if row.get("would_pass_relaxed_shadow")]
    top_candidates = sorted(relaxed_candidates or rows, key=lambda row: (float(row.get("score") or 0.0), -float(row.get("score_gap") or 0.0)), reverse=True)
    top = top_candidates[0] if top_candidates else {}
    primary = _top_blocker(top.get("primary_blockers", []))
    secondary = _top_blocker(top.get("secondary_blockers", []))
    recommendation = _overall_recommendation(comparison, top)
    state = {
        "structural_audit_enabled": True,
        "structural_audit_mode": MODE,
        "structural_audit_last_run_at": _utc_now_iso(),
        "structural_audit_candidates": len(relaxed_candidates),
        "structural_audit_top_setup": str(top.get("setup_name") or ""),
        "structural_audit_top_symbol": str(top.get("symbol") or ""),
        "structural_audit_top_score": top.get("score"),
        "structural_audit_top_gap": top.get("score_gap"),
        "structural_audit_primary_blocker": primary,
        "structural_audit_secondary_blocker": secondary,
        "structural_audit_recommendation": recommendation,
        "structural_audit_should_adjust_strategy": recommendation in {"ajustar trend_pullback_breakout", "criar setup complementar", "revisar RSI/momentum", "revisar timeframe"},
        "structural_audit_setup_comparison": comparison,
        **comparison_maps,
        **_audit_notes(comparison),
        "structural_audit_recent_candidates": top_candidates[:MAX_RECENT_CANDIDATES],
        "structural_audit_reason": _diagnostic_reason(comparison, top),
    }
    return state


def _build_setup_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_setup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_setup[str(row.get("setup_name") or "unknown")].append(row)

    comparison: list[dict[str, Any]] = []
    for setup in SETUPS:
        items = by_setup.get(setup, [])
        relaxed = [item for item in items if item.get("would_pass_relaxed_shadow")]
        scores = [_as_float(item.get("score")) for item in items]
        gaps = [_as_float(item.get("score_gap")) for item in items]
        best = max(items, key=lambda item: _as_float(item.get("score")), default={})
        primary = Counter(reason for item in items for reason in list(item.get("primary_blockers", []) or []))
        secondary = Counter(reason for item in items for reason in list(item.get("secondary_blockers", []) or []))
        risk = Counter(reason for item in items for reason in list(item.get("risk_blockers", []) or []))
        dominant = _dominant_blocker(primary, secondary, risk)
        comparison.append(
            {
                "setup": setup,
                "total_samples": len(items),
                "shadow_candidates": len(relaxed),
                "best_symbol": str(best.get("symbol") or ""),
                "best_score": best.get("score"),
                "best_gap": best.get("score_gap"),
                "average_score": round(sum(scores) / len(scores), 4) if scores else None,
                "average_gap": round(sum(gaps) / len(gaps), 4) if gaps else None,
                "dominant_blocker": dominant,
                "primary_blockers": dict(primary.most_common(5)),
                "secondary_blockers": dict(secondary.most_common(5)),
                "risk_blockers": dict(risk.most_common(5)),
                "recommendation": _setup_recommendation(setup, len(relaxed), dominant),
            }
        )
    return comparison


def _comparison_maps(comparison: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    total_candidates_by_setup: dict[str, Any] = {}
    near_candidates_by_setup: dict[str, Any] = {}
    primary_blockers_by_setup: dict[str, Any] = {}
    secondary_blockers_by_setup: dict[str, Any] = {}
    average_score_by_setup: dict[str, Any] = {}
    average_gap_by_setup: dict[str, Any] = {}
    best_candidate_by_setup: dict[str, Any] = {}

    for row in comparison:
        setup = str(row.get("setup") or "unknown")
        total_candidates_by_setup[setup] = int(row.get("total_samples", 0) or 0)
        near_candidates_by_setup[setup] = int(row.get("shadow_candidates", 0) or 0)
        primary_blockers_by_setup[setup] = dict(row.get("primary_blockers", {}) or {})
        secondary_blockers_by_setup[setup] = dict(row.get("secondary_blockers", {}) or {})
        average_score_by_setup[setup] = row.get("average_score")
        average_gap_by_setup[setup] = row.get("average_gap")
        best_candidate_by_setup[setup] = {
            "symbol": str(row.get("best_symbol") or ""),
            "score": row.get("best_score"),
            "gap": row.get("best_gap"),
        }

    return {
        "structural_audit_total_candidates_by_setup": total_candidates_by_setup,
        "structural_audit_near_candidates_by_setup": near_candidates_by_setup,
        "structural_audit_primary_blockers_by_setup": primary_blockers_by_setup,
        "structural_audit_secondary_blockers_by_setup": secondary_blockers_by_setup,
        "structural_audit_average_score_by_setup": average_score_by_setup,
        "structural_audit_average_gap_by_setup": average_gap_by_setup,
        "structural_audit_best_candidate_by_setup": best_candidate_by_setup,
    }


def _audit_notes(comparison: list[dict[str, Any]]) -> dict[str, str]:
    dominant_blockers = [str(row.get("dominant_blocker") or "") for row in comparison]
    blocker_blob = " ".join(dominant_blockers).lower()
    reversal_row = next((row for row in comparison if str(row.get("setup")) == "reversal_v_pattern"), {})

    if "trend_not_confirmed" in blocker_blob or "countertrend" in blocker_blob:
        timeframe_note = (
            "Possivel gargalo de temporalidade/tendencia: o diario pode estar atrasado ou divergente. "
            "Diagnostico inconclusivo sem multi-timeframe ativo."
        )
    else:
        timeframe_note = "Sem evidencia forte de gargalo exclusivo de temporalidade neste ciclo."

    if any(token in blocker_blob for token in ("rsi", "momentum", "confidence_too_low", "reversal_not_eligible")):
        rsi_momentum_note = (
            "RSI/momentum aparecem entre os bloqueadores estruturais; observar se sao marginais ou primarios "
            "antes de qualquer ajuste."
        )
    else:
        rsi_momentum_note = "Sem evidencia forte de que RSI/momentum expliquem sozinhos a rejeicao estrutural."

    if int(reversal_row.get("shadow_candidates", 0) or 0) > 0:
        reversal_note = (
            "Reversal v-pattern apareceu como candidato shadow. Pode merecer estudo complementar, "
            "mas segue sem autoridade de trade."
        )
    else:
        reversal_note = "Reversal v-pattern permanece bloqueado/inconclusivo; nao foi ativado como gatilho real."

    return {
        "structural_audit_timeframe_note": timeframe_note,
        "structural_audit_rsi_momentum_note": rsi_momentum_note,
        "structural_audit_reversal_note": reversal_note,
    }


def _dominant_blocker(primary: Counter, secondary: Counter, risk: Counter) -> str:
    combined = Counter()
    combined.update(primary)
    combined.update(secondary)
    combined.update(risk)
    if not combined:
        return "none"
    return str(combined.most_common(1)[0][0])


def _setup_recommendation(setup: str, relaxed_count: int, dominant: str) -> str:
    if relaxed_count <= 0:
        if dominant in {"trend_not_confirmed", "countertrend"}:
            return "revisar timeframe"
        if dominant in {"reversal_not_eligible", "rsi_not_reversal_zone"}:
            return "revisar RSI/momentum"
        return "sem dados suficientes"
    if setup == "trend_pullback_breakout":
        return "ajustar trend_pullback_breakout"
    return "criar setup complementar"


def _overall_recommendation(comparison: list[dict[str, Any]], top: dict[str, Any]) -> str:
    if not top:
        return "sem dados suficientes"
    top_setup = str(top.get("setup_name") or "")
    if not bool(top.get("would_pass_relaxed_shadow", False)):
        dominant = str(top.get("primary_blockers", [""])[0] if top.get("primary_blockers") else "")
        if dominant == "trend_not_confirmed":
            return "revisar timeframe"
        return "sem dados suficientes"
    if top_setup == "trend_pullback_breakout":
        return "ajustar trend_pullback_breakout"
    if top_setup == "reversal_v_pattern":
        return "criar setup complementar"
    return "criar setup complementar"


def _top_blocker(values: list[str] | None) -> str:
    if not values:
        return ""
    return str(Counter(values).most_common(1)[0][0])


def _diagnostic_reason(comparison: list[dict[str, Any]], top: dict[str, Any]) -> str:
    if not comparison:
        return "Sem dados suficientes para auditar a estrutura."
    if top and bool(top.get("would_pass_relaxed_shadow", False)):
        return (
            f"Melhor candidato shadow: {top.get('setup_name')} em {top.get('symbol')} "
            f"com score {top.get('score')} e gap {top.get('score_gap')}. Shadow only; nenhuma decisao real alterada."
        )
    dominant = str((comparison[0] or {}).get("dominant_blocker") or "unknown")
    return f"Nenhum candidato shadow forte; bloqueador dominante observado: {dominant}."
