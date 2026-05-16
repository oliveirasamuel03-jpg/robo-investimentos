"""Study-only controlled micro-adjustment evaluation.

This module ranks possible future micro-adjustment studies after the 10-day
PAPER validation cycle. It never applies adjustments, changes scores or
thresholds, alters broker/provider behavior, creates orders, changes wallet,
positions, PnL, or history.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


MODE = "STUDY_ONLY"
DIAGNOSTIC_MODE = "DIAGNOSTIC_ONLY"
SAFETY_MODE = "SHADOW_ONLY"
SOURCE_PHASE = "FASE 2.6A"
NEXT_PHASE = "FASE_2_6C_ONLY_IF_CONDITIONS_PASS"
VALIDATION_WINDOW = "Novo ciclo PAPER controlado apos autorizacao da FASE 2.6C"

SAFE_RECOMMENDATIONS = {
    "observe_more_before_adjustment",
    "prepare_single_micro_adjustment_phase",
    "study_secondary_confirmation_only",
    "study_breakout_confirmation_only",
    "study_real_rule_mapping_only",
    "study_h1_after_h4_bos_only",
    "study_pullback_quality_only",
    "no_threshold_change_recommended",
    "no_profile_change_recommended",
    "no_real_money_recommended",
    "insufficient_data",
}

BLOCKED_ACTIONS = [
    "start_real_money",
    "lower_global_min_signal_score_now",
    "change_profile_to_aggressive_now",
    "apply_micro_adjustment_now",
    "bypass_guards",
    "remove_no_setup_eligible",
    "remove_reversal_not_eligible",
    "convert_shadow_to_real_signal",
    "use_bos_as_direct_trigger",
    "use_fibonacci_as_direct_trigger",
    "use_h4_bos_as_direct_trigger",
    "use_h1_confirmation_as_direct_trigger",
    "increase_risk_now",
    "increase_ticket_now",
    "increase_max_open_positions_now",
]

REQUIRED_CONDITIONS_FOR_2_6C = [
    "feed_live_or_acceptable_delayed_without_current_fallback_dominance",
    "context_not_critical",
    "preferably_context_favorable_or_neutral",
    "worker_online_consecutive_failures_zero",
    "paper_trading_preserved",
    "no_real_orders_sent",
    "safe_near_approved_or_repeated_structural_diagnostic",
    "candidate_adjustment_low_or_controlled_medium_risk",
    "rollback_defined",
    "logs_state_ui_coherent",
    "no_critical_guard_active",
]

BASE_GUARDS = [
    "PAPER_ONLY",
    "broker_paper",
    "no_real_orders",
    "no_official_paper_order_mutation",
    "no_global_threshold_change",
    "daily_loss_guard_preserved",
    "position_limit_guard_preserved",
    "feed_guard_preserved",
    "rollback_required",
]

UNSAFE_ADJUSTMENTS = [
    {
        "id": "lower_global_min_signal_score_now",
        "reason": "Global threshold change is not authorized by this study.",
        "blocked": True,
    },
    {
        "id": "start_real_money",
        "reason": "Real money is not authorized after an APROVADO COM RESSALVAS cycle.",
        "blocked": True,
    },
    {
        "id": "change_profile_to_aggressive_now",
        "reason": "Profile changes would alter risk and are blocked.",
        "blocked": True,
    },
    {
        "id": "convert_shadow_to_real_signal",
        "reason": "Shadow diagnostics cannot become operational authority.",
        "blocked": True,
    },
    {
        "id": "use_bos_or_fibonacci_as_direct_trigger",
        "reason": "BOS, H1/H4, Pivot, Fibonacci and Multi-TF remain diagnostic only.",
        "blocked": True,
    },
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _upper(value: Any, default: str = "") -> str:
    return _text(value, default).upper()


def _safe_recommendation(value: Any, default: str = "observe_more_before_adjustment") -> str:
    recommendation = _text(value, default)
    return recommendation if recommendation in SAFE_RECOMMENDATIONS else default


def _source_dict(root: dict[str, Any], report: dict[str, Any], cycle: dict[str, Any], key: str) -> dict[str, Any]:
    return _as_dict(report.get(key) or cycle.get(key) or root.get(key))


def _market_context_status(root: dict[str, Any], report: dict[str, Any]) -> str:
    context = _as_dict(report.get("market_context") or root.get("market_context"))
    value = (
        context.get("market_context_status")
        or context.get("context_status")
        or report.get("context_status")
        or root.get("context_status")
    )
    return _upper(value, "UNKNOWN")


def _context_blocks_adjustment(status: str) -> bool:
    return status in {"DESFAVORAVEL", "CRITICO", "CRITICAL", "UNFAVORABLE"}


def _feed_status(root: dict[str, Any], report: dict[str, Any], feed_scope: dict[str, Any]) -> tuple[str, bool, str]:
    market = _as_dict(root.get("market_data"))
    feed = _upper(market.get("feed_status") or market.get("status") or report.get("feed_status"), "UNKNOWN")
    current_clean = bool(feed_scope.get("current_feed_is_clean", False))
    if "current_fallback_count" in feed_scope:
        current_fallback = _as_int(feed_scope.get("current_fallback_count"), 0)
    else:
        current_fallback = _as_int(_as_dict(market.get("source_breakdown")).get("fallback"), 0)
    if feed in {"LIVE", "DELAYED"} and current_fallback == 0:
        current_clean = True
    fallback_scope = _text(feed_scope.get("fallback_blocker_scope"), "UNKNOWN")
    return feed, current_clean, fallback_scope


def _preview_values(calibration_preview: dict[str, Any], shadow_simulator: dict[str, Any]) -> tuple[int, float | None, float | None, float | None]:
    near_count = _as_int(
        calibration_preview.get("near_approved_count")
        if calibration_preview.get("near_approved_count") is not None
        else shadow_simulator.get("preview_near_approved_count"),
        0,
    )
    best_seen = _as_float(
        calibration_preview.get("best_score_seen")
        or calibration_preview.get("best_score")
        or shadow_simulator.get("shadow_best_candidate_score"),
        None,
    )
    min_score = _as_float(
        calibration_preview.get("min_score_current")
        or calibration_preview.get("min_score")
        or shadow_simulator.get("min_score"),
        None,
    )
    preview_floor = _as_float(calibration_preview.get("preview_score_floor"), None)
    return near_count, best_seen, min_score, preview_floor


def _dominant_bottleneck(plan: dict[str, Any], bottleneck: dict[str, Any], report: dict[str, Any], taxonomy: dict[str, Any]) -> str:
    metrics = _as_dict(report.get("metrics"))
    return _text(
        plan.get("dominant_bottleneck")
        or bottleneck.get("dominant_bottleneck")
        or metrics.get("rejection_top_reason")
        or taxonomy.get("normalized_primary_reason"),
        "UNKNOWN",
    )


def _dominant_setup(plan: dict[str, Any], bottleneck: dict[str, Any], report: dict[str, Any], taxonomy: dict[str, Any]) -> str:
    metrics = _as_dict(report.get("metrics"))
    return _text(
        plan.get("dominant_setup")
        or bottleneck.get("dominant_setup")
        or metrics.get("rejection_top_strategy")
        or taxonomy.get("top_setup"),
        "trend_pullback_breakout",
    )


def _has_structure_confirmed_no_setup(no_setup: dict[str, Any], bridge: dict[str, Any]) -> bool:
    bucket = _upper(no_setup.get("top_reason_bucket"))
    blocker = _upper(no_setup.get("top_real_blocker") or bridge.get("top_real_blocker"))
    bridge_status = _upper(bridge.get("top_bridge_status") or bridge.get("top_structure_status"))
    return (
        "STRUCTURE_CONFIRMED" in bucket
        or _as_int(no_setup.get("structure_confirmed_but_no_setup_count"), 0) > 0
        or (blocker == "NO_SETUP_ELIGIBLE" and "STRUCTURE_CONFIRMED" in bridge_status)
    )


def _has_h4_confirmed_h1_pending(h1_after_h4: dict[str, Any], bos_quality: dict[str, Any]) -> bool:
    h4_bos = _upper(h1_after_h4.get("h4_bos_state") or bos_quality.get("h4_bos_state"))
    h1_bos = _upper(h1_after_h4.get("h1_bos_state") or bos_quality.get("h1_bos_state"))
    status = _upper(h1_after_h4.get("h1_confirmation_status"))
    h4_confirmed = h4_bos in {"BOS_RETEST_CONFIRMED", "BOS_CONFIRMED_STRONG", "BOS_BY_CLOSE_CONFIRMED"}
    h1_pending = h1_bos in {"INSUFFICIENT_DATA", "NO_BOS", "BOS_RETEST_PENDING"} or (
        status and status != "H1_CONFIRMED_AFTER_H4_BOS"
    )
    return (h4_confirmed and h1_pending) or (
        _as_int(h1_after_h4.get("h4_bos_confirmed_count"), 0) > 0
        and _as_int(h1_after_h4.get("h1_missing_confirmation_count"), 0) > 0
    )


def _has_secondary_confirmation_issue(bottleneck: dict[str, Any], dominant: str, near_count: int) -> bool:
    return (
        "SECONDARY_CONFIRMATION_WEAK" in _upper(dominant)
        or _as_int(bottleneck.get("secondary_confirmation_weak_count"), 0) > 0
        or ("SCORE_BELOW_MIN" in _upper(dominant) and near_count > 0)
    )


def _has_breakout_issue(bottleneck: dict[str, Any], taxonomy: dict[str, Any], no_setup: dict[str, Any]) -> bool:
    values = " ".join(
        [
            _upper(bottleneck.get("dominant_bottleneck")),
            _upper(taxonomy.get("normalized_primary_reason")),
            _upper(taxonomy.get("taxonomy_status")),
            _upper(no_setup.get("top_secondary_blocker")),
            _upper(no_setup.get("top_reason_bucket")),
        ]
    )
    return "BREAKOUT" in values or "BOS_MISSING" in values


def _has_pullback_issue(taxonomy: dict[str, Any], no_setup: dict[str, Any]) -> bool:
    values = " ".join(
        [
            _upper(taxonomy.get("taxonomy_status")),
            _upper(taxonomy.get("normalized_primary_reason")),
            _upper(no_setup.get("top_reason_bucket")),
            _upper(no_setup.get("recommendation")),
        ]
    )
    return "PULLBACK" in values or "REACTION" in values or "FIB_STRUCTURE" in values


def _clean_cycle_needed(plan: dict[str, Any], report: dict[str, Any]) -> bool:
    consistency = _as_dict(report.get("consistency"))
    proposals = [item for item in _as_list(plan.get("proposed_micro_adjustments")) if isinstance(item, dict)]
    return (
        consistency.get("capital_phase_aligned") is False
        or any(_text(item.get("id")) == "clean_cycle_baseline_alignment_study" for item in proposals)
    )


def _candidate(
    *,
    item_id: str,
    title: str,
    target: str,
    hypothesis: str,
    evidence: str,
    expected_effect: str,
    risk_level: str,
    reason_to_block_now: str,
    suggested_next_phase: str = NEXT_PHASE,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "title": title,
        "target": target,
        "hypothesis": hypothesis,
        "evidence": evidence,
        "expected_effect": expected_effect,
        "risk_level": risk_level,
        "allowed_now": False,
        "requires_next_phase": True,
        "can_change_threshold": False,
        "can_change_profile": False,
        "can_affect_real_trade": False,
        "validation_window": VALIDATION_WINDOW,
        "required_guards": list(BASE_GUARDS),
        "rollback_condition": "Abortar se aumentar entradas fracas, drawdown, inconsistencias de estado, ruido de sinais ou confusao UI/log.",
        "reason_to_block_now": reason_to_block_now,
        "suggested_next_phase": suggested_next_phase,
    }


def _build_candidates(
    *,
    context_blocks: bool,
    dominant: str,
    near_count: int,
    structure_confirmed_no_setup: bool,
    h4_h1_pending: bool,
    secondary_issue: bool,
    breakout_issue: bool,
    pullback_issue: bool,
    clean_cycle_needed: bool,
) -> list[dict[str, Any]]:
    context_block_reason = (
        "context_not_safe_for_adjustment_now" if context_blocks else "study_only_requires_explicit_future_phase"
    )
    candidates = [
        _candidate(
            item_id="secondary_confirmation_micro_adjustment_study",
            title="Estudar confirmacao secundaria marginal sem reduzir score global",
            target="secondary_confirmation",
            hypothesis="A confirmacao secundaria pode estar rigida demais em sinais quase aprovados.",
            evidence=dominant if secondary_issue else "secondary_confirmation_candidate_for_monitoring",
            expected_effect="Medir reducao teorica de rejeicoes por ruido sem mudar score real.",
            risk_level="LOW" if secondary_issue and near_count > 0 else "MEDIUM",
            reason_to_block_now=(
                "no_safe_near_approved_sample" if near_count <= 0 else context_block_reason
            ),
        ),
        _candidate(
            item_id="breakout_confirmation_quality_study",
            title="Estudar qualidade de breakout confirmation",
            target="breakout_confirmation",
            hypothesis="Breakout/BOS pode estar ausente, fraco ou dependente de fechamento estrutural.",
            evidence="breakout_or_bos_missing" if breakout_issue else "breakout_quality_monitoring",
            expected_effect="Separar rejeicao correta de rigidez excessiva sem transformar breakout em gatilho.",
            risk_level="LOW" if breakout_issue else "MEDIUM",
            reason_to_block_now=context_block_reason,
        ),
        _candidate(
            item_id="real_rule_mapping_study",
            title="Estudar mapeamento da regra real para estrutura confirmada",
            target="real_rule_mapping",
            hypothesis="Estrutura shadow confirmada pode nao estar mapeada para elegibilidade real do setup.",
            evidence="structure_confirmed_but_no_setup" if structure_confirmed_no_setup else "rule_mapping_monitoring",
            expected_effect="Explicar lacuna entre diagnostico estrutural e NO_SETUP_ELIGIBLE sem liberar trade.",
            risk_level="MEDIUM" if structure_confirmed_no_setup else "LOW",
            reason_to_block_now=context_block_reason,
        ),
        _candidate(
            item_id="h1_after_h4_bos_mapping_study",
            title="Estudar confirmacao 1H apos BOS/reteste 4H",
            target="h1_after_h4_bos_mapping",
            hypothesis="H4 pode confirmar antes do 1H, exigindo leitura de timing sem virar gatilho.",
            evidence="h4_confirmed_h1_pending" if h4_h1_pending else "h1_h4_mapping_monitoring",
            expected_effect="Medir qualidade de confirmacao 1H apos estrutura 4H sem operar por H4.",
            risk_level="LOW" if h4_h1_pending else "MEDIUM",
            reason_to_block_now=context_block_reason,
        ),
        _candidate(
            item_id="pullback_quality_study",
            title="Estudar qualidade do pullback/reacao",
            target="pullback_quality",
            hypothesis="Estrutura/Fibonacci pode estar boa, mas sem pullback ou reacao objetiva.",
            evidence="pullback_or_reaction_missing" if pullback_issue else "pullback_quality_monitoring",
            expected_effect="Separar zona estrutural de reacao operacional sem usar Fibonacci como gatilho.",
            risk_level="LOW" if pullback_issue else "MEDIUM",
            reason_to_block_now=context_block_reason,
        ),
        _candidate(
            item_id="clean_cycle_reset_study",
            title="Estudar reset de ciclo limpo com capital-base alinhado",
            target="next_paper_validation_baseline",
            hypothesis="Um novo ciclo limpo melhora comparabilidade sem alterar capital automaticamente.",
            evidence="clean_cycle_needed" if clean_cycle_needed else "baseline_monitoring",
            expected_effect="Melhorar rastreabilidade da proxima validacao PAPER.",
            risk_level="LOW",
            reason_to_block_now="manual_cycle_planning_required",
        ),
    ]
    return candidates


def _select_candidate(
    *,
    candidates: list[dict[str, Any]],
    structure_confirmed_no_setup: bool,
    h4_h1_pending: bool,
    secondary_issue: bool,
    near_count: int,
    breakout_issue: bool,
    pullback_issue: bool,
    clean_cycle_needed: bool,
) -> dict[str, Any] | None:
    preference = []
    if structure_confirmed_no_setup:
        preference.append("real_rule_mapping_study")
    if h4_h1_pending:
        preference.append("h1_after_h4_bos_mapping_study")
    if secondary_issue and near_count > 0:
        preference.append("secondary_confirmation_micro_adjustment_study")
    if breakout_issue:
        preference.append("breakout_confirmation_quality_study")
    if pullback_issue:
        preference.append("pullback_quality_study")
    if clean_cycle_needed:
        preference.append("clean_cycle_reset_study")
    preference.extend(
        [
            "real_rule_mapping_study",
            "h1_after_h4_bos_mapping_study",
            "secondary_confirmation_micro_adjustment_study",
            "breakout_confirmation_quality_study",
            "pullback_quality_study",
            "clean_cycle_reset_study",
        ]
    )
    by_id = {item["id"]: item for item in candidates}
    for item_id in preference:
        candidate = by_id.get(item_id)
        if candidate:
            return candidate
    return None


def _recommendation_for(selected_id: str, context_blocks: bool, insufficient: bool, near_count: int) -> str:
    if insufficient:
        return "insufficient_data"
    if context_blocks:
        return "observe_more_before_adjustment"
    mapping = {
        "secondary_confirmation_micro_adjustment_study": "study_secondary_confirmation_only",
        "breakout_confirmation_quality_study": "study_breakout_confirmation_only",
        "real_rule_mapping_study": "study_real_rule_mapping_only",
        "h1_after_h4_bos_mapping_study": "study_h1_after_h4_bos_only",
        "pullback_quality_study": "study_pullback_quality_only",
    }
    if near_count <= 0 and selected_id == "secondary_confirmation_micro_adjustment_study":
        return "no_threshold_change_recommended"
    return mapping.get(selected_id, "prepare_single_micro_adjustment_phase")


def _study_status(
    *,
    insufficient: bool,
    context_blocks: bool,
    near_count: int,
    selected: dict[str, Any] | None,
    dominant: str,
) -> str:
    if insufficient:
        return "INSUFFICIENT_DATA_FOR_MICRO_ADJUSTMENT"
    if context_blocks:
        return "CONTEXT_NOT_SAFE_FOR_ADJUSTMENT"
    if near_count <= 0 and "SCORE_BELOW_MIN" in _upper(dominant):
        return "STRATEGY_TOO_SELECTIVE_BUT_NOT_CALIBRATABLE_YET"
    if near_count <= 0:
        return "NO_NEAR_APPROVED_SAMPLE"
    if selected:
        return "MICRO_ADJUSTMENT_CANDIDATE_IDENTIFIED"
    return "STUDY_READY_BUT_NO_ACTION"


def default_controlled_micro_adjustment_study_state(
    reason: str = "No controlled micro-adjustment study data yet.",
) -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": MODE,
        "diagnostic_mode": DIAGNOSTIC_MODE,
        "safety_mode": SAFETY_MODE,
        "generated_at": "",
        "source_phase": SOURCE_PHASE,
        "study_status": "INSUFFICIENT_DATA_FOR_MICRO_ADJUSTMENT",
        "market_context_status": "UNKNOWN",
        "context_allows_adjustment_now": False,
        "feed_status": "UNKNOWN",
        "current_feed_is_clean": False,
        "fallback_blocker_scope": "UNKNOWN",
        "dominant_bottleneck": "UNKNOWN",
        "dominant_setup": "trend_pullback_breakout",
        "near_approved_count": 0,
        "best_seen_score": None,
        "current_min_score": None,
        "preview_floor": None,
        "selected_candidate_adjustment": "",
        "selected_candidate_reason": "insufficient_data",
        "selected_candidate_risk_level": "BLOCKED",
        "selected_candidate_allowed_now": False,
        "selected_candidate_requires_next_phase": True,
        "theoretical_impact_summary": "Insufficient data for theoretical impact study.",
        "theoretical_impact_notes": [],
        "candidate_adjustments": [],
        "unsafe_adjustments": list(UNSAFE_ADJUSTMENTS),
        "blocked_actions": list(BLOCKED_ACTIONS),
        "required_conditions_for_2_6c": list(REQUIRED_CONDITIONS_FOR_2_6C),
        "recommendation": "insufficient_data",
        "recommended_next_phase": "",
        "should_continue_paper": True,
        "should_start_real_money": False,
        "should_change_threshold_now": False,
        "should_change_profile_now": False,
        "should_apply_micro_adjustment_now": False,
        "study_only": True,
        "diagnostic_only": True,
        "shadow_only": True,
        "notes": reason,
    }


def build_controlled_micro_adjustment_study(
    *,
    state: Mapping[str, Any] | None = None,
    validation_report: Mapping[str, Any] | None = None,
    cycle_validation: Mapping[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Build a study-only ranking of candidate micro-adjustments."""
    if not enabled:
        result = default_controlled_micro_adjustment_study_state("Controlled micro-adjustment study disabled.")
        result.update({"enabled": False})
        return result

    root = _as_dict(state)
    report = _as_dict(validation_report) or _as_dict(_as_dict(root.get("validation")).get("last_report"))
    cycle = _as_dict(cycle_validation)
    if not root and not report and not cycle:
        return default_controlled_micro_adjustment_study_state()

    plan = _source_dict(root, report, cycle, "post_10d_calibration_plan")
    calibration_preview = _source_dict(root, report, cycle, "calibration_preview")
    strategy_bottleneck = _source_dict(root, report, cycle, "strategy_bottleneck")
    shadow_simulator = _source_dict(root, report, cycle, "shadow_decision_simulator")
    feed_scope = _source_dict(root, report, cycle, "feed_scope_reconciliation")
    taxonomy = _source_dict(root, report, cycle, "setup_blocker_taxonomy_audit")
    bos_quality = _source_dict(root, report, cycle, "bos_confirmation_quality_audit")
    h1_after_h4 = _source_dict(root, report, cycle, "h1_confirmation_after_h4_bos_audit")
    no_setup = _source_dict(root, report, cycle, "no_setup_eligible_decomposition")
    bridge = _source_dict(root, report, cycle, "strategy_decision_bridge_trace")

    context_status = _market_context_status(root, report)
    context_blocks = _context_blocks_adjustment(context_status)
    feed_status, current_feed_is_clean, fallback_scope = _feed_status(root, report, feed_scope)
    near_count, best_seen, current_min, preview_floor = _preview_values(calibration_preview, shadow_simulator)
    dominant = _dominant_bottleneck(plan, strategy_bottleneck, report, taxonomy)
    setup = _dominant_setup(plan, strategy_bottleneck, report, taxonomy)

    structure_confirmed_no_setup = _has_structure_confirmed_no_setup(no_setup, bridge)
    h4_h1_pending = _has_h4_confirmed_h1_pending(h1_after_h4, bos_quality)
    secondary_issue = _has_secondary_confirmation_issue(strategy_bottleneck, dominant, near_count)
    breakout_issue = _has_breakout_issue(strategy_bottleneck, taxonomy, no_setup)
    pullback_issue = _has_pullback_issue(taxonomy, no_setup)
    clean_cycle = _clean_cycle_needed(plan, report)

    candidates = _build_candidates(
        context_blocks=context_blocks,
        dominant=dominant,
        near_count=near_count,
        structure_confirmed_no_setup=structure_confirmed_no_setup,
        h4_h1_pending=h4_h1_pending,
        secondary_issue=secondary_issue,
        breakout_issue=breakout_issue,
        pullback_issue=pullback_issue,
        clean_cycle_needed=clean_cycle,
    )
    selected = _select_candidate(
        candidates=candidates,
        structure_confirmed_no_setup=structure_confirmed_no_setup,
        h4_h1_pending=h4_h1_pending,
        secondary_issue=secondary_issue,
        near_count=near_count,
        breakout_issue=breakout_issue,
        pullback_issue=pullback_issue,
        clean_cycle_needed=clean_cycle,
    )
    insufficient = not bool(plan or calibration_preview or strategy_bottleneck or taxonomy or no_setup or bridge)
    selected_id = _text(selected.get("id") if selected else "")
    recommendation = _safe_recommendation(
        _recommendation_for(selected_id, context_blocks, insufficient, near_count),
        "observe_more_before_adjustment",
    )
    status = _study_status(
        insufficient=insufficient,
        context_blocks=context_blocks,
        near_count=near_count,
        selected=selected,
        dominant=dominant,
    )
    if status == "MICRO_ADJUSTMENT_CANDIDATE_IDENTIFIED" and selected:
        status = "MICRO_ADJUSTMENT_REQUIRES_NEXT_PHASE"

    score_gap = None
    if best_seen is not None and current_min is not None:
        score_gap = round(float(current_min) - float(best_seen), 6)
    theoretical_notes = [
        {
            "note": "No operational behavior changes in FASE 2.6B.",
            "detail": "All candidate adjustments remain study-only with allowed_now=false.",
        },
        {
            "note": "Threshold change remains blocked.",
            "detail": f"near_approved_count={near_count}; best_seen_score={best_seen}; current_min_score={current_min}; score_gap={score_gap}",
        },
        {
            "note": "Context gate remains authoritative for study selection.",
            "detail": (
                f"context={context_status}; context_gate_would_not_block={not context_blocks}; "
                "context_allows_adjustment_now=false"
            ),
        },
    ]
    if selected:
        theoretical_notes.append(
            {
                "note": "Selected candidate is only a future study.",
                "detail": f"{selected_id}; risk={selected.get('risk_level')}; requires_next_phase=true",
            }
        )

    return {
        "enabled": True,
        "mode": MODE,
        "diagnostic_mode": DIAGNOSTIC_MODE,
        "safety_mode": SAFETY_MODE,
        "generated_at": _utc_now_iso(),
        "source_phase": SOURCE_PHASE,
        "study_status": status,
        "market_context_status": context_status,
        "context_allows_adjustment_now": False,
        "feed_status": feed_status,
        "current_feed_is_clean": bool(current_feed_is_clean),
        "fallback_blocker_scope": fallback_scope,
        "dominant_bottleneck": dominant,
        "dominant_setup": setup,
        "near_approved_count": near_count,
        "best_seen_score": best_seen,
        "current_min_score": current_min,
        "preview_floor": preview_floor,
        "selected_candidate_adjustment": selected_id,
        "selected_candidate_reason": _text(selected.get("hypothesis") if selected else "", "insufficient_data"),
        "selected_candidate_risk_level": _text(selected.get("risk_level") if selected else "", "BLOCKED"),
        "selected_candidate_allowed_now": False,
        "selected_candidate_requires_next_phase": True,
        "theoretical_impact_summary": (
            "Study-only theoretical impact: no score, threshold, profile, broker, order, PnL, history, wallet, or position changed."
        ),
        "theoretical_impact_notes": theoretical_notes,
        "candidate_adjustments": candidates,
        "unsafe_adjustments": list(UNSAFE_ADJUSTMENTS),
        "blocked_actions": list(BLOCKED_ACTIONS),
        "required_conditions_for_2_6c": list(REQUIRED_CONDITIONS_FOR_2_6C),
        "recommendation": recommendation,
        "recommended_next_phase": "" if recommendation == "insufficient_data" else NEXT_PHASE,
        "should_continue_paper": True,
        "should_start_real_money": False,
        "should_change_threshold_now": False,
        "should_change_profile_now": False,
        "should_apply_micro_adjustment_now": False,
        "study_only": True,
        "diagnostic_only": True,
        "shadow_only": True,
        "notes": "Study-only controlled micro-adjustment ranking. No trade decision changed.",
    }


__all__ = [
    "build_controlled_micro_adjustment_study",
    "default_controlled_micro_adjustment_study_state",
]
