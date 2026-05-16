"""Planning-only post-10D calibration plan.

This module consolidates the 10-day PAPER validation outcome and proposes
future study items. It never changes strategy decisions, scores, thresholds,
broker behavior, provider budget, orders, positions, wallet, PnL, or history.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


MODE = "PLANNING_ONLY"
DIAGNOSTIC_MODE = "DIAGNOSTIC_ONLY"
SAFETY_MODE = "SHADOW_ONLY"
EVALUATION_MODEL = "10-day final"
NEXT_PHASE = "FASE 2.6B - Controlled Micro-Adjustment Study"
NEXT_VALIDATION_WINDOW = "Novo ciclo PAPER de 10 dias apos estudo 2.6B"

SAFE_RECOMMENDATIONS = {
    "continue_paper_with_controlled_plan",
    "prepare_micro_adjustment_study",
    "observe_more_before_adjustment",
    "no_threshold_change_recommended",
    "no_real_money_recommended",
    "keep_current_strategy_until_next_validation",
    "insufficient_data",
}

BLOCKED_ACTIONS = [
    "start_real_money",
    "lower_global_min_signal_score_now",
    "bypass_guards",
    "remove_no_setup_eligible",
    "remove_reversal_not_eligible",
    "convert_shadow_to_real_signal",
    "use_bos_as_direct_trigger",
    "use_fibonacci_as_direct_trigger",
    "increase_risk_now",
    "change_profile_to_aggressive_now",
]

SUCCESS_CRITERIA_NEXT_CYCLE = [
    "worker_online_consecutive_failures_zero",
    "feed_live_or_acceptable_delayed_without_current_fallback_dominance",
    "paper_trading_preserved",
    "no_real_orders_sent",
    "official_pnl_history_positions_coherent",
    "some_safe_near_approved_candidates_observed",
    "reduce_noise_rejections_without_weak_entries",
    "drawdown_remains_controlled",
    "blockers_explained_clearly",
    "logs_state_ui_coherent",
    "rollback_ready",
]

NO_CHANGE_ITEMS = [
    "min_signal_score_global",
    "broker",
    "real_order_execution",
    "paper_official_order_flow",
    "ticket_capital_max_open_positions",
    "twelve_data_provider_cache_ttl_budget",
    "guards",
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "sim", "y"}
    return bool(value)


def _text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _upper(value: Any, default: str = "") -> str:
    return _text(value, default).upper()


def _safe_recommendation(value: Any, default: str = "observe_more_before_adjustment") -> str:
    recommendation = _text(value, default)
    return recommendation if recommendation in SAFE_RECOMMENDATIONS else default


def _classification_from_report(report: dict[str, Any], final_report: dict[str, Any]) -> str:
    direct = _text(final_report.get("final_classification") or report.get("final_classification"))
    if direct:
        normalized = _upper(direct).replace("_", " ")
        if normalized in {"APROVADO COM RESSALVAS", "APROVADO COM AJUSTES"}:
            return "APROVADO COM RESSALVAS"
        return direct
    grade = _upper(report.get("final_validation_grade"))
    if grade == "APROVADO_COM_AJUSTES":
        return "APROVADO COM RESSALVAS"
    if grade == "APROVADO":
        return "APROVADO PARA CONTINUAR EM PAPER"
    if grade.startswith("REPROVADO"):
        return "REPROVADO"
    return ""


def _operational_status(state: dict[str, Any], report: dict[str, Any]) -> str:
    metrics = _as_dict(report.get("metrics"))
    production = _as_dict(state.get("production"))
    worker_status = _text(state.get("worker_status") or metrics.get("worker_status")).lower()
    heartbeat = _as_float(production.get("heartbeat_age_seconds") or metrics.get("heartbeat_age_seconds"), None)
    consecutive = _as_int(production.get("consecutive_errors") or metrics.get("consecutive_errors"), 0)
    operational_errors = _as_int(metrics.get("operational_errors"), 0)
    worker_ok = worker_status in {"online", "running", "healthy"} or bool(production.get("worker_online"))
    heartbeat_ok = heartbeat is None or heartbeat <= 120
    if worker_ok and heartbeat_ok and consecutive == 0 and operational_errors == 0:
        return "APPROVED"
    if worker_ok and consecutive <= 1:
        return "APPROVED_WITH_MONITORING"
    return "NEEDS_REVIEW"


def _feed_status(state: dict[str, Any], report: dict[str, Any]) -> tuple[str, str, str, int, int, int]:
    market = _as_dict(state.get("market_data"))
    feed_scope = _as_dict(report.get("feed_scope_reconciliation") or state.get("feed_scope_reconciliation"))
    feed_state = _upper(market.get("feed_status") or market.get("status") or report.get("feed_status"), "UNKNOWN")
    provider = _text(market.get("provider_effective") or market.get("provider") or report.get("provider_effective"), "unknown")
    source_breakdown = _as_dict(market.get("source_breakdown"))
    symbols = _as_list(market.get("symbols"))
    live_count = _as_int(market.get("live_count") or source_breakdown.get("market") or source_breakdown.get("live"), 0)
    total_symbols = _as_int(market.get("total_symbols") or len(symbols), live_count)
    if "current_fallback_count" in feed_scope:
        current_fallback = _as_int(feed_scope.get("current_fallback_count"), 0)
    else:
        current_fallback = _as_int(source_breakdown.get("fallback"), 0)
    if feed_state == "LIVE" and total_symbols and live_count >= total_symbols and current_fallback == 0:
        return "APPROVED", feed_state, provider, live_count, total_symbols, current_fallback
    if feed_state in {"LIVE", "DELAYED"} and current_fallback == 0:
        return "APPROVED_WITH_MONITORING", feed_state, provider, live_count, total_symbols, current_fallback
    return "NEEDS_REVIEW", feed_state, provider, live_count, total_symbols, current_fallback


def _paper_mode_confirmed(state: dict[str, Any]) -> bool:
    broker = _as_dict(state.get("broker"))
    validation = _as_dict(state.get("validation"))
    broker_values = {
        _text(broker.get("mode")).lower(),
        _text(broker.get("provider")).lower(),
        _text(validation.get("trading_mode")).lower(),
    }
    return bool({"paper", "simulado", "simulated", "validation"} & broker_values) or not _as_bool(
        validation.get("live_trading_enabled")
    )


def _approval_rate(metrics: dict[str, Any]) -> float:
    direct = _as_float(metrics.get("approval_rate") or metrics.get("signal_approval_rate"), None)
    if direct is not None:
        return direct
    approved = _as_int(metrics.get("signals_approved"), 0)
    total = _as_int(metrics.get("signals_total"), 0)
    if total <= 0:
        total = approved + _as_int(metrics.get("signals_rejected"), 0)
    return 0.0 if total <= 0 else round((approved / total) * 100.0, 4)


def _dominant_assets(*sources: dict[str, Any]) -> list[str]:
    assets: list[str] = []
    keys = ("dominant_asset", "top_asset", "top_symbol", "structural_audit_top_symbol", "market_structure_top_symbol")
    for source in sources:
        for key in keys:
            value = _text(source.get(key)).upper()
            if value and value not in assets:
                assets.append(value)
    return assets[:5]


def _micro_adjustment(
    *,
    item_id: str,
    title: str,
    target: str,
    reason: str,
    evidence: str,
    risk_level: str = "LOW",
    expected_effect: str = "Melhorar leitura diagnostica sem alterar execucao oficial.",
    validation_window: str = NEXT_VALIDATION_WINDOW,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "title": title,
        "target": target,
        "reason": reason,
        "evidence": evidence,
        "risk_level": risk_level,
        "expected_effect": expected_effect,
        "safety_constraints": [
            "PAPER_ONLY",
            "no_real_orders",
            "no_global_threshold_change",
            "no_broker_change",
            "guards_preserved",
            "shadow_diagnostics_not_trade_authority",
        ],
        "allowed_now": False,
        "requires_next_phase": True,
        "validation_window": validation_window,
        "rollback_condition": "Reverter estudo se aumentar entradas fracas, drawdown, inconsistencias de estado ou ruido de sinais.",
    }


def _proposed_micro_adjustments(
    *,
    dominant_bottleneck: str,
    calibration_preview: dict[str, Any],
    taxonomy: dict[str, Any],
    bos_quality: dict[str, Any],
    h1_after_h4: dict[str, Any],
    no_setup: dict[str, Any],
    signal_too_selective: bool,
) -> list[dict[str, Any]]:
    proposals = [
        _micro_adjustment(
            item_id="secondary_confirmation_micro_adjustment_study",
            title="Estudar confirmacao secundaria marginal sem reduzir score global",
            target="secondary_confirmation",
            reason="Gargalos de confirmacao secundaria aparecem junto de score baixo.",
            evidence=dominant_bottleneck or "strategy_bottleneck",
            risk_level="LOW",
            expected_effect="Reduzir rejeicoes por ruido mantendo score global e guards.",
        ),
        _micro_adjustment(
            item_id="breakout_confirmation_rigidity_study",
            title="Estudar rigidez de breakout confirmation",
            target="breakout_confirmation",
            reason="Breakout ausente/fragil apareceu como gargalo recorrente.",
            evidence=_text(taxonomy.get("normalized_primary_reason") or taxonomy.get("taxonomy_status"), "BOS_MISSING"),
            risk_level="LOW",
        ),
        _micro_adjustment(
            item_id="real_rule_mapping_structure_confirmed_study",
            title="Estudar mapeamento da regra real quando estrutura shadow confirma",
            target="trend_pullback_breakout_rule_mapping",
            reason="Estrutura pode aparecer confirmada em shadow sem virar setup real elegivel.",
            evidence=_text(no_setup.get("top_reason_bucket") or no_setup.get("top_real_blocker"), "NO_SETUP_ELIGIBLE"),
            risk_level="MEDIUM",
        ),
        _micro_adjustment(
            item_id="h1_after_h4_bos_mapping_study",
            title="Estudar confirmacao 1H apos BOS/reteste 4H",
            target="h1_after_h4_bos_mapping",
            reason="A confirmacao 1H ainda precisa explicar atrasos, ausencia de BOS ou reteste pendente.",
            evidence=_text(h1_after_h4.get("h1_confirmation_status"), "INSUFFICIENT_DATA_FOR_H1_CONFIRMATION"),
            risk_level="LOW",
        ),
        _micro_adjustment(
            item_id="pullback_quality_study",
            title="Estudar qualidade do pullback antes de qualquer ajuste operacional",
            target="pullback_quality",
            reason="Pullbacks podem estar sendo lidos como estrutura parcial sem reacao objetiva.",
            evidence=_text(bos_quality.get("bos_failure_reason") or taxonomy.get("taxonomy_status"), "BOS_MISSING"),
            risk_level="LOW",
        ),
    ]
    if signal_too_selective or _as_int(calibration_preview.get("near_approved_count"), 0) <= 0:
        proposals.append(
            _micro_adjustment(
                item_id="clean_cycle_baseline_alignment_study",
                title="Estudar reset de ciclo limpo com capital-base alinhado",
                target="next_paper_validation_baseline",
                reason="Nova rodada deve separar resultado acumulado de proxima validacao PAPER.",
                evidence="post_10d_final_cycle",
                risk_level="LOW",
                expected_effect="Melhorar rastreabilidade sem alterar execucao ou risco.",
            )
        )
    return proposals


def _allowed_future_studies(dominant_bottleneck: str, taxonomy: dict[str, Any], h1_after_h4: dict[str, Any]) -> list[str]:
    studies = ["strategy_selectivity_review", "paper_micro_adjustment_design"]
    bottleneck = _upper(dominant_bottleneck)
    if "SECONDARY" in bottleneck or "SCORE" in bottleneck:
        studies.append("secondary_confirmation_micro_adjustment_study")
    if "BREAKOUT" in bottleneck or _upper(taxonomy.get("normalized_primary_reason")) in {"BOS_MISSING", "BREAKOUT_NOT_CONFIRMED"}:
        studies.append("breakout_confirmation_rigidity_study")
    if _text(h1_after_h4.get("h1_confirmation_status")):
        studies.append("h1_after_h4_bos_mapping")
    studies.extend(["bos_confirmation_quality", "pullback_quality", "taxonomy_mapping_readability"])
    return list(dict.fromkeys(studies))


def _plan_status(classification: str, operational_status: str, feed_status: str, signal_too_selective: bool) -> str:
    if not classification:
        return "INSUFFICIENT_DATA_FOR_PLAN"
    if classification == "APROVADO COM RESSALVAS":
        return "OPERATIONAL_APPROVED_STRATEGY_WITH_RESSALVAS"
    if operational_status.startswith("APPROVED") and feed_status.startswith("APPROVED") and signal_too_selective:
        return "STRATEGY_TOO_SELECTIVE_NEEDS_CONTROLLED_STUDY"
    if operational_status.startswith("APPROVED") and feed_status.startswith("APPROVED"):
        return "READY_FOR_PAPER_MICRO_ADJUSTMENT_STUDY"
    return "OPERATIONAL_APPROVED_NEEDS_MORE_DATA"


def default_post_10d_calibration_plan_state(
    reason: str = "No post-10D calibration plan data yet.",
) -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": MODE,
        "diagnostic_mode": DIAGNOSTIC_MODE,
        "safety_mode": SAFETY_MODE,
        "plan_status": "INSUFFICIENT_DATA_FOR_PLAN",
        "generated_at": "",
        "evaluation_model": EVALUATION_MODEL,
        "final_classification": "",
        "operational_status": "UNKNOWN",
        "feed_status": "UNKNOWN",
        "provider_effective": "unknown",
        "worker_reliability_status": "UNKNOWN",
        "ui_state_coherence_status": "UNKNOWN",
        "paper_mode_confirmed": True,
        "real_trade_allowed": False,
        "capital_change_allowed": False,
        "threshold_change_allowed_now": False,
        "strategy_change_allowed_now": False,
        "recommended_next_phase": "",
        "recommended_validation_window": NEXT_VALIDATION_WINDOW,
        "dominant_bottleneck": "",
        "dominant_setup": "",
        "dominant_assets": [],
        "key_findings": [],
        "approved_findings": [],
        "caution_findings": [],
        "blocked_actions": list(BLOCKED_ACTIONS),
        "allowed_future_studies": [],
        "proposed_micro_adjustments": [],
        "proposed_no_change_items": list(NO_CHANGE_ITEMS),
        "required_success_criteria_next_cycle": list(SUCCESS_CRITERIA_NEXT_CYCLE),
        "rollback_requirements": [
            "feature_flag_or_commit_revert_ready",
            "restore_previous_thresholds_and_guards",
            "preserve_official_paper_history_without_rewrite",
        ],
        "recommendation": "insufficient_data",
        "should_continue_paper": True,
        "should_start_real_money": False,
        "should_change_threshold_now": False,
        "should_change_profile_now": False,
        "should_reset_cycle_before_next_validation": False,
        "notes": reason,
        "planning_only": True,
        "diagnostic_only": True,
        "shadow_only": True,
    }


def build_post_10d_calibration_plan(
    *,
    state: Mapping[str, Any] | None = None,
    validation_report: Mapping[str, Any] | None = None,
    cycle_validation: Mapping[str, Any] | None = None,
    final_evaluation_report: Mapping[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Build a planning-only post-10D calibration plan."""
    if not enabled:
        result = default_post_10d_calibration_plan_state("Post-10D calibration plan disabled.")
        result.update({"enabled": False, "plan_status": "CALIBRATION_NOT_AUTHORIZED_YET"})
        return result

    root_state = _as_dict(state)
    report = _as_dict(validation_report) or _as_dict(_as_dict(root_state.get("validation")).get("last_report"))
    cycle = _as_dict(cycle_validation)
    final_report = _as_dict(final_evaluation_report)
    if not report and not final_report and not root_state:
        return default_post_10d_calibration_plan_state()

    strategy_bottleneck = _as_dict(
        report.get("strategy_bottleneck") or cycle.get("strategy_bottleneck") or root_state.get("strategy_bottleneck")
    )
    calibration_preview = _as_dict(
        report.get("calibration_preview") or cycle.get("calibration_preview") or root_state.get("calibration_preview")
    )
    shadow_simulator = _as_dict(
        report.get("shadow_decision_simulator") or cycle.get("shadow_decision_simulator") or root_state.get("shadow_decision_simulator")
    )
    feed_scope = _as_dict(
        report.get("feed_scope_reconciliation") or cycle.get("feed_scope_reconciliation") or root_state.get("feed_scope_reconciliation")
    )
    taxonomy = _as_dict(
        report.get("setup_blocker_taxonomy_audit") or cycle.get("setup_blocker_taxonomy_audit") or root_state.get("setup_blocker_taxonomy_audit")
    )
    bos_quality = _as_dict(
        report.get("bos_confirmation_quality_audit") or cycle.get("bos_confirmation_quality_audit") or root_state.get("bos_confirmation_quality_audit")
    )
    h1_after_h4 = _as_dict(
        report.get("h1_confirmation_after_h4_bos_audit")
        or cycle.get("h1_confirmation_after_h4_bos_audit")
        or root_state.get("h1_confirmation_after_h4_bos_audit")
    )
    no_setup = _as_dict(
        report.get("no_setup_eligible_decomposition") or cycle.get("no_setup_eligible_decomposition") or root_state.get("no_setup_eligible_decomposition")
    )
    bridge = _as_dict(
        report.get("strategy_decision_bridge_trace") or cycle.get("strategy_decision_bridge_trace") or root_state.get("strategy_decision_bridge_trace")
    )
    mtf = _as_dict(
        report.get("multi_timeframe_swing_audit") or cycle.get("multi_timeframe_swing_audit") or root_state.get("multi_timeframe_swing_audit")
    )
    metrics = _as_dict(report.get("metrics"))
    performance = _as_dict(report.get("performance"))
    consistency = _as_dict(report.get("consistency"))

    final_classification = _classification_from_report(report, final_report)
    operational_status = _operational_status(root_state, report)
    feed_status, feed_effective, provider, live_count, total_symbols, current_fallback = _feed_status(root_state, report)
    paper_mode_confirmed = _paper_mode_confirmed(root_state)
    approval_rate = _approval_rate(metrics)
    signals_approved = _as_int(metrics.get("signals_approved"), 0)
    signal_too_selective = bool(approval_rate <= 0.0 or signals_approved == 0)
    preview_safe_count = _as_int(calibration_preview.get("safe_conditions_met_count"), 0)
    preview_near_count = _as_int(calibration_preview.get("near_approved_count"), 0)
    threshold_allowed = False
    dominant_bottleneck = _text(
        strategy_bottleneck.get("dominant_bottleneck") or metrics.get("rejection_top_reason") or taxonomy.get("normalized_primary_reason"),
        "UNKNOWN",
    )
    dominant_setup = _text(
        strategy_bottleneck.get("dominant_setup") or metrics.get("rejection_top_strategy") or taxonomy.get("top_setup"),
        "trend_pullback_breakout",
    )
    dominant_assets = _dominant_assets(
        strategy_bottleneck,
        calibration_preview,
        taxonomy,
        bos_quality,
        h1_after_h4,
        no_setup,
        bridge,
        mtf,
    )
    plan_status = _plan_status(final_classification, operational_status, feed_status, signal_too_selective)
    recommendation = "prepare_micro_adjustment_study" if final_classification == "APROVADO COM RESSALVAS" else "observe_more_before_adjustment"
    if plan_status == "INSUFFICIENT_DATA_FOR_PLAN":
        recommendation = "insufficient_data"
    if signal_too_selective and recommendation != "insufficient_data":
        recommendation = "prepare_micro_adjustment_study"
    recommendation = _safe_recommendation(recommendation)

    approved_findings = []
    if operational_status.startswith("APPROVED"):
        approved_findings.append("operational_infrastructure_approved")
    if feed_status.startswith("APPROVED"):
        approved_findings.append("feed_live_provider_approved")
    if paper_mode_confirmed:
        approved_findings.append("paper_mode_confirmed")
    if _as_float(metrics.get("max_drawdown_pct") or performance.get("max_drawdown_pct"), 0.0) == 0.0:
        approved_findings.append("drawdown_controlled")
    if _as_float(performance.get("profit_factor") or performance.get("payoff"), None) is not None:
        approved_findings.append("profit_factor_observed")

    caution_findings = []
    if final_classification == "APROVADO COM RESSALVAS":
        caution_findings.append("strategy_approved_with_ressalvas")
    if signal_too_selective:
        caution_findings.append("strategy_too_selective")
    if preview_near_count <= 0 or preview_safe_count <= 0:
        caution_findings.append("insufficient_safe_near_approved_sample")
    if current_fallback == 0 and bool(feed_scope.get("current_feed_is_clean", False)):
        caution_findings.append("fallback_not_current_blocker")
    if _text(taxonomy.get("normalized_primary_reason")).upper() in {"BOS_MISSING", "PIVOT_WITHOUT_BOS"}:
        caution_findings.append("structure_diagnostics_not_operational_trigger")

    key_findings = [
        f"final_classification={final_classification or 'unknown'}",
        f"operational_status={operational_status}",
        f"feed_status={feed_status}; feed={feed_effective}; provider={provider}; live={live_count}/{total_symbols}",
        f"dominant_bottleneck={dominant_bottleneck}",
        f"dominant_setup={dominant_setup}",
        f"approval_rate={approval_rate}",
        "real_money_not_authorized",
        "global_threshold_change_not_authorized",
    ]

    proposed_micro_adjustments = _proposed_micro_adjustments(
        dominant_bottleneck=dominant_bottleneck,
        calibration_preview=calibration_preview,
        taxonomy=taxonomy,
        bos_quality=bos_quality,
        h1_after_h4=h1_after_h4,
        no_setup=no_setup,
        signal_too_selective=signal_too_selective,
    )
    allowed_future_studies = _allowed_future_studies(dominant_bottleneck, taxonomy, h1_after_h4)

    return {
        "enabled": True,
        "mode": MODE,
        "diagnostic_mode": DIAGNOSTIC_MODE,
        "safety_mode": SAFETY_MODE,
        "plan_status": plan_status,
        "generated_at": _utc_now_iso(),
        "evaluation_model": EVALUATION_MODEL,
        "final_classification": final_classification,
        "operational_status": operational_status,
        "feed_status": feed_status,
        "provider_effective": provider,
        "worker_reliability_status": operational_status,
        "ui_state_coherence_status": "APPROVED"
        if _text(_as_dict(root_state.get("market_data")).get("state_writer")).lower() == "worker"
        else _text(consistency.get("validation_reading_label"), "UNKNOWN"),
        "paper_mode_confirmed": paper_mode_confirmed,
        "real_trade_allowed": False,
        "capital_change_allowed": False,
        "threshold_change_allowed_now": threshold_allowed,
        "strategy_change_allowed_now": False,
        "recommended_next_phase": NEXT_PHASE if recommendation != "insufficient_data" else "",
        "recommended_validation_window": NEXT_VALIDATION_WINDOW,
        "dominant_bottleneck": dominant_bottleneck,
        "dominant_setup": dominant_setup,
        "dominant_assets": dominant_assets,
        "key_findings": key_findings,
        "approved_findings": approved_findings,
        "caution_findings": caution_findings,
        "blocked_actions": list(BLOCKED_ACTIONS),
        "allowed_future_studies": allowed_future_studies,
        "proposed_micro_adjustments": proposed_micro_adjustments,
        "proposed_no_change_items": list(NO_CHANGE_ITEMS),
        "required_success_criteria_next_cycle": list(SUCCESS_CRITERIA_NEXT_CYCLE),
        "rollback_requirements": [
            "feature_flag_or_commit_revert_ready",
            "restore_previous_thresholds_and_guards",
            "preserve_official_paper_history_without_rewrite",
        ],
        "recommendation": recommendation,
        "should_continue_paper": True,
        "should_start_real_money": False,
        "should_change_threshold_now": False,
        "should_change_profile_now": False,
        "should_reset_cycle_before_next_validation": True,
        "notes": "Planning-only post-10D calibration plan. No trade decision changed.",
        "planning_only": True,
        "diagnostic_only": True,
        "shadow_only": True,
    }


__all__ = [
    "build_post_10d_calibration_plan",
    "default_post_10d_calibration_plan_state",
]
