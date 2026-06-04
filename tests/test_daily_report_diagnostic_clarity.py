from __future__ import annotations

from tests.conftest import load_module


def _module():
    return load_module("core.daily_report_diagnostic_clarity")


def _state(*, worker_status="online", feed="LIVE", provider="twelvedata", context="NEUTRO", errors=0):
    return {
        "worker_status": worker_status,
        "production": {"health_level": "healthy", "consecutive_errors": errors},
        "market_context": {"market_context_status": context},
        "market_data": {
            "feed_status": feed,
            "provider_effective": provider,
            "provider": provider,
            "fallback_count": 0,
        },
    }


def _report(
    *,
    worker_feed="LIVE",
    visual_feed="LIVE",
    worker_provider="twelvedata",
    visual_provider="twelvedata",
    worker_fallback=False,
    visual_only_fallback=False,
    risk_429=False,
    context=None,
    dominant_bottleneck="SCORE_BELOW_MIN",
):
    report = {
        "provider_budget_visual_fallback": {
            "enabled": True,
            "mode": "OBSERVABILITY_ONLY",
            "diagnostic_mode": "DIAGNOSTIC_ONLY",
            "safety_mode": "SHADOW_ONLY",
            "worker_feed_status": worker_feed,
            "visual_feed_status": visual_feed,
            "provider_effective_worker": worker_provider,
            "provider_effective_visual": visual_provider,
            "worker_fallback_operational": worker_fallback,
            "visual_only_fallback": visual_only_fallback,
            "risk_429": risk_429,
            "daily_budget_limit": 800,
            "daily_budget_source": "estimated",
            "daily_budget_status": "DAILY_BUDGET_CONFIGURED_ONLY",
            "minute_limit": 8,
            "minute_limit_source": "estimated",
            "minute_limit_status": "MINUTE_LIMIT_CONFIGURED_ONLY",
            "provider_budget_status": "OK",
            "provider_budget_recommendation": "observe_provider_budget",
        },
        "feed_scope_reconciliation": {
            "current_feed_status": worker_feed,
            "current_fallback_count": 0 if not worker_fallback else 5,
            "fallback_scope_status": "NONE" if not worker_fallback else "CURRENT_CYCLE",
        },
        "strategy_bottleneck": {
            "enabled": True,
            "dominant_bottleneck": dominant_bottleneck,
            "dominant_setup": "trend_pullback_breakout",
        },
    }
    if context is not None:
        report["controlled_micro_adjustment_study"] = {"market_context_status": context}
    return report


def test_worker_live_visual_live_has_high_reliability_and_no_authority(isolated_storage):
    module = _module()
    result = module.build_atlas_daily_decision(_report(), state=_state())

    assert result["mode"] == "REPORT_ONLY"
    assert result["observability_mode"] == "OBSERVABILITY_ONLY"
    assert result["diagnostic_mode"] == "DIAGNOSTIC_ONLY"
    assert result["operational_status"] == "SAUDÁVEL"
    assert result["reading_reliability"] == "ALTA"
    assert result["worker_feed"] == "LIVE"
    assert result["visual_feed"] == "LIVE"
    assert result["paper_required"] is True
    assert result["trade_authority"] is False
    assert result["score_authority"] is False
    assert result["broker_authority"] is False
    assert result["provider_authority"] is False
    assert result["threshold_authority"] is False
    assert result["execution_authority"] is False
    assert result["can_advance_2_6c"] is False


def test_worker_live_visual_fallback_is_medium_reliability_visual_only(isolated_storage):
    module = _module()
    result = module.build_atlas_daily_decision(
        _report(visual_feed="FALLBACK", visual_only_fallback=True),
        state=_state(),
    )
    block = module.format_atlas_daily_decision_block(result)

    assert result["operational_status"] == "ATENÇÃO"
    assert result["reading_reliability"] == "MÉDIA"
    assert result["worker_fallback"] is False
    assert result["visual_only_fallback"] is True
    assert "visual fallback não deve ser confundido com fallback operacional" in block


def test_worker_fallback_by_429_is_low_reliability_feed_block(isolated_storage):
    module = _module()
    report = _report(
        worker_feed="FALLBACK",
        visual_feed="FALLBACK",
        worker_provider="synthetic",
        worker_fallback=True,
        risk_429=True,
    )
    report["provider_budget_visual_fallback"]["provider_budget_status"] = "PROVIDER_RATE_LIMIT_OBSERVED"
    result = module.build_atlas_daily_decision(report, state=_state(feed="FALLBACK", provider="synthetic"))

    assert result["operational_status"] == "BLOQUEADO POR FEED"
    assert result["strategic_status"] == "BLOQUEADO POR FEED"
    assert result["reading_reliability"] == "BAIXA"
    assert result["risk_429"] is True
    assert "nenhuma decisão operacional autorizada" in result["recommended_action"]


def test_context_critical_blocks_strategy_without_operational_authority(isolated_storage):
    module = _module()
    result = module.build_atlas_daily_decision(
        _report(context="CRITICO", dominant_bottleneck="SECONDARY_CONFIRMATION_WEAK"),
        state=_state(context="CRITICO"),
    )

    assert result["strategic_status"] == "BLOQUEADO POR CONTEXTO"
    assert result["trade_authority"] is False
    assert result["threshold_authority"] is False


def test_score_below_min_classifies_score_blocker(isolated_storage):
    module = _module()
    result = module.build_atlas_daily_decision(
        _report(dominant_bottleneck="SCORE_BELOW_MIN"),
        state=_state(),
    )

    assert result["strategic_status"] == "BLOQUEADO POR SCORE"
    assert result["main_blocker"] == "SCORE_BELOW_MIN"
    assert "não alterar threshold" in result["prohibited_actions"]
    assert "não alterar score" in result["prohibited_actions"]


def test_missing_previous_report_uses_safe_comparison_fallback(isolated_storage):
    module = _module()
    result = module.build_atlas_daily_decision(_report(), state=_state())
    block = module.format_atlas_daily_decision_block(result)

    assert result["what_changed_since_yesterday"] == [module.NO_COMPARISON_TEXT]
    assert "O que mudou desde ontem" in block
    assert module.NO_COMPARISON_TEXT in block


def test_report_block_lists_prohibitions_and_next_phase_preconditions(isolated_storage):
    module = _module()
    result = module.build_atlas_daily_decision(_report(), state=_state())
    block = module.format_atlas_daily_decision_block(result)

    assert "Decisão Atlas do Dia" in block
    assert "O que está proibido hoje" in block
    assert "Pré-condições para próxima fase" in block
    assert "Resumo de Provider Budget" in block
    assert "não avançar FASE 2.6C" in block
    assert "revisão humana antes de qualquer PR operacional" in block
    assert "estimated/configured não é medição real" in block
    assert "PAPER TRADING obrigatório" in block
