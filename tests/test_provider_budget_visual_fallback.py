from __future__ import annotations

from tests.conftest import load_module


def _module():
    return load_module("core.provider_budget_visual_fallback")


def _worker_status(**overrides):
    payload = {
        "provider": "twelvedata",
        "provider_effective": "twelvedata",
        "configured_provider": "twelvedata",
        "feed_status": "LIVE",
        "source_breakdown": {"market": 5, "cached": 0, "fallback": 0, "unknown": 0},
        "live_symbols": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD"],
        "fallback_symbols": [],
        "cache_hits": 4,
        "cache_misses": 1,
        "stale_cache_hits": 0,
        "estimated_provider_calls": 1,
        "provider_calls_attempted": 1,
        "provider_calls_skipped": 0,
        "provider_diagnostics": {
            "twelvedata": {
                "api_key_present": True,
                "request_attempted": True,
                "success_count": 5,
                "symbols_total": 5,
            }
        },
    }
    payload.update(overrides)
    return payload


def _visual_status(**overrides):
    payload = {
        "provider": "twelvedata",
        "provider_effective": "twelvedata",
        "feed_status": "LIVE",
        "source_breakdown": {"market": 1, "cached": 0, "fallback": 0, "unknown": 0},
        "fallback_symbols": [],
        "requested_by": "trader_chart",
    }
    payload.update(overrides)
    return payload


def _build(worker=None, visual=None):
    module = _module()
    return module.build_provider_budget_visual_fallback_audit(
        market_data_status=worker or _worker_status(),
        visual_chart_status=visual or _visual_status(),
        feed_scope_reconciliation={
            "fallback_scope_status": "NONE",
            "fallback_blocker_scope": "NONE",
            "current_feed_is_clean": True,
        },
    )


def test_worker_operational_fallback_marks_strategy_reading_unreliable(isolated_storage):
    result = _build(
        worker=_worker_status(
            feed_status="FALLBACK",
            source_breakdown={"market": 0, "cached": 0, "fallback": 5, "unknown": 0},
            fallback_symbols=["BTC-USD", "ETH-USD"],
        ),
    )

    assert result["worker_fallback_operational"] is True
    assert result["visual_only_fallback"] is False
    assert result["worker_strategy_reading_reliable"] is False
    assert result["provider_budget_status"] == "WORKER_OPERATIONAL_FALLBACK"
    assert result["recommendation"] == "mark_worker_fallback_not_reliable"
    assert result["trade_authority"] is False
    assert result["paper_required"] is True


def test_visual_only_fallback_is_separated_from_worker_feed(isolated_storage):
    result = _build(
        worker=_worker_status(feed_status="LIVE"),
        visual=_visual_status(
            feed_status="FALLBACK",
            source_breakdown={"market": 0, "cached": 0, "fallback": 1, "unknown": 0},
            fallback_symbols=["BTC-USD"],
        ),
    )

    assert result["worker_feed_status"] == "LIVE"
    assert result["visual_feed_status"] == "FALLBACK"
    assert result["worker_fallback_operational"] is False
    assert result["visual_only_fallback"] is True
    assert result["worker_strategy_reading_reliable"] is True
    assert result["recommendation"] == "mark_visual_only_fallback"
    assert result["provider_authority"] is False


def test_minutely_maximum_at_limit_is_burst_risk_not_trade_authority(isolated_storage):
    result = _build(
        worker=_worker_status(
            twelvedata_minutely_maximum=8,
            twelvedata_minutely_limit=8,
            twelvedata_minutely_average=3,
        ),
    )

    assert result["minute_limit_status"] == "MINUTE_BURST_RISK"
    assert result["minute_burst_risk"] is True
    assert result["risk_429"] is False
    assert result["trade_authority"] is False
    assert result["should_change_threshold_now"] is False


def test_daily_budget_pressure_warns_without_changing_provider_or_score(isolated_storage):
    result = _build(
        worker=_worker_status(
            twelvedata_daily_credits_used=760,
            twelvedata_daily_credit_limit=800,
        ),
    )

    assert result["daily_budget_status"] == "DAILY_BUDGET_CRITICAL"
    assert result["daily_credit_usage_pct"] == 0.95
    assert result["can_change_provider"] is False
    assert result["can_change_score"] is False
    assert any(row["id"] == "daily_budget_pressure" for row in result["ui_alerts"])


def test_429_is_observed_as_budget_risk_only(isolated_storage):
    result = _build(
        worker=_worker_status(
            response_status_code=429,
            last_error="Twelve Data HTTP 429 credits limit",
        ),
    )

    assert result["risk_429"] is True
    assert result["daily_budget_status"] == "LIMIT_EXCEEDED_OR_429"
    assert result["provider_budget_status"] == "PROVIDER_RATE_LIMIT_OBSERVED"
    assert result["trade_authority"] is False
    assert result["should_start_real_money"] is False


def test_safety_flags_keep_paper_and_no_operational_authority(isolated_storage):
    result = _build()

    assert result["paper_required"] is True
    assert result["observability_only"] is True
    assert result["diagnostic_only"] is True
    assert result["shadow_only"] is True
    assert result["should_continue_paper"] is True
    assert result["should_start_real_money"] is False
    assert result["should_change_threshold_now"] is False
    assert result["should_change_score_now"] is False
    assert result["should_change_profile_now"] is False
    assert result["should_apply_micro_adjustment_now"] is False
    assert result["should_advance_2_6c_now"] is False
    assert result["trade_authority"] is False
    assert result["score_authority"] is False
    assert result["broker_authority"] is False
    assert result["provider_authority"] is False
    assert result["threshold_authority"] is False
    assert result["execution_authority"] is False
    assert "start_real_money" in result["blocked_actions"]
    assert "advance_to_phase_2_6c" in result["blocked_actions"]


def test_log_lines_include_markers_and_safety_payload(isolated_storage):
    module = _module()
    result = _build()
    lines = module.build_provider_budget_visual_fallback_log_lines(result)

    for marker in module.PROVIDER_BUDGET_VISUAL_FALLBACK_MARKERS:
        assert any(marker in line for line in lines)

    safety = next(line for line in lines if "[provider_budget_visual_fallback_safety]" in line)
    assert "should_continue_paper=true" in safety
    assert "should_start_real_money=false" in safety
    assert "should_change_threshold_now=false" in safety
    assert "should_change_profile_now=false" in safety
    assert "should_apply_micro_adjustment_now=false" in safety
    assert "trade_authority=false" in safety
    assert "score_authority=false" in safety
    assert "broker_authority=false" in safety
    assert "provider_authority=false" in safety
    assert "threshold_authority=false" in safety
    assert "execution_authority=false" in safety
    assert "paper_required=true" in safety
