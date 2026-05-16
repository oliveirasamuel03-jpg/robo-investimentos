from __future__ import annotations

from copy import deepcopy

from tests.conftest import load_module


def _module():
    return load_module("core.controlled_micro_adjustment_observability")


def _report() -> dict:
    return {
        "market_data": {
            "feed_status": "LIVE",
            "provider_effective": "twelvedata",
        },
        "broker": {
            "mode": "paper",
            "provider": "paper",
        },
        "controlled_micro_adjustment_study": {
            "mode": "STUDY_ONLY",
            "diagnostic_mode": "DIAGNOSTIC_ONLY",
            "safety_mode": "SHADOW_ONLY",
            "shadow_only": True,
            "study_status": "CONTEXT_NOT_SAFE_FOR_ADJUSTMENT",
            "market_context_status": "DESFAVORAVEL",
            "selected_candidate_adjustment": "real_rule_mapping_study",
            "selected_candidate_risk_level": "MEDIUM",
            "selected_candidate_allowed_now": False,
            "selected_candidate_requires_next_phase": True,
            "recommended_next_phase": "FASE_2_6C_ONLY_IF_CONDITIONS_PASS",
            "recommendation": "observe_more_before_adjustment",
            "candidate_adjustments": [
                {
                    "id": "real_rule_mapping_study",
                    "allowed_now": False,
                    "requires_next_phase": True,
                    "can_change_threshold": False,
                    "can_change_profile": False,
                    "can_affect_real_trade": False,
                }
            ],
            "blocked_actions": [
                "start_real_money",
                "lower_global_min_signal_score_now",
                "apply_micro_adjustment_now",
                "use_h4_bos_as_direct_trigger",
            ],
        },
    }


def test_builds_all_railway_searchable_markers():
    module = _module()
    lines = module.build_controlled_micro_adjustment_log_lines(_report())

    assert len(lines) == 5
    for marker in module.CONTROLLED_MICRO_ADJUSTMENT_MARKERS:
        assert any(marker in line for line in lines)


def test_summary_payload_contains_phase_mode_provider_feed_and_broker():
    lines = _module().build_controlled_micro_adjustment_log_lines(_report())
    summary = next(line for line in lines if "[controlled_micro_adjustment_study_summary]" in line)

    assert 'phase="2.6B"' in summary
    assert "mode=STUDY_ONLY" in summary
    assert "diagnostic_mode=DIAGNOSTIC_ONLY" in summary
    assert "safety_mode=SHADOW_ONLY" in summary
    assert "shadow_only=true" in summary
    assert "paper_required=true" in summary
    assert "provider_effective=twelvedata" in summary
    assert "feed_status=LIVE" in summary
    assert "broker_status=paper" in summary
    assert "study_status=CONTEXT_NOT_SAFE_FOR_ADJUSTMENT" in summary
    assert "selected_candidate=real_rule_mapping_study" in summary
    assert "selected_risk=MEDIUM" in summary
    assert "selected_allowed_now=false" in summary
    assert "requires_next_phase=true" in summary


def test_safety_payload_preserves_all_expected_false_authority_flags():
    lines = _module().build_controlled_micro_adjustment_log_lines(_report())
    safety = next(line for line in lines if "[controlled_micro_adjustment_study_safety]" in line)

    expected_flags = {
        "should_continue_paper=true",
        "should_start_real_money=false",
        "should_change_threshold_now=false",
        "should_change_profile_now=false",
        "should_apply_micro_adjustment_now=false",
        "trade_authority=false",
        "score_authority=false",
        "broker_authority=false",
        "threshold_authority=false",
        "paper_required=true",
        "shadow_only=true",
    }
    for flag in expected_flags:
        assert flag in safety

    assert "trade_authority=true" not in safety
    assert "score_authority=true" not in safety
    assert "broker_authority=true" not in safety
    assert "threshold_authority=true" not in safety


def test_candidate_payload_keeps_micro_adjustments_non_operational():
    lines = _module().build_controlled_micro_adjustment_log_lines(_report())
    candidates = next(line for line in lines if "[controlled_micro_adjustment_study_candidates]" in line)

    assert "candidate_count=1" in candidates
    assert "candidate_ids=real_rule_mapping_study" in candidates
    assert "allowed_now=false" in candidates
    assert "requires_next_phase=true" in candidates
    assert "can_change_threshold=false" in candidates
    assert "can_change_profile=false" in candidates
    assert "can_affect_real_trade=false" in candidates


def test_missing_optional_fields_use_unknown_without_breaking():
    lines = _module().build_controlled_micro_adjustment_log_lines(
        {"controlled_micro_adjustment_study": {"mode": "STUDY_ONLY"}}
    )

    assert len(lines) == 5
    assert all("provider_effective=unknown" in line for line in lines)
    assert all("feed_status=unknown" in line for line in lines)
    assert all("broker_status=unknown" in line for line in lines)


def test_observability_builder_does_not_mutate_report():
    report = _report()
    before = deepcopy(report)

    _module().build_controlled_micro_adjustment_log_lines(report)

    assert report == before
