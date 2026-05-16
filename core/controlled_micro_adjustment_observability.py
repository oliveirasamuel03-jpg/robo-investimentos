"""Railway-friendly observability logs for FASE 2.6B.

This module only formats diagnostic log lines. It never mutates state, changes
decisions, touches broker/provider configuration, or applies micro-adjustments.
"""

from __future__ import annotations

from typing import Any, Mapping


CONTROLLED_MICRO_ADJUSTMENT_MARKERS = (
    "[controlled_micro_adjustment_study_summary]",
    "[controlled_micro_adjustment_study_candidates]",
    "[controlled_micro_adjustment_study_selected]",
    "[controlled_micro_adjustment_study_blocked_actions]",
    "[controlled_micro_adjustment_study_safety]",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _text(value: Any, default: str = "unknown") -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        text = default
    return (
        text.replace("\n", " ")
        .replace("\r", " ")
        .replace(";", ",")
        .replace("|", ",")
    )


def _bool_text(value: Any, default: bool = False) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "sim"}:
            return "true"
        if normalized in {"0", "false", "no", "nao", "não"}:
            return "false"
    if value is None:
        return "true" if default else "false"
    return "true" if bool(value) else "false"


def _first_text(default: str, *values: Any) -> str:
    for value in values:
        text = _text(value, "")
        if text:
            return text
    return default


def _provider_effective(report: dict[str, Any], study: dict[str, Any]) -> str:
    market_data = _as_dict(report.get("market_data"))
    feed_quality = _as_dict(report.get("feed_quality"))
    operational = _as_dict(report.get("operational_consistency"))
    return _first_text(
        "unknown",
        study.get("provider_effective"),
        report.get("provider_effective"),
        market_data.get("provider_effective"),
        market_data.get("provider"),
        feed_quality.get("provider_effective"),
        feed_quality.get("provider"),
        operational.get("provider_effective"),
        operational.get("provider"),
    )


def _feed_status(report: dict[str, Any], study: dict[str, Any]) -> str:
    market_data = _as_dict(report.get("market_data"))
    feed_quality = _as_dict(report.get("feed_quality"))
    operational = _as_dict(report.get("operational_consistency"))
    return _first_text(
        "unknown",
        study.get("feed_status"),
        report.get("feed_status"),
        market_data.get("feed_status"),
        market_data.get("status"),
        feed_quality.get("feed_status"),
        feed_quality.get("status"),
        operational.get("feed_status"),
    )


def _broker_status(report: dict[str, Any]) -> str:
    broker = _as_dict(report.get("broker"))
    validation = _as_dict(report.get("validation"))
    operational = _as_dict(report.get("operational_consistency"))
    return _first_text(
        "unknown",
        report.get("broker_status"),
        broker.get("mode"),
        broker.get("provider"),
        validation.get("trading_mode"),
        operational.get("broker_status"),
    )


def _base_payload(report: dict[str, Any], study: dict[str, Any]) -> str:
    return (
        'phase="2.6B";'
        f"mode={_text(study.get('mode'), 'STUDY_ONLY')};"
        f"diagnostic_mode={_text(study.get('diagnostic_mode'), 'DIAGNOSTIC_ONLY')};"
        f"safety_mode={_text(study.get('safety_mode'), 'SHADOW_ONLY')};"
        f"shadow_only={_bool_text(study.get('shadow_only'), True)};"
        "paper_required=true;"
        f"provider_effective={_provider_effective(report, study)};"
        f"feed_status={_feed_status(report, study)};"
        f"broker_status={_broker_status(report)}"
    )


def build_controlled_micro_adjustment_log_lines(
    validation_report: Mapping[str, Any] | None,
) -> list[str]:
    """Return one Railway-searchable line per FASE 2.6B observability marker."""
    report = _as_dict(validation_report)
    study = _as_dict(report.get("controlled_micro_adjustment_study"))
    if not study:
        return []

    candidates = [item for item in _as_list(study.get("candidate_adjustments")) if isinstance(item, Mapping)]
    blocked_actions = [str(item) for item in _as_list(study.get("blocked_actions"))]
    candidate_ids = ",".join(_text(item.get("id"), "unknown") for item in candidates[:8]) or "none"
    blocked_action_ids = ",".join(_text(item, "unknown") for item in blocked_actions[:16]) or "none"
    base = _base_payload(report, study)

    return [
        (
            f"{CONTROLLED_MICRO_ADJUSTMENT_MARKERS[0]} "
            f"{base};"
            f"study_status={_text(study.get('study_status'), 'INSUFFICIENT_DATA_FOR_MICRO_ADJUSTMENT')};"
            f"market_context_status={_text(study.get('market_context_status'), 'unknown')};"
            f"selected_candidate={_text(study.get('selected_candidate_adjustment'), 'none')};"
            f"selected_risk={_text(study.get('selected_candidate_risk_level'), 'BLOCKED')};"
            f"selected_allowed_now={_bool_text(study.get('selected_candidate_allowed_now'), False)};"
            f"requires_next_phase={_bool_text(study.get('selected_candidate_requires_next_phase'), True)};"
            f"recommendation={_text(study.get('recommendation'), 'insufficient_data')}"
        ),
        (
            f"{CONTROLLED_MICRO_ADJUSTMENT_MARKERS[1]} "
            f"{base};"
            f"candidate_count={len(candidates)};"
            f"candidate_ids={candidate_ids};"
            "allowed_now=false;"
            "requires_next_phase=true;"
            "can_change_threshold=false;"
            "can_change_profile=false;"
            "can_affect_real_trade=false"
        ),
        (
            f"{CONTROLLED_MICRO_ADJUSTMENT_MARKERS[2]} "
            f"{base};"
            f"selected_candidate={_text(study.get('selected_candidate_adjustment'), 'none')};"
            f"selected_risk={_text(study.get('selected_candidate_risk_level'), 'BLOCKED')};"
            "selected_allowed_now=false;"
            "requires_next_phase=true;"
            f"recommended_next_phase={_text(study.get('recommended_next_phase'), 'none')}"
        ),
        (
            f"{CONTROLLED_MICRO_ADJUSTMENT_MARKERS[3]} "
            f"{base};"
            f"blocked_actions_count={len(blocked_actions)};"
            f"blocked_actions={blocked_action_ids}"
        ),
        (
            f"{CONTROLLED_MICRO_ADJUSTMENT_MARKERS[4]} "
            f"{base};"
            "should_continue_paper=true;"
            "should_start_real_money=false;"
            "should_change_threshold_now=false;"
            "should_change_profile_now=false;"
            "should_apply_micro_adjustment_now=false;"
            "trade_authority=false;"
            "score_authority=false;"
            "broker_authority=false;"
            "threshold_authority=false;"
            "paper_required=true;"
            "shadow_only=true"
        ),
    ]


__all__ = [
    "CONTROLLED_MICRO_ADJUSTMENT_MARKERS",
    "build_controlled_micro_adjustment_log_lines",
]
