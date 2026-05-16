from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from core.alerts import send_email_alert, send_final_validation_email, send_recovery_email
from core.config import (
    ALERT_EMAIL_ENABLED,
    BUILD_TIMESTAMP,
    MARKET_DATA_BUILD_LABEL,
    MARKET_DATA_FALLBACK_PROVIDER,
    MARKET_DATA_PROVIDER,
    PRODUCTION_MODE,
    RAILWAY_GIT_COMMIT_SHA,
    SERVICE_NAME,
)
from core.controlled_micro_adjustment_observability import build_controlled_micro_adjustment_log_lines
from core.email_reports import final_report_path_reachable, process_report_email_delivery
from core.market_data import build_feed_quality_snapshot
from core.production_monitor import evaluate_production_health
from core.retention import run_retention_job, should_run_retention_job
from core.state_store import (
    load_bot_state,
    log_event,
    persist_worker_cycle_state,
    save_bot_state,
    update_production_status,
    update_retention_status,
    update_validation_status,
)
from core.swing_validation import refresh_swing_validation_cycle
from engines.trader_engine import refresh_daily_loss_guard, run_trader_cycle

SLEEP_SECONDS = 60
PAUSED_SLEEP_SECONDS = 5
WORKER_RUNTIME_STARTED_AT = datetime.now(timezone.utc).isoformat()
WORKER_PROCESS_ROLE = "worker"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit_startup_marker(message: str) -> None:
    print(message, flush=True)
    log_event("INFO", message)


def emit_worker_observability_log(message: str) -> None:
    """Best-effort stdout + persisted event log for Railway observability."""
    try:
        print(message, flush=True)
    except Exception:
        pass
    try:
        log_event("INFO", message)
    except Exception:
        pass


def update_runtime_state(last_action: str, next_run_delta_seconds: int) -> None:
    persist_worker_cycle_state(
        last_action=last_action,
        next_run_delta_seconds=next_run_delta_seconds,
        worker_status="online",
        runtime_started_at=WORKER_RUNTIME_STARTED_AT,
        process_role=WORKER_PROCESS_ROLE,
    )


def mark_error(message: str) -> None:
    persist_worker_cycle_state(
        last_action=f"Erro: {message}",
        next_run_delta_seconds=SLEEP_SECONDS,
        worker_status="error",
        market_data_payload={
            "requested_by": "worker_cycle",
            "build_active": MARKET_DATA_BUILD_LABEL,
            "git_sha": str(RAILWAY_GIT_COMMIT_SHA or ""),
            "build_timestamp": str(BUILD_TIMESTAMP or ""),
            "service_name": str(SERVICE_NAME or ""),
            "process_role": WORKER_PROCESS_ROLE,
            "last_stage": "worker_exception",
            "last_error": str(message or ""),
        },
        runtime_started_at=WORKER_RUNTIME_STARTED_AT,
        process_role=WORKER_PROCESS_ROLE,
    )


def build_action_text(result: dict) -> str:
    cycle_result = result.get("cycle_result", {}) if isinstance(result, dict) else {}
    trades_executed = int(cycle_result.get("trades_executed", 0))
    if bool(cycle_result.get("entries_blocked_by_daily_loss", False)):
        return "Entradas bloqueadas por limite de perda diaria"

    if trades_executed > 0:
        return f"{trades_executed} trade(s) executado(s) nesta rodada"

    return "Rodada concluida sem novas operacoes"


def _send_health_alert(state: dict, health_payload: dict, *, alert_type: str, current_time: datetime) -> None:
    subject = f"[Trade Ops Desk] Alerta de producao: {alert_type}"
    body = (
        "O monitoramento do trader detectou uma condicao de atencao.\n\n"
        f"Health level: {health_payload.get('health_level')}\n"
        f"Motivo: {health_payload.get('health_reason')}\n"
        f"Mensagem: {health_payload.get('health_message')}\n"
        f"Worker: {health_payload.get('worker_status')}\n"
        f"Heartbeat age (s): {health_payload.get('heartbeat_age_seconds')}\n"
        f"Feed: {health_payload.get('feed_status')}\n"
        f"Broker: {health_payload.get('broker_status')}\n"
        f"Falhas consecutivas: {health_payload.get('consecutive_errors')}\n"
        f"Ultima execucao: {health_payload.get('last_execution_at')}\n"
        f"Ultimo sucesso: {health_payload.get('last_success_at')}\n"
        f"Data UTC: {current_time.isoformat()}\n"
    )
    send_email_alert(subject, body, alert_type=alert_type, state=state, now=current_time)


def _log_cycle_health(health_payload: dict) -> None:
    fallback_flag = 1 if str(health_payload.get("feed_status") or "").strip().upper() == "FALLBACK" else 0
    message = (
        "[cycle_health] "
        f"health_level={str(health_payload.get('health_level') or 'healthy').lower()};"
        f"provider={str(health_payload.get('provider') or 'unknown').lower()};"
        f"feed_status={str(health_payload.get('feed_status') or 'unknown').lower()};"
        f"broker_status={str(health_payload.get('broker_status') or 'paper').lower()};"
        f"worker_status={str(health_payload.get('worker_status') or 'online').lower()};"
        f"consecutive_errors={int(health_payload.get('consecutive_errors') or 0)};"
        f"fallback={fallback_flag}"
    )
    log_event("INFO", message)


def _log_validation_cycle(validation_report: dict) -> None:
    metrics = dict(validation_report.get("metrics", {}) or {})
    rejections = dict(metrics.get("signal_rejections", {}) or {})
    current_context = dict(validation_report.get("current_market_context", {}) or {})
    weak_score_count = int(
        rejections.get("score_below_minimum", rejections.get("weak_score", 0)) or 0
    )
    feed_blocked_count = int(
        rejections.get("feed_quality_blocked", 0) or 0
    ) + int(rejections.get("fallback_blocked", 0) or 0) + int(rejections.get("provider_unknown", 0) or 0)
    message = (
        "[validation_signal] "
        f"day={int(validation_report.get('validation_day_number', 1) or 1)};"
        f"phase={str(validation_report.get('validation_phase') or '').lower().replace(' ', '_')};"
        f"approved={int(metrics.get('signals_approved', 0) or 0)};"
        f"rejected={int(metrics.get('signals_rejected', 0) or 0)};"
        f"against_trend={int(metrics.get('against_trend_entries', 0) or 0)};"
        f"weak_score={weak_score_count};"
        f"feed_unreliable={feed_blocked_count};"
        f"context={str(current_context.get('market_context_status') or 'NEUTRO').upper()};"
        f"context_blocked={int(metrics.get('context_blocked_signals', 0) or 0)}"
    )
    log_event("INFO", message)


def _log_feed_quality_summary(market_data_status: dict | None) -> None:
    quality = build_feed_quality_snapshot(market_data_status)
    success_rate = quality.get("twelvedata_success_rate")
    success_rate_label = "-" if success_rate is None else f"{float(success_rate or 0.0) * 100:.1f}%"
    message = (
        "[feed_quality_summary] "
        f"status={str(quality.get('feed_status') or 'UNKNOWN').lower()};"
        f"provider={str(quality.get('provider_effective') or 'unknown').lower()};"
        f"success_rate={success_rate_label};"
        f"live={int(quality.get('live_count') or 0)}/{int(quality.get('total_symbols') or 0)};"
        f"fallback={int(quality.get('fallback_count') or 0)};"
        f"last_success={str(quality.get('last_success_at') or 'na')};"
        f"reason={str(quality.get('fallback_reason') or 'none')}"
    )
    log_event("INFO", message)


def _log_signal_quality_summary(validation_report: dict) -> None:
    metrics = dict(validation_report.get("metrics", {}) or {})
    consistency = dict(validation_report.get("consistency", {}) or {})
    approval_rate = consistency.get("signal_approval_rate")
    approval_rate_label = "-" if approval_rate is None else f"{float(approval_rate or 0.0) * 100:.1f}%"
    message = (
        "[signal_quality_summary] "
        f"approved={int(metrics.get('signals_approved', 0) or 0)};"
        f"rejected={int(metrics.get('signals_rejected', 0) or 0)};"
        f"approval_rate={approval_rate_label};"
        f"sample={str(consistency.get('sample_quality_label') or 'Sem leitura').lower()};"
        f"posture={str(consistency.get('operational_posture_label') or 'Indefinida').lower()};"
        f"watchlist={'coerente' if bool(consistency.get('watchlist_phase_aligned')) else 'fora_da_fase'};"
        f"reading={str(consistency.get('signal_quality_label') or 'Baixa').lower()}"
    )
    log_event("INFO", message)


def _log_signal_rejection_summary(validation_report: dict, cycle_result: dict | None) -> None:
    cycle_validation = dict((cycle_result or {}).get("validation_cycle", {}) or {})
    rejection_summary = dict(cycle_validation.get("rejection_summary", {}) or validation_report.get("rejection_quality", {}) or {})
    top_reason = str(rejection_summary.get("top_rejection_reason") or rejection_summary.get("top_reason") or "")
    top_layer = str(rejection_summary.get("top_rejection_layer") or rejection_summary.get("top_layer") or "")
    top_strategy = str(rejection_summary.get("top_rejection_strategy") or rejection_summary.get("top_strategy") or "")
    reason_breakdown = dict(rejection_summary.get("rejected_by_reason", {}) or rejection_summary.get("reason_breakdown", {}) or {})
    layer_breakdown = dict(rejection_summary.get("rejected_by_layer", {}) or rejection_summary.get("layer_breakdown", {}) or {})
    log_event(
        "INFO",
        (
            "[signal_rejection_summary] "
            f"rejected={int(rejection_summary.get('total_rejected_signals', 0) or 0)};"
            f"events={int(rejection_summary.get('total_rejection_events', 0) or 0)};"
            f"top_reason={top_reason or 'none'};"
            f"top_layer={top_layer or 'none'};"
            f"top_strategy={top_strategy or 'none'}"
        ),
    )
    if top_reason:
        log_event(
            "INFO",
            (
                "[signal_rejection_top_reason] "
                f"reason={top_reason};"
                f"count={int(reason_breakdown.get(top_reason, 0) or 0)}"
            ),
        )
    if layer_breakdown:
        summary = ",".join(
            f"{str(layer)}:{int(count or 0)}"
            for layer, count in sorted(layer_breakdown.items(), key=lambda item: int(item[1] or 0), reverse=True)
        )
        log_event("INFO", f"[signal_rejection_layer_summary] {summary}")
    feed_diag = dict(validation_report.get("feed_rejection_consistency", {}) or {})
    if feed_diag:
        log_event(
            "INFO",
            (
                "[feed_rejection_consistency] "
                f"feed_status={str(feed_diag.get('feed_status') or 'UNKNOWN').lower()};"
                f"provider={str(feed_diag.get('provider_effective') or 'unknown').lower()};"
                f"live={int(feed_diag.get('live_assets_count', 0) or 0)};"
                f"fallback={int(feed_diag.get('fallback_assets_count', 0) or 0)};"
                f"scope={str(feed_diag.get('dominant_rejection_scope') or 'unknown')};"
                f"stale_fallback={1 if bool(feed_diag.get('possible_stale_fallback_label')) else 0};"
                f"current_fallback={1 if bool(feed_diag.get('is_fallback_rejection_current')) else 0}"
            ),
        )


def _log_calibration_preview_summary(validation_report: dict) -> None:
    preview = dict(validation_report.get("calibration_preview", {}) or {})
    if not preview:
        return
    log_event(
        "INFO",
        (
            "[calibration_preview_summary] "
            f"mode={str(preview.get('mode') or 'PREVIEW_ONLY').lower()};"
            f"near={int(preview.get('near_approved_count', 0) or 0)};"
            f"rate={float(preview.get('near_approved_rate', 0.0) or 0.0):.4f};"
            f"best={preview.get('best_score_seen') if preview.get('best_score_seen') is not None else 'none'};"
            f"min={preview.get('min_score_current') if preview.get('min_score_current') is not None else 'none'};"
            f"floor={preview.get('preview_score_floor') if preview.get('preview_score_floor') is not None else 'none'};"
            "trade_approval_changed=0"
        ),
    )


def _log_strategy_bottleneck_summary(validation_report: dict) -> None:
    bottleneck = dict(validation_report.get("strategy_bottleneck", {}) or {})
    if not bottleneck:
        return
    log_event(
        "INFO",
        (
            "[strategy_bottleneck_summary] "
            f"mode={str(bottleneck.get('mode') or 'DIAGNOSTIC_ONLY').lower()};"
            f"dominant={str(bottleneck.get('dominant_bottleneck') or 'none')};"
            f"setup={str(bottleneck.get('dominant_setup') or 'none')};"
            f"asset={str(bottleneck.get('dominant_asset') or 'none')};"
            f"strategy_rejections={int(bottleneck.get('total_strategy_rejections', 0) or 0)};"
            "trade_approval_changed=0"
        ),
    )


def _log_strategy_structure_audit_summary(validation_report: dict) -> None:
    audit = dict(validation_report.get("strategy_structure_audit", {}) or {})
    if not audit:
        return
    comparison = list(audit.get("structural_audit_setup_comparison", []) or [])
    recent = list(audit.get("structural_audit_recent_candidates", []) or [])
    log_event(
        "INFO",
        (
            "[strategy_structure_audit_summary] "
            f"mode={str(audit.get('structural_audit_mode') or 'SHADOW_ONLY').lower()};"
            f"candidates={int(audit.get('structural_audit_candidates', 0) or 0)};"
            f"top_setup={str(audit.get('structural_audit_top_setup') or 'none')};"
            f"top_symbol={str(audit.get('structural_audit_top_symbol') or 'none')};"
            f"top_score={audit.get('structural_audit_top_score') if audit.get('structural_audit_top_score') is not None else 'none'};"
            f"top_gap={audit.get('structural_audit_top_gap') if audit.get('structural_audit_top_gap') is not None else 'none'};"
            f"recommendation={str(audit.get('structural_audit_recommendation') or 'sem_dados_suficientes').replace(' ', '_')};"
            "trade_approval_changed=0"
        ),
    )
    if recent:
        top = dict(recent[0] or {})
        log_event(
            "INFO",
            (
                "[strategy_structure_audit_top_candidate] "
                f"setup={str(top.get('setup_name') or 'none')};"
                f"symbol={str(top.get('symbol') or 'none')};"
                f"score={top.get('score') if top.get('score') is not None else 'none'};"
                f"gap={top.get('score_gap') if top.get('score_gap') is not None else 'none'};"
                f"primary={','.join(list(top.get('primary_blockers', []) or [])) or 'none'};"
                f"secondary={','.join(list(top.get('secondary_blockers', []) or [])) or 'none'};"
                f"recommendation={str(top.get('recommendation') or 'none').replace(' ', '_')};"
                "shadow_only=1"
            ),
        )
    for item in comparison[:3]:
        row = dict(item or {})
        log_event(
            "INFO",
            (
                "[strategy_structure_audit_setup_comparison] "
                f"setup={str(row.get('setup') or 'unknown')};"
                f"shadow_candidates={int(row.get('shadow_candidates', 0) or 0)};"
                f"best_symbol={str(row.get('best_symbol') or 'none')};"
                f"average_gap={row.get('average_gap') if row.get('average_gap') is not None else 'none'};"
                f"dominant_blocker={str(row.get('dominant_blocker') or 'none')};"
                f"recommendation={str(row.get('recommendation') or 'none').replace(' ', '_')};"
                "shadow_only=1"
            ),
        )


def _log_market_structure_audit_summary(validation_report: dict) -> None:
    audit = dict(validation_report.get("market_structure_audit", {}) or {})
    if not audit:
        return
    candidates = list(audit.get("market_structure_best_candidates", []) or [])
    confluence = dict(audit.get("market_structure_setup_confluence", {}) or {})
    log_event(
        "INFO",
        (
            "[market_structure_audit_summary] "
            f"mode={str(audit.get('market_structure_audit_mode') or 'SHADOW_ONLY').lower()};"
            f"top_symbol={str(audit.get('market_structure_top_symbol') or 'none')};"
            f"top_score={audit.get('market_structure_top_score') if audit.get('market_structure_top_score') is not None else 'none'};"
            f"top_zone={str(audit.get('market_structure_top_zone') or 'none')};"
            f"candidates={int(audit.get('market_structure_candidates_count', 0) or 0)};"
            f"recommendation={str(audit.get('market_structure_top_recommendation') or 'sem_dados_suficientes').replace(' ', '_')};"
            "shadow_only=true"
        ),
    )
    if candidates:
        top = dict(candidates[0] or {})
        log_event(
            "INFO",
            (
                "[market_structure_top_candidate] "
                f"symbol={str(top.get('symbol') or 'none')};"
                f"score={top.get('market_structure_score') if top.get('market_structure_score') is not None else 'none'};"
                f"fib_zone={str(top.get('current_fib_zone') or 'none')};"
                f"bos_detected={int(bool(top.get('bos_detected', False)))};"
                f"pivot_detected={int(bool(top.get('pivot_detected', False)))};"
                f"false_breakout_risk={int(bool(top.get('false_breakout_risk', False)))};"
                f"recommendation={str(top.get('structure_recommendation') or 'none').replace(' ', '_')};"
                "shadow_only=true"
            ),
        )
        log_event(
            "INFO",
            (
                "[market_structure_fib_confluence] "
                f"symbol={str(top.get('symbol') or 'none')};"
                f"fib_zone={str(top.get('current_fib_zone') or 'none')};"
                f"trend_pullback={int(bool(top.get('structure_confirms_trend_pullback', False)))};"
                f"breakout={int(bool(top.get('structure_confirms_breakout', False)))};"
                f"reversal={int(bool(top.get('structure_confirms_reversal', False)))};"
                "shadow_only=true"
            ),
        )
    log_event(
        "INFO",
        (
            "[market_structure_setup_comparison] "
            f"trend_pullback={int(confluence.get('trend_pullback', 0) or 0)};"
            f"breakout={int(confluence.get('breakout', 0) or 0)};"
            f"reversal={int(confluence.get('reversal', 0) or 0)};"
            f"would_improve_quality={int(confluence.get('would_improve_quality', 0) or 0)};"
            "shadow_only=true"
        ),
    )


def _log_fib_alignment_audit_summary(validation_report: dict) -> None:
    audit = dict(validation_report.get("fib_alignment_audit", {}) or {})
    if not audit:
        return
    score = audit.get("fib_alignment_score")
    score_text = "none" if score is None else f"{float(score):.4f}"
    log_event(
        "INFO",
        (
            "[fib_alignment_audit_summary] "
            f"mode={str(audit.get('fib_alignment_mode') or 'SHADOW_ONLY').lower()};"
            f"symbol={str(audit.get('fib_alignment_top_symbol') or 'none')};"
            f"score={score_text};"
            f"status={str(audit.get('fib_alignment_status') or 'insufficient_data')};"
            f"recommendation={str(audit.get('fib_alignment_recommendation') or 'insufficient_data')};"
            "shadow_only=true"
        ),
    )
    checklist = [item for item in list(audit.get("fib_alignment_checklist", []) or []) if isinstance(item, dict)]
    for row in checklist[:3]:
        log_event(
            "INFO",
            (
                "[fib_alignment_rule_comparison] "
                f"item={str(row.get('item') or 'unknown').replace(' ', '_')};"
                f"expected={str(row.get('esperado_pelo_video_pdf') or 'none').replace(' ', '_')};"
                f"detected={str(row.get('detectado_pelo_app') or 'none').replace(' ', '_')};"
                f"status={str(row.get('status') or 'unknown')};"
                "shadow_only=true"
            ),
        )
    why_differs = str(audit.get("fib_alignment_why_differs") or "").strip()
    if why_differs:
        log_event(
            "INFO",
            (
                "[fib_alignment_divergence] "
                f"symbol={str(audit.get('fib_alignment_top_symbol') or 'none')};"
                f"why={why_differs.replace(' ', '_')};"
                "shadow_only=true"
            ),
        )


def _log_multi_timeframe_swing_audit_summary(validation_report: dict) -> None:
    audit = dict(validation_report.get("multi_timeframe_swing_audit", {}) or {})
    if not audit:
        return
    score = audit.get("top_alignment_score")
    score_text = "none" if score is None else f"{float(score):.4f}"
    log_event(
        "INFO",
        (
            "[multi_tf_swing_audit_summary] "
            f"mode={str(audit.get('mode') or 'SHADOW_ONLY').lower()};"
            f"feed_status={str(audit.get('feed_status') or 'UNKNOWN')};"
            f"provider={str(audit.get('provider_effective') or 'unknown')};"
            f"symbols={int(audit.get('symbols_analyzed', 0) or 0)};"
            f"top_symbol={str(audit.get('top_symbol') or 'none')};"
            f"score={score_text};"
            f"status={str(audit.get('top_alignment_status') or 'INSUFFICIENT_DATA')};"
            f"missing={str(audit.get('top_missing_confirmation') or 'none')};"
            f"calls={int(audit.get('estimated_provider_calls', 0) or 0)};"
            f"cache={str(audit.get('cache_status') or 'cycle_data_resample_only')};"
            "shadow_only=true"
        ),
    )
    candidates = [item for item in list(audit.get("recent_candidates", []) or []) if isinstance(item, dict)]
    for item in candidates[:3]:
        log_event(
            "INFO",
            (
                "[multi_tf_swing_audit_candidate] "
                f"symbol={str(item.get('symbol') or 'none')};"
                f"daily={str(item.get('daily_bias') or 'INCONCLUSIVE')};"
                f"h4={str(item.get('h4_structure') or 'INCONCLUSIVE')};"
                f"h1={str(item.get('h1_confirmation') or 'INCONCLUSIVE')};"
                f"alignment={item.get('alignment_score') if item.get('alignment_score') is not None else 'none'};"
                f"status={str(item.get('alignment_status') or 'INSUFFICIENT_DATA')};"
                f"support={int(bool(item.get('supports_trend_pullback_breakout', False)))};"
                f"keep_blocked={int(bool(item.get('should_keep_blocked', True)))};"
                "shadow_only=true"
            ),
        )
    log_event(
        "INFO",
        (
            "[multi_tf_swing_alignment] "
            f"strong={int(audit.get('strong_alignment_count', 0) or 0)};"
            f"partial={int(audit.get('partial_alignment_count', 0) or 0)};"
            f"conflict={int(audit.get('conflict_count', 0) or 0)};"
            f"insufficient={int(audit.get('insufficient_data_count', 0) or 0)};"
            f"setup_support={int(audit.get('setup_support_count', 0) or 0)};"
            "shadow_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[multi_tf_swing_missing_confirmation] "
            f"dominant={str(audit.get('dominant_conflict_reason') or audit.get('top_missing_confirmation') or 'none')};"
            f"recommendation={str(audit.get('top_recommendation') or 'observe_more')};"
            "shadow_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[multi_tf_swing_provider_guard] "
            f"guard={str(audit.get('provider_guard') or 'not_evaluated')};"
            f"cache_ttl={int(audit.get('cache_ttl_seconds', 0) or 0)};"
            f"estimated_provider_calls={int(audit.get('estimated_provider_calls', 0) or 0)};"
            "shadow_only=true"
        ),
    )


def _log_multi_timeframe_intraday_fetch_summary(validation_report: dict) -> None:
    fetcher = dict(validation_report.get("multi_timeframe_intraday_fetcher", {}) or {})
    audit = dict(validation_report.get("multi_timeframe_swing_audit", {}) or {})
    if not fetcher:
        return
    log_event(
        "INFO",
        (
            "[multi_tf_intraday_fetch_summary] "
            f"enabled={int(bool(fetcher.get('enabled', True)))};"
            f"mode={str(fetcher.get('mode') or 'SHADOW_ONLY').lower()};"
            f"feed_status={str(fetcher.get('feed_status') or 'UNKNOWN')};"
            f"provider={str(fetcher.get('provider_effective') or 'unknown')};"
            f"symbols_requested={len(list(fetcher.get('symbols_requested', []) or []))};"
            f"symbols_fetched={len(list(fetcher.get('symbols_fetched', []) or []))};"
            f"timeframes={','.join(list(fetcher.get('timeframes_requested', []) or []))};"
            f"quality={str(fetcher.get('intraday_data_quality') or 'NO_DATA')};"
            "shadow_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[multi_tf_intraday_fetch_cache] "
            f"hits={int(fetcher.get('cache_hits', 0) or 0)};"
            f"misses={int(fetcher.get('cache_misses', 0) or 0)};"
            f"calls_attempted={int(fetcher.get('provider_calls_attempted', 0) or 0)};"
            f"calls_skipped={int(fetcher.get('provider_calls_skipped', 0) or 0)};"
            f"estimated_calls={int(fetcher.get('estimated_provider_calls', 0) or 0)};"
            "shadow_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[multi_tf_intraday_fetch_provider_guard] "
            f"budget_guard={int(bool(fetcher.get('provider_budget_guard_active', False)))};"
            f"reason={str(fetcher.get('provider_guard_reason') or 'none')};"
            f"recommendation={str(fetcher.get('intraday_fetch_recommendation') or 'observe_more')};"
            "shadow_only=true"
        ),
    )
    diagnostics = [item for item in list(fetcher.get("diagnostics", []) or []) if isinstance(item, dict)]
    for row in diagnostics[:5]:
        log_event(
            "INFO",
            (
                "[multi_tf_intraday_fetch_symbol] "
                f"symbol={str(row.get('symbol') or 'none')};"
                f"timeframe={str(row.get('timeframe') or 'none')};"
                f"candles={int(row.get('candles_available', 0) or 0)};"
                f"quality={str(row.get('data_quality') or 'missing')};"
                f"cache={str(row.get('cache_status') or 'unknown')};"
                f"call_attempted={int(bool(row.get('provider_call_attempted', False)))};"
                "shadow_only=true"
            ),
        )
    if str(fetcher.get("last_error") or "").strip():
        log_event(
            "WARNING",
            (
                "[multi_tf_intraday_fetch_error] "
                f"error={str(fetcher.get('last_error') or '').replace(' ', '_')};"
                "shadow_only=true"
            ),
        )
    log_event(
        "INFO",
        (
            "[multi_tf_swing_audit_with_intraday] "
            f"uses_real_intraday={int(bool(audit.get('uses_real_intraday_data', False)))};"
            f"available={','.join(list(audit.get('intraday_timeframes_available', []) or []))};"
            f"top_symbol={str(audit.get('intraday_top_symbol') or audit.get('top_symbol') or 'none')};"
            f"missing={str(audit.get('intraday_missing_reason') or 'none')};"
            "shadow_only=true"
        ),
    )


def _log_bos_pivot_trace_audit_summary(validation_report: dict) -> None:
    audit = dict(validation_report.get("bos_pivot_trace_audit", {}) or {})
    if not audit:
        return
    log_event(
        "INFO",
        (
            "[bos_pivot_trace_summary] "
            f"mode={str(audit.get('mode') or 'SHADOW_ONLY').lower()};"
            f"feed={str(audit.get('feed_status') or 'UNKNOWN')};"
            f"provider={str(audit.get('provider_effective') or 'unknown')};"
            f"top_symbol={str(audit.get('top_symbol') or 'none')};"
            f"top_timeframe={str(audit.get('top_timeframe') or 'none')};"
            f"top_pivot={str(audit.get('top_pivot_state') or 'INSUFFICIENT_DATA')};"
            f"top_bos={str(audit.get('top_bos_state') or 'INSUFFICIENT_DATA')};"
            f"relationship={str(audit.get('top_relationship') or 'INSUFFICIENT_DATA')};"
            f"missing={str(audit.get('dominant_missing_piece') or audit.get('top_primary_missing_piece') or 'none')};"
            "shadow_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[bos_pivot_trace_missing_piece] "
            f"h4_bos_missing={int(audit.get('h4_bos_missing_count', 0) or 0)};"
            f"h1_bos_only={int(audit.get('h1_bos_only_count', 0) or 0)};"
            f"wick_only={int(audit.get('wick_only_bos_count', 0) or 0)};"
            f"weak_close={int(audit.get('weak_close_bos_count', 0) or 0)};"
            f"confirmed={int(audit.get('confirmed_bos_count', 0) or 0)};"
            f"keep_blocked={int(audit.get('should_keep_blocked_count', 0) or 0)};"
            "shadow_only=true"
        ),
    )
    candidates = [item for item in list(audit.get("recent_candidates", []) or []) if isinstance(item, dict)]
    for row in candidates[:4]:
        tag = "[bos_pivot_trace_h4] " if str(row.get("timeframe") or "") == "4h" else "[bos_pivot_trace_h1] "
        log_event(
            "INFO",
            (
                f"{tag}"
                f"symbol={str(row.get('symbol') or 'none')};"
                f"timeframe={str(row.get('timeframe') or 'none')};"
                f"trend={str(row.get('trend_direction') or 'INCONCLUSIVE')};"
                f"pivot={str(row.get('pivot_state') or 'INSUFFICIENT_DATA')};"
                f"bos={str(row.get('bos_state') or 'INSUFFICIENT_DATA')};"
                f"close_distance={row.get('close_distance_to_bos_pct') if row.get('close_distance_to_bos_pct') is not None else 'none'};"
                f"wick_crossed={int(bool(row.get('wick_crossed_level', False)))};"
                f"close_confirmed={int(bool(row.get('close_confirmed_level', False)))};"
                f"keep_blocked={int(bool(row.get('should_keep_blocked', True)))};"
                "shadow_only=true"
            ),
        )
    if candidates:
        top = candidates[0]
        log_event(
            "INFO",
            (
                "[bos_pivot_trace_relationship] "
                f"symbol={str(top.get('symbol') or 'none')};"
                f"relationship={str(top.get('relationship_to_higher_tf') or 'INSUFFICIENT_DATA')};"
                f"reason={str(top.get('relationship_reason') or 'none')};"
                "shadow_only=true"
            ),
        )
    log_event(
        "INFO",
        (
            "[bos_pivot_trace_safety] "
            "should_keep_blocked=true;"
            "trade_authority=false;"
            "score_authority=false;"
            "broker_authority=false;"
            "paper_required=true;"
            "shadow_only=true"
        ),
    )


def _log_strategy_decision_bridge_trace_summary(validation_report: dict) -> None:
    bridge = dict(validation_report.get("strategy_decision_bridge_trace", {}) or {})
    if not bridge:
        return
    log_event(
        "INFO",
        (
            "[strategy_decision_bridge_summary] "
            f"mode={str(bridge.get('mode') or 'SHADOW_ONLY').lower()};"
            f"top_symbol={str(bridge.get('top_symbol') or 'none')};"
            f"bridge_status={str(bridge.get('top_bridge_status') or 'INSUFFICIENT_TRACE_DATA')};"
            f"real_blocker={str(bridge.get('top_real_blocker') or 'none')};"
            f"structure_status={str(bridge.get('top_structure_status') or 'none')};"
            f"reconciliation={str(bridge.get('top_reconciliation_status') or 'UNKNOWN_MISMATCH')};"
            f"recommendation={str(bridge.get('recommendation') or 'observe_more')};"
            "shadow_only=true"
        ),
    )
    candidates = [item for item in list(bridge.get("recent_candidates", []) or []) if isinstance(item, dict)]
    for row in candidates[:4]:
        log_event(
            "INFO",
            (
                "[strategy_decision_bridge_candidate] "
                f"symbol={str(row.get('symbol') or 'none')};"
                f"score={row.get('real_score') if row.get('real_score') is not None else 'none'};"
                f"min_score={row.get('min_score') if row.get('min_score') is not None else 'none'};"
                f"gap={row.get('score_gap') if row.get('score_gap') is not None else 'none'};"
                f"bridge_status={str(row.get('decision_bridge_status') or 'unknown')};"
                f"real_blocker={str(row.get('primary_real_blocker') or row.get('real_bottleneck_category') or 'none')};"
                f"fallback_scope={str(row.get('fallback_blocker_scope') or 'UNKNOWN')};"
                f"keep_blocked={int(bool(row.get('should_keep_blocked', True)))};"
                "shadow_only=true"
            ),
        )
    if candidates:
        top = candidates[0]
        log_event(
            "INFO",
            (
                "[strategy_decision_bridge_reconciliation] "
                f"symbol={str(top.get('symbol') or 'none')};"
                f"multi_tf={str(top.get('multi_tf_alignment_status') or 'none')};"
                f"h4_bos={str(top.get('bos_state_4h') or 'none')};"
                f"h1_bos={str(top.get('bos_state_1h') or 'none')};"
                f"reconciliation={str(top.get('reconciliation_status') or 'UNKNOWN_MISMATCH')};"
                f"reason={str(top.get('reconciliation_reason') or 'none').replace(' ', '_')};"
                "shadow_only=true"
            ),
        )
        log_event(
            "INFO",
            (
                "[strategy_decision_bridge_real_blocker] "
                f"symbol={str(top.get('symbol') or 'none')};"
                f"primary={str(top.get('primary_real_blocker') or 'none')};"
                f"secondary={str(top.get('secondary_real_blocker') or 'none')};"
                f"reason={str(top.get('real_rejection_reason') or 'none').replace(' ', '_')};"
                "shadow_only=true"
            ),
        )
    log_event(
        "INFO",
        (
            "[strategy_decision_bridge_safety] "
            f"keep_blocked={int(bridge.get('should_keep_blocked_count', 0) or 0)};"
            "trade_authority=false;"
            "score_authority=false;"
            "broker_authority=false;"
            "paper_required=true;"
            "shadow_only=true"
        ),
    )


def _log_feed_scope_reconciliation_summary(validation_report: dict) -> None:
    feed_scope = dict(validation_report.get("feed_scope_reconciliation", {}) or {})
    if not feed_scope:
        return
    log_event(
        "INFO",
        (
            "[feed_scope_reconciliation_summary] "
            f"mode={str(feed_scope.get('mode') or 'DIAGNOSTIC_ONLY').lower()};"
            f"current_feed_status={str(feed_scope.get('current_feed_status') or 'UNKNOWN')};"
            f"current_fallback_count={int(feed_scope.get('current_fallback_count', 0) or 0)};"
            f"current_live_count={int(feed_scope.get('current_live_count', 0) or 0)};"
            f"accumulated_fallback_count={int(feed_scope.get('accumulated_fallback_count', 0) or 0)};"
            f"fallback_scope_status={str(feed_scope.get('fallback_scope_status') or 'UNKNOWN_SCOPE')};"
            f"fallback_blocker_scope={str(feed_scope.get('fallback_blocker_scope') or 'UNKNOWN')};"
            f"current_feed_clean={int(bool(feed_scope.get('current_feed_is_clean', False)))};"
            f"recommendation={str(feed_scope.get('recommendation') or 'observe_more')};"
            "diagnostic_only=true"
        ),
    )
    flags = dict(feed_scope.get("candidate_fallback_flags", {}) or {})
    log_event(
        "INFO",
        (
            "[feed_scope_reconciliation_current] "
            f"worker_feed={str(feed_scope.get('worker_feed_status') or 'UNKNOWN')};"
            f"provider={str(feed_scope.get('provider_effective') or 'unknown')};"
            f"live={int(feed_scope.get('current_live_count', 0) or 0)};"
            f"fallback={int(feed_scope.get('current_fallback_count', 0) or 0)};"
            f"unknown={int(feed_scope.get('current_cycle_unknown_count', 0) or 0)};"
            f"current_rejection={str(feed_scope.get('dominant_rejection_current') or 'none')};"
            "diagnostic_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[feed_scope_reconciliation_accumulated] "
            f"accumulated_fallback={int(feed_scope.get('accumulated_fallback_count', 0) or 0)};"
            f"historical_fallback={int(feed_scope.get('historical_fallback_count', 0) or 0)};"
            f"accumulated_strategy={int(feed_scope.get('accumulated_strategy_count', 0) or 0)};"
            f"accumulated_rejection={str(feed_scope.get('dominant_rejection_accumulated') or 'none')};"
            "diagnostic_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[feed_scope_reconciliation_candidate] "
            f"current_candidate_fallback={int(flags.get('current_cycle_candidate_fallback_count', 0) or 0)};"
            f"accumulated_candidate_fallback={int(flags.get('accumulated_candidate_fallback_count', 0) or 0)};"
            f"visual_chart_fallback={int(bool(flags.get('visual_chart_fallback', False)))};"
            f"visual_feed={str(feed_scope.get('visual_feed_status') or 'UNKNOWN')};"
            "diagnostic_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[feed_scope_reconciliation_safety] "
            "trade_authority=false;"
            "score_authority=false;"
            "broker_authority=false;"
            "threshold_authority=false;"
            "paper_required=true;"
            "diagnostic_only=true"
        ),
    )


def _log_no_setup_eligible_decomposition_summary(validation_report: dict) -> None:
    decomposition = dict(validation_report.get("no_setup_eligible_decomposition", {}) or {})
    if not decomposition:
        return
    log_event(
        "INFO",
        (
            "[no_setup_decomposition_summary] "
            f"mode={str(decomposition.get('mode') or 'DIAGNOSTIC_ONLY').lower()};"
            f"safety_mode={str(decomposition.get('safety_mode') or 'SHADOW_ONLY').lower()};"
            f"top_symbol={str(decomposition.get('top_symbol') or 'none')};"
            f"top_setup={str(decomposition.get('top_setup') or 'trend_pullback_breakout')};"
            f"top_reason_bucket={str(decomposition.get('top_reason_bucket') or 'INSUFFICIENT_DATA_FOR_DECOMPOSITION')};"
            f"no_setup_count={int(decomposition.get('no_setup_eligible_count', 0) or 0)};"
            f"structure_confirmed_but_no_setup={int(decomposition.get('structure_confirmed_but_no_setup_count', 0) or 0)};"
            f"near_approved_no_setup={int(decomposition.get('near_approved_no_setup_count', 0) or 0)};"
            f"current_feed_clean={int(bool(decomposition.get('current_feed_is_clean', False)))};"
            f"fallback_scope={str(decomposition.get('fallback_blocker_scope') or 'UNKNOWN')};"
            f"recommendation={str(decomposition.get('recommendation') or 'insufficient_data')};"
            f"keep_blocked={int(bool(decomposition.get('should_keep_blocked', True)))};"
            "diagnostic_only=true"
        ),
    )
    candidates = [item for item in list(decomposition.get("candidates", []) or []) if isinstance(item, dict)]
    for row in candidates[:4]:
        log_event(
            "INFO",
            (
                "[no_setup_decomposition_candidate] "
                f"symbol={str(row.get('symbol') or 'none')};"
                f"setup={str(row.get('setup') or 'trend_pullback_breakout')};"
                f"score={row.get('score') if row.get('score') is not None else 'none'};"
                f"min_score={row.get('min_score') if row.get('min_score') is not None else 'none'};"
                f"gap={row.get('score_gap') if row.get('score_gap') is not None else 'none'};"
                f"primary={str(row.get('primary_real_blocker') or 'none')};"
                f"secondary={str(row.get('secondary_real_blocker') or 'none')};"
                f"bucket={str(row.get('reason_bucket') or 'UNKNOWN_NO_SETUP_REASON')};"
                f"keep_blocked={int(bool(row.get('should_keep_blocked', True)))};"
                "diagnostic_only=true"
            ),
        )
    if candidates:
        top = candidates[0]
        log_event(
            "INFO",
            (
                "[no_setup_decomposition_bucket] "
                f"symbol={str(top.get('symbol') or 'none')};"
                f"bucket={str(top.get('reason_bucket') or 'UNKNOWN_NO_SETUP_REASON')};"
                f"detail={str(top.get('reason_detail') or 'none').replace(' ', '_')};"
                f"future_study={str(top.get('suggested_future_study') or 'observe_more')};"
                "diagnostic_only=true"
            ),
        )
    log_event(
        "INFO",
        (
            "[no_setup_decomposition_recommendation] "
            f"recommendation={str(decomposition.get('recommendation') or 'insufficient_data')};"
            "allowed=diagnostic_whitelist;"
            "diagnostic_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[no_setup_decomposition_safety] "
            "should_keep_blocked=true;"
            "safe_to_change_threshold_now=false;"
            "trade_authority=false;"
            "score_authority=false;"
            "broker_authority=false;"
            "threshold_authority=false;"
            "paper_required=true;"
            "shadow_only=true"
        ),
    )


def _log_reversal_blocker_routing_audit_summary(validation_report: dict) -> None:
    audit = dict(validation_report.get("reversal_blocker_routing_audit", {}) or {})
    if not audit:
        return
    log_event(
        "INFO",
        (
            "[reversal_routing_audit_summary] "
            f"mode={str(audit.get('mode') or 'DIAGNOSTIC_ONLY').lower()};"
            f"safety_mode={str(audit.get('safety_mode') or 'SHADOW_ONLY').lower()};"
            f"top_symbol={str(audit.get('top_symbol') or 'none')};"
            f"top_setup={str(audit.get('top_setup') or 'trend_pullback_breakout')};"
            f"route_status={str(audit.get('top_route_status') or 'INSUFFICIENT_DATA_FOR_ROUTING')};"
            f"alternative_bucket={str(audit.get('top_alternative_bucket') or 'INSUFFICIENT_DATA_FOR_ROUTING')};"
            f"reversal_blocker_count={int(audit.get('reversal_blocker_count', 0) or 0)};"
            f"trend_candidates_with_reversal_blocker={int(audit.get('trend_candidates_with_reversal_blocker', 0) or 0)};"
            f"mixed_routing_count={int(audit.get('mixed_routing_count', 0) or 0)};"
            f"current_feed_clean={int(bool(audit.get('current_feed_is_clean', False)))};"
            f"fallback_scope={str(audit.get('fallback_blocker_scope') or 'UNKNOWN')};"
            f"recommendation={str(audit.get('recommendation') or 'insufficient_data')};"
            f"keep_blocked={int(bool(audit.get('should_keep_blocked', True)))};"
            f"safe_to_change_strategy_now={int(bool(audit.get('safe_to_change_strategy_now', False)))};"
            "diagnostic_only=true"
        ),
    )
    candidates = [item for item in list(audit.get("candidates", []) or []) if isinstance(item, dict)]
    for row in candidates[:4]:
        log_event(
            "INFO",
            (
                "[reversal_routing_audit_candidate] "
                f"symbol={str(row.get('symbol') or 'none')};"
                f"setup={str(row.get('setup') or 'trend_pullback_breakout')};"
                f"score={row.get('score') if row.get('score') is not None else 'none'};"
                f"min_score={row.get('min_score') if row.get('min_score') is not None else 'none'};"
                f"gap={row.get('score_gap') if row.get('score_gap') is not None else 'none'};"
                f"primary={str(row.get('primary_real_blocker') or 'none')};"
                f"secondary={str(row.get('secondary_real_blocker') or 'none')};"
                f"route_status={str(row.get('route_status') or 'UNKNOWN_ROUTING_REASON')};"
                f"alternative={str(row.get('alternative_bucket') or 'UNKNOWN_ROUTING_REASON')};"
                f"keep_blocked={int(bool(row.get('should_keep_blocked', True)))};"
                "diagnostic_only=true"
            ),
        )
    if candidates:
        top = candidates[0]
        log_event(
            "INFO",
            (
                "[reversal_routing_audit_status] "
                f"symbol={str(top.get('symbol') or 'none')};"
                f"route_status={str(top.get('route_status') or 'UNKNOWN_ROUTING_REASON')};"
                f"alternative_bucket={str(top.get('alternative_bucket') or 'UNKNOWN_ROUTING_REASON')};"
                f"detail={str(top.get('route_detail') or 'none').replace(' ', '_')};"
                "diagnostic_only=true"
            ),
        )
    log_event(
        "INFO",
        (
            "[reversal_routing_audit_recommendation] "
            f"recommendation={str(audit.get('recommendation') or 'insufficient_data')};"
            "allowed=diagnostic_whitelist;"
            "diagnostic_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[reversal_routing_audit_safety] "
            "should_keep_blocked=true;"
            "safe_to_change_strategy_now=false;"
            "trade_authority=false;"
            "score_authority=false;"
            "broker_authority=false;"
            "threshold_authority=false;"
            "paper_required=true;"
            "shadow_only=true"
        ),
    )


def _log_setup_blocker_taxonomy_audit_summary(validation_report: dict) -> None:
    audit = dict(validation_report.get("setup_blocker_taxonomy_audit", {}) or {})
    if not audit:
        return
    log_event(
        "INFO",
        (
            "[setup_blocker_taxonomy_summary] "
            f"mode={str(audit.get('mode') or 'DIAGNOSTIC_ONLY').lower()};"
            f"safety_mode={str(audit.get('safety_mode') or 'SHADOW_ONLY').lower()};"
            f"top_symbol={str(audit.get('top_symbol') or 'none')};"
            f"top_setup={str(audit.get('top_setup') or 'trend_pullback_breakout')};"
            f"official_primary_blocker={str(audit.get('official_primary_blocker') or 'none')};"
            f"normalized_primary_reason={str(audit.get('normalized_primary_reason') or 'UNKNOWN')};"
            f"taxonomy_status={str(audit.get('taxonomy_status') or 'INSUFFICIENT_DATA_FOR_TAXONOMY')};"
            f"mixed_taxonomy_count={int(audit.get('mixed_taxonomy_count', 0) or 0)};"
            f"reversal_as_context_count={int(audit.get('reversal_as_context_count', 0) or 0)};"
            f"current_feed_clean={int(bool(audit.get('current_feed_is_clean', False)))};"
            f"fallback_scope={str(audit.get('fallback_blocker_scope') or 'UNKNOWN')};"
            f"recommendation={str(audit.get('recommendation') or 'insufficient_data')};"
            f"keep_blocked={int(bool(audit.get('should_keep_blocked', True)))};"
            f"safe_to_change_strategy_now={int(bool(audit.get('safe_to_change_strategy_now', False)))};"
            f"safe_to_change_threshold_now={int(bool(audit.get('safe_to_change_threshold_now', False)))};"
            "diagnostic_only=true"
        ),
    )
    candidates = [item for item in list(audit.get("candidates", []) or []) if isinstance(item, dict)]
    for row in candidates[:4]:
        log_event(
            "INFO",
            (
                "[setup_blocker_taxonomy_candidate] "
                f"symbol={str(row.get('symbol') or 'none')};"
                f"setup={str(row.get('setup') or 'trend_pullback_breakout')};"
                f"score={row.get('score') if row.get('score') is not None else 'none'};"
                f"min_score={row.get('min_score') if row.get('min_score') is not None else 'none'};"
                f"gap={row.get('score_gap') if row.get('score_gap') is not None else 'none'};"
                f"official_primary={str(row.get('official_primary_blocker') or 'none')};"
                f"official_secondary={str(row.get('official_secondary_blocker') or 'none')};"
                f"normalized_primary={str(row.get('normalized_primary_reason') or 'UNKNOWN')};"
                f"taxonomy_status={str(row.get('taxonomy_status') or 'UNKNOWN_TAXONOMY')};"
                f"keep_blocked={int(bool(row.get('should_keep_blocked', True)))};"
                "diagnostic_only=true"
            ),
        )
    if candidates:
        top = candidates[0]
        log_event(
            "INFO",
            (
                "[setup_blocker_taxonomy_status] "
                f"symbol={str(top.get('symbol') or 'none')};"
                f"taxonomy_status={str(top.get('taxonomy_status') or 'UNKNOWN_TAXONOMY')};"
                f"normalized_primary={str(top.get('normalized_primary_reason') or 'UNKNOWN')};"
                f"normalized_secondary={str(top.get('normalized_secondary_reason') or 'UNKNOWN')};"
                f"route_status={str(top.get('route_status') or 'UNKNOWN_ROUTING_REASON')};"
                f"no_setup_bucket={str(top.get('no_setup_bucket') or 'UNKNOWN_NO_SETUP_REASON')};"
                "diagnostic_only=true"
            ),
        )
        log_event(
            "INFO",
            (
                "[setup_blocker_taxonomy_message] "
                f"symbol={str(top.get('symbol') or 'none')};"
                f"message={str(top.get('suggested_ui_message') or 'none').replace(' ', '_')};"
                "operational_language=false;"
                "diagnostic_only=true"
            ),
        )
    log_event(
        "INFO",
        (
            "[setup_blocker_taxonomy_safety] "
            "should_keep_blocked=true;"
            "safe_to_change_strategy_now=false;"
            "safe_to_change_threshold_now=false;"
            "trade_authority=false;"
            "score_authority=false;"
            "broker_authority=false;"
            "threshold_authority=false;"
            "paper_required=true;"
            "shadow_only=true"
        ),
    )


def _log_bos_confirmation_quality_audit_summary(validation_report: dict) -> None:
    audit = dict(validation_report.get("bos_confirmation_quality_audit", {}) or {})
    if not audit:
        return
    log_event(
        "INFO",
        (
            "[bos_confirmation_quality_summary] "
            f"mode={str(audit.get('mode') or 'DIAGNOSTIC_ONLY').lower()};"
            f"safety_mode={str(audit.get('safety_mode') or 'SHADOW_ONLY').lower()};"
            f"top_symbol={str(audit.get('top_symbol') or 'none')};"
            f"top_setup={str(audit.get('top_setup') or 'trend_pullback_breakout')};"
            f"bos_quality_status={str(audit.get('bos_quality_status') or 'INSUFFICIENT_DATA_FOR_BOS_QUALITY')};"
            f"bos_failure_reason={str(audit.get('bos_failure_reason') or 'insufficient_data')};"
            f"h1_bos_state={str(audit.get('h1_bos_state') or 'INSUFFICIENT_DATA')};"
            f"h4_bos_state={str(audit.get('h4_bos_state') or 'INSUFFICIENT_DATA')};"
            f"pivot_state={str(audit.get('pivot_state') or 'INSUFFICIENT_DATA')};"
            f"close_distance_to_bos_pct={audit.get('close_distance_to_bos_pct') if audit.get('close_distance_to_bos_pct') is not None else 'none'};"
            f"current_feed_clean={int(bool(audit.get('current_feed_is_clean', False)))};"
            f"fallback_scope={str(audit.get('fallback_blocker_scope') or 'UNKNOWN')};"
            f"recommendation={str(audit.get('recommendation') or 'insufficient_data')};"
            f"keep_blocked={int(bool(audit.get('should_keep_blocked', True)))};"
            f"safe_to_change_strategy_now={int(bool(audit.get('safe_to_change_strategy_now', False)))};"
            f"safe_to_change_threshold_now={int(bool(audit.get('safe_to_change_threshold_now', False)))};"
            "diagnostic_only=true"
        ),
    )
    candidates = [item for item in list(audit.get("candidates", []) or []) if isinstance(item, dict)]
    for row in candidates[:4]:
        log_event(
            "INFO",
            (
                "[bos_confirmation_quality_candidate] "
                f"symbol={str(row.get('symbol') or 'none')};"
                f"setup={str(row.get('setup') or 'trend_pullback_breakout')};"
                f"score={row.get('score') if row.get('score') is not None else 'none'};"
                f"gap={row.get('score_gap') if row.get('score_gap') is not None else 'none'};"
                f"status={str(row.get('bos_quality_status') or 'UNKNOWN_BOS_QUALITY')};"
                f"reason={str(row.get('bos_failure_reason') or 'unknown')};"
                f"h1_bos={str(row.get('h1_bos_state') or 'UNKNOWN')};"
                f"h4_bos={str(row.get('h4_bos_state') or 'UNKNOWN')};"
                f"pivot={str(row.get('pivot_state') or 'UNKNOWN')};"
                f"wick_crossed={int(bool(row.get('wick_crossed_level', False)))};"
                f"close_confirmed={int(bool(row.get('close_confirmed_beyond_level', False)))};"
                f"keep_blocked={int(bool(row.get('should_keep_blocked', True)))};"
                "diagnostic_only=true"
            ),
        )
    if candidates:
        top = candidates[0]
        log_event(
            "INFO",
            (
                "[bos_confirmation_quality_status] "
                f"symbol={str(top.get('symbol') or 'none')};"
                f"bos_quality_status={str(top.get('bos_quality_status') or 'UNKNOWN_BOS_QUALITY')};"
                f"bos_failure_reason={str(top.get('bos_failure_reason') or 'unknown')};"
                f"relationship={str(top.get('h1_h4_relationship') or 'UNKNOWN')};"
                f"multi_tf={str(top.get('multi_tf_alignment_status') or 'UNKNOWN')};"
                "diagnostic_only=true"
            ),
        )
        log_event(
            "INFO",
            (
                "[bos_confirmation_quality_message] "
                f"symbol={str(top.get('symbol') or 'none')};"
                f"message={str(top.get('suggested_ui_message') or 'none').replace(' ', '_')};"
                "operational_language=false;"
                "diagnostic_only=true"
            ),
        )
    log_event(
        "INFO",
        (
            "[bos_confirmation_quality_safety] "
            "should_keep_blocked=true;"
            "safe_to_change_strategy_now=false;"
            "safe_to_change_threshold_now=false;"
            "trade_authority=false;"
            "score_authority=false;"
            "broker_authority=false;"
            "threshold_authority=false;"
            "paper_required=true;"
            "shadow_only=true"
        ),
    )


def _log_h1_confirmation_after_h4_bos_audit_summary(validation_report: dict) -> None:
    audit = dict(validation_report.get("h1_confirmation_after_h4_bos_audit", {}) or {})
    if not audit:
        return
    log_event(
        "INFO",
        (
            "[h1_after_h4_bos_summary] "
            f"mode={str(audit.get('mode') or 'DIAGNOSTIC_ONLY').lower()};"
            f"safety_mode={str(audit.get('safety_mode') or 'SHADOW_ONLY').lower()};"
            f"top_symbol={str(audit.get('top_symbol') or 'none')};"
            f"top_setup={str(audit.get('top_setup') or 'trend_pullback_breakout')};"
            f"h4_bos_state={str(audit.get('h4_bos_state') or 'INSUFFICIENT_DATA')};"
            f"h1_bos_state={str(audit.get('h1_bos_state') or 'INSUFFICIENT_DATA')};"
            f"h1_confirmation_status={str(audit.get('h1_confirmation_status') or 'INSUFFICIENT_DATA_FOR_H1_CONFIRMATION')};"
            f"h1_failure_reason={str(audit.get('h1_failure_reason') or 'insufficient_data')};"
            f"h1_h4_alignment={str(audit.get('h1_h4_alignment') or 'INSUFFICIENT_DATA')};"
            f"current_feed_clean={int(bool(audit.get('current_feed_is_clean', False)))};"
            f"fallback_scope={str(audit.get('fallback_blocker_scope') or 'UNKNOWN')};"
            f"recommendation={str(audit.get('recommendation') or 'insufficient_data')};"
            f"keep_blocked={int(bool(audit.get('should_keep_blocked', True)))};"
            f"safe_to_change_strategy_now={int(bool(audit.get('safe_to_change_strategy_now', False)))};"
            f"safe_to_change_threshold_now={int(bool(audit.get('safe_to_change_threshold_now', False)))};"
            "diagnostic_only=true"
        ),
    )
    candidates = [item for item in list(audit.get("candidates", []) or []) if isinstance(item, dict)]
    for row in candidates[:4]:
        log_event(
            "INFO",
            (
                "[h1_after_h4_bos_candidate] "
                f"symbol={str(row.get('symbol') or 'none')};"
                f"setup={str(row.get('setup') or 'trend_pullback_breakout')};"
                f"score={row.get('score') if row.get('score') is not None else 'none'};"
                f"gap={row.get('score_gap') if row.get('score_gap') is not None else 'none'};"
                f"h4_bos={str(row.get('h4_bos_state') or 'UNKNOWN')};"
                f"h1_bos={str(row.get('h1_bos_state') or 'UNKNOWN')};"
                f"status={str(row.get('h1_confirmation_status') or 'UNKNOWN_H1_CONFIRMATION_STATUS')};"
                f"reason={str(row.get('h1_failure_reason') or 'unknown')};"
                f"alignment={str(row.get('h1_h4_alignment') or 'UNKNOWN')};"
                f"timing_risk={str(row.get('h1_entry_timing_risk') or 'UNKNOWN')};"
                f"keep_blocked={int(bool(row.get('should_keep_blocked', True)))};"
                "diagnostic_only=true"
            ),
        )
    if candidates:
        top = candidates[0]
        log_event(
            "INFO",
            (
                "[h1_after_h4_bos_status] "
                f"symbol={str(top.get('symbol') or 'none')};"
                f"h1_confirmation_status={str(top.get('h1_confirmation_status') or 'UNKNOWN_H1_CONFIRMATION_STATUS')};"
                f"h1_failure_reason={str(top.get('h1_failure_reason') or 'unknown')};"
                f"h1_data_quality={str(top.get('h1_data_quality') or 'missing')};"
                f"h1_retest_state={str(top.get('h1_retest_state') or 'UNKNOWN')};"
                "diagnostic_only=true"
            ),
        )
        log_event(
            "INFO",
            (
                "[h1_after_h4_bos_message] "
                f"symbol={str(top.get('symbol') or 'none')};"
                f"message={str(top.get('suggested_ui_message') or 'none').replace(' ', '_')};"
                "operational_language=false;"
                "diagnostic_only=true"
            ),
        )
    log_event(
        "INFO",
        (
            "[h1_after_h4_bos_safety] "
            "should_keep_blocked=true;"
            "safe_to_change_strategy_now=false;"
            "safe_to_change_threshold_now=false;"
            "trade_authority=false;"
            "score_authority=false;"
            "broker_authority=false;"
            "threshold_authority=false;"
            "paper_required=true;"
            "shadow_only=true"
        ),
    )


def _log_post_10d_calibration_plan_summary(validation_report: dict) -> None:
    plan = dict(validation_report.get("post_10d_calibration_plan", {}) or {})
    if not plan:
        return
    micro_adjustments = [item for item in list(plan.get("proposed_micro_adjustments", []) or []) if isinstance(item, dict)]
    blocked_actions = [str(item) for item in list(plan.get("blocked_actions", []) or [])]
    key_findings = [str(item) for item in list(plan.get("key_findings", []) or [])]
    approved_findings = [str(item) for item in list(plan.get("approved_findings", []) or [])]
    caution_findings = [str(item) for item in list(plan.get("caution_findings", []) or [])]
    log_event(
        "INFO",
        (
            "[post_10d_calibration_plan_summary] "
            f"mode={str(plan.get('mode') or 'PLANNING_ONLY').lower()};"
            f"safety_mode={str(plan.get('safety_mode') or 'SHADOW_ONLY').lower()};"
            f"final_classification={str(plan.get('final_classification') or 'none')};"
            f"plan_status={str(plan.get('plan_status') or 'INSUFFICIENT_DATA_FOR_PLAN')};"
            f"recommended_next_phase={str(plan.get('recommended_next_phase') or 'none').replace(' ', '_')};"
            f"should_continue_paper={int(bool(plan.get('should_continue_paper', True)))};"
            f"should_start_real_money={int(bool(plan.get('should_start_real_money', False)))};"
            f"should_change_threshold_now={int(bool(plan.get('should_change_threshold_now', False)))};"
            f"dominant_bottleneck={str(plan.get('dominant_bottleneck') or 'UNKNOWN')};"
            f"proposed_micro_adjustments_count={len(micro_adjustments)};"
            f"blocked_actions_count={len(blocked_actions)};"
            "planning_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[post_10d_calibration_plan_findings] "
            f"key_findings={'|'.join(key_findings[:6]) or 'none'};"
            f"approved={'|'.join(approved_findings[:6]) or 'none'};"
            f"caution={'|'.join(caution_findings[:6]) or 'none'};"
            "planning_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[post_10d_calibration_plan_micro_adjustments] "
            f"count={len(micro_adjustments)};"
            f"ids={'|'.join(str(item.get('id') or 'unknown') for item in micro_adjustments[:8]) or 'none'};"
            "allowed_now=false;"
            "requires_next_phase=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[post_10d_calibration_plan_blocked_actions] "
            f"count={len(blocked_actions)};"
            f"actions={'|'.join(blocked_actions[:12]) or 'none'};"
            "planning_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[post_10d_calibration_plan_safety] "
            "real_trade_allowed=false;"
            "capital_change_allowed=false;"
            "threshold_change_allowed_now=false;"
            "strategy_change_allowed_now=false;"
            "paper_required=true;"
            "shadow_only=true;"
            "diagnostic_only=true;"
            "planning_only=true"
        ),
    )


def _log_controlled_micro_adjustment_study_summary(validation_report: dict) -> None:
    try:
        for message in build_controlled_micro_adjustment_log_lines(validation_report):
            emit_worker_observability_log(message)
    except Exception as exc:
        emit_worker_observability_log(
            "[controlled_micro_adjustment_study_observability_warning] "
            f"phase=\"2.6B\";mode=STUDY_ONLY;shadow_only=true;paper_required=true;"
            f"error={str(exc or 'unknown')}"
        )


def _log_shadow_decision_simulator_summary(validation_report: dict) -> None:
    simulator = dict(validation_report.get("shadow_decision_simulator", {}) or {})
    if not simulator:
        return
    log_event(
        "INFO",
        (
            "[shadow_decision_simulator_summary] "
            f"mode={str(simulator.get('shadow_decision_mode') or 'SHADOW_ONLY').lower()};"
            f"preview_near={int(simulator.get('preview_near_approved_count', simulator.get('shadow_near_approved_count', 0)) or 0)};"
            f"raw_near={int(simulator.get('shadow_raw_near_approved_count', simulator.get('shadow_near_approved_count', 0)) or 0)};"
            f"safe={int(simulator.get('shadow_safe_near_approved_count', 0) or 0)};"
            f"marginal={int(simulator.get('shadow_marginal_near_approved_count', simulator.get('shadow_marginal_count', 0)) or 0)};"
            f"unsafe={int(simulator.get('shadow_unsafe_count', simulator.get('shadow_unsafe_rejection_count', 0)) or 0)};"
            f"would_enter={int(simulator.get('shadow_would_enter_count', 0) or 0)};"
            f"pending={int(simulator.get('shadow_pending_count', 0) or 0)};"
            f"recommendation={str(simulator.get('shadow_policy_recommendation') or 'observe_more')};"
            "shadow_only=true"
        ),
    )
    candidates = [item for item in list(simulator.get("shadow_recent_candidates", []) or []) if isinstance(item, dict)]
    for item in candidates[:3]:
        log_event(
            "INFO",
            (
                "[shadow_decision_candidate] "
                f"symbol={str(item.get('symbol') or 'none')};"
                f"strategy={str(item.get('strategy') or 'none')};"
                f"score={item.get('current_score') if item.get('current_score') is not None else 'none'};"
                f"gap={item.get('score_gap') if item.get('score_gap') is not None else 'none'};"
                f"class={str(item.get('candidate_class') or 'unknown')};"
                f"raw_near={int(bool(item.get('raw_near_approved', False)))};"
                f"would_enter={int(bool(item.get('shadow_would_enter', False)))};"
                f"why_not_safe={str(item.get('why_not_safe') or 'none')};"
                f"outcome={str(item.get('outcome_label') or 'UNKNOWN')};"
                "shadow_only=true"
            ),
        )
    outcome = dict(simulator.get("shadow_outcome_summary", {}) or {})
    log_event(
        "INFO",
        (
            "[shadow_decision_outcome_update] "
            f"pending={int(outcome.get('pending', 0) or 0)};"
            f"would_win={int(outcome.get('would_win', 0) or 0)};"
            f"would_lose={int(outcome.get('would_lose', 0) or 0)};"
            f"invalidated={int(outcome.get('invalidated', 0) or 0)};"
            "shadow_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[shadow_decision_policy_recommendation] "
            f"recommendation={str(simulator.get('shadow_policy_recommendation') or 'observe_more')};"
            f"dominant_block={str(simulator.get('shadow_dominant_block_reason') or 'none')};"
            "shadow_only=true"
        ),
    )
    current_received = int(
        simulator.get("shadow_current_cycle_received_count", simulator.get("shadow_candidates_received_count", 0)) or 0
    )
    current_new_unique = int(
        simulator.get("shadow_current_cycle_new_unique_count", simulator.get("shadow_candidates_unique_count", 0)) or 0
    )
    current_duplicate = int(
        simulator.get("shadow_current_cycle_duplicate_count", simulator.get("shadow_candidates_ignored_count", 0)) or 0
    )
    current_analyzed = int(
        simulator.get("shadow_current_cycle_analyzed_new_count", simulator.get("shadow_current_cycle_analyzed_count", 0)) or 0
    )
    current_classified = int(
        simulator.get("shadow_current_cycle_classified_new_count", simulator.get("shadow_current_cycle_classified_count", 0))
        or 0
    )
    current_unsafe = int(
        simulator.get("shadow_current_cycle_unsafe_new_count", simulator.get("shadow_current_cycle_unsafe_count", 0)) or 0
    )
    current_safe = int(simulator.get("shadow_current_cycle_safe_near_approved_count", 0) or 0)
    current_marginal = int(simulator.get("shadow_current_cycle_marginal_near_approved_count", 0) or 0)
    accumulated_unique = int(
        simulator.get("shadow_accumulated_unique_candidates_count", simulator.get("shadow_accumulated_candidates_count", 0))
        or 0
    )
    accumulated_raw = int(
        simulator.get("shadow_accumulated_raw_received_count", simulator.get("shadow_accumulated_received_count", 0)) or 0
    )
    accumulated_unsafe = int(
        simulator.get("shadow_accumulated_unsafe_unique_count", simulator.get("shadow_accumulated_unsafe_count", 0)) or 0
    )
    duplicate_ratio = float(simulator.get("shadow_duplicate_ratio", 0.0) or 0.0)
    table_scope = str(simulator.get("shadow_counts_scope") or "current_cycle_and_accumulated_recent")
    invariant_ok = not bool(simulator.get("shadow_scope_warning", simulator.get("shadow_counter_warning", False)))
    log_event(
        "INFO",
        (
            "[shadow_traceability_summary] "
            f"preview_count={int(simulator.get('preview_near_approved_count', 0) or 0)};"
            f"simulator_received_count={current_received};"
            f"analyzed_count={current_analyzed};"
            f"unsafe_count={current_unsafe};"
            f"ignored_count={current_duplicate};"
            f"dominant_exclusion_reason={str(simulator.get('shadow_dominant_block_reason') or 'none')};"
            "scope=current_cycle;"
            "shadow_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[shadow_counter_consistency_summary] "
            f"received={current_received};"
            f"unique={current_new_unique};"
            f"ignored={current_duplicate};"
            f"analyzed={current_analyzed};"
            f"classified={current_classified};"
            f"unsafe={current_unsafe};"
            f"safe={current_safe};"
            f"marginal={current_marginal};"
            f"invariant_ok={str(invariant_ok).lower()};"
            f"warning={str(simulator.get('shadow_scope_warning_reason') or simulator.get('shadow_counter_warning_reason') or 'none')};"
            "scope=current_cycle;"
            "shadow_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[shadow_counter_candidate_flow] "
            f"preview={int(simulator.get('preview_near_approved_count', 0) or 0)};"
            f"received={current_received};"
            f"unique={current_new_unique};"
            f"classified={current_classified};"
            f"ignored={current_duplicate};"
            f"accumulated_unique={accumulated_unique};"
            f"accumulated_raw={accumulated_raw};"
            f"accumulated_unsafe={accumulated_unsafe};"
            "shadow_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[shadow_counter_invariant_check] "
            f"invariant_ok={str(invariant_ok).lower()};"
            f"warning_reason={str(simulator.get('shadow_scope_warning_reason') or simulator.get('shadow_counter_warning_reason') or 'none')};"
            "non_blocking=true;"
            "shadow_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[shadow_scope_normalization_summary] "
            f"received_current={current_received};"
            f"new_unique_current={current_new_unique};"
            f"duplicate_current={current_duplicate};"
            f"analyzed_new_current={current_analyzed};"
            f"unsafe_new_current={current_unsafe};"
            f"accumulated_unique={accumulated_unique};"
            f"accumulated_raw={accumulated_raw};"
            f"duplicate_ratio={duplicate_ratio:.4f};"
            f"table_scope={table_scope};"
            f"invariant_ok={str(invariant_ok).lower()};"
            "shadow_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[shadow_scope_candidate_flow] "
            f"received_current={current_received};"
            f"new_unique_current={current_new_unique};"
            f"duplicate_current={current_duplicate};"
            f"already_analyzed_current={int(simulator.get('shadow_current_cycle_already_analyzed_count', current_duplicate) or 0)};"
            f"classified_new_current={current_classified};"
            "shadow_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[shadow_scope_duplicate_summary] "
            f"duplicate_current={current_duplicate};"
            f"duplicate_ratio={duplicate_ratio:.4f};"
            f"ignored_reason={str(simulator.get('shadow_ignored_reason') or 'none')};"
            f"health={str(simulator.get('shadow_counter_health_status') or 'healthy')};"
            "shadow_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[shadow_scope_table_scope] "
            f"table_scope={table_scope};"
            f"current_table_count={len(list(simulator.get('shadow_current_cycle_candidates', []) or []))};"
            f"accumulated_recent_table_count={len(list(simulator.get('shadow_accumulated_recent_candidates', simulator.get('shadow_recent_candidates', [])) or []))};"
            "shadow_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[shadow_scope_invariant_check] "
            f"invariant_ok={str(invariant_ok).lower()};"
            f"warning_reason={str(simulator.get('shadow_scope_warning_reason') or 'none')};"
            "non_blocking=true;"
            "shadow_only=true"
        ),
    )
    for item in candidates[:3]:
        log_event(
            "INFO",
            (
                "[shadow_traceability_candidate] "
                f"symbol={str(item.get('symbol') or 'none')};"
                f"raw_near={int(bool(item.get('raw_near_approved', False)))};"
                f"class={str(item.get('candidate_class') or 'unknown')};"
                f"safe={int(bool(item.get('safe_candidate', False)))};"
                f"why_not_safe={str(item.get('why_not_safe') or 'none')};"
                f"why_would_not_enter={str(item.get('why_would_not_enter') or 'none')};"
                "shadow_only=true"
            ),
        )
    log_event(
        "INFO",
        (
            "[shadow_traceability_exclusion_reason] "
            f"dominant={str(simulator.get('shadow_dominant_block_reason') or 'none')};"
            f"primary_blocked={int(simulator.get('shadow_primary_blocked_count', 0) or 0)};"
            f"secondary_blocked={int(simulator.get('shadow_secondary_blocked_count', 0) or 0)};"
            "shadow_only=true"
        ),
    )
    log_event(
        "INFO",
        (
            "[shadow_traceability_preview_vs_simulator] "
            f"preview_count={int(simulator.get('preview_near_approved_count', 0) or 0)};"
            f"raw_near={int(simulator.get('shadow_raw_near_approved_count', 0) or 0)};"
            f"safe={int(simulator.get('shadow_safe_near_approved_count', 0) or 0)};"
            f"marginal={int(simulator.get('shadow_marginal_near_approved_count', 0) or 0)};"
            f"unsafe={int(simulator.get('shadow_unsafe_count', 0) or 0)};"
            "shadow_only=true"
        ),
    )


def _log_phase2_fine_tune_summary(validation_report: dict) -> None:
    fine_tune = dict(validation_report.get("phase2_fine_tune", {}) or {})
    if not fine_tune:
        return
    log_event(
        "INFO",
        (
            "[phase2_fine_tune_summary] "
            f"enabled={int(bool(fine_tune.get('fine_tune_enabled', False)))};"
            f"target={str(fine_tune.get('fine_tune_target') or 'none')};"
            f"applied={int(fine_tune.get('fine_tune_applied_count', 0) or 0)};"
            f"blocked={int(fine_tune.get('fine_tune_blocked_count', 0) or 0)};"
            f"last_guard={str(fine_tune.get('fine_tune_last_guard_reason') or 'none')};"
            "paper_only=1"
        ),
    )


def _log_phase2_1_fine_tune_summary(validation_report: dict) -> None:
    fine_tune = dict(validation_report.get("phase2_1_fine_tune", {}) or {})
    if not fine_tune:
        return
    log_event(
        "INFO",
        (
            "[phase2_1_fine_tune_summary] "
            f"enabled={int(bool(fine_tune.get('phase2_1_fine_tune_enabled', False)))};"
            f"target={str(fine_tune.get('phase2_1_fine_tune_target') or 'none')};"
            f"applied={int(fine_tune.get('phase2_1_fine_tune_applied_count', 0) or 0)};"
            f"blocked={int(fine_tune.get('phase2_1_fine_tune_blocked_count', 0) or 0)};"
            f"last_guard={str(fine_tune.get('phase2_1_fine_tune_last_guard') or 'none')};"
            f"last_decision={str(fine_tune.get('phase2_1_fine_tune_last_decision') or 'none')};"
            f"score_gap={fine_tune.get('phase2_1_fine_tune_score_gap')};"
            "paper_only=1"
        ),
    )


def _log_macro_alert_summary(cycle_result: dict | None) -> None:
    payload = dict(cycle_result or {})
    alert = dict(payload.get("macro_alert", {}) or {})
    cycle_validation = dict(payload.get("validation_cycle", {}) or {})
    rejections = dict(cycle_validation.get("rejections", {}) or {})
    macro_blocks = int(rejections.get("macro_alert_guard", 0) or 0)
    log_event(
        "INFO",
        (
            "[macro_alert_summary] "
            f"active={1 if bool(alert.get('macro_alert_active', False)) else 0};"
            f"level={str(alert.get('macro_alert_level') or 'LOW').lower()};"
            f"window={str(alert.get('macro_alert_window_status') or 'INACTIVE').lower()};"
            f"currency={str(alert.get('macro_alert_currency') or 'none').lower()};"
            f"blocks_new_entries={1 if bool(alert.get('macro_alert_blocks_new_entries', False)) else 0};"
            f"penalty={float(alert.get('macro_alert_penalty', 0.0) or 0.0):.4f};"
            f"guard_blocks={macro_blocks}"
        ),
    )
    if macro_blocks > 0:
        log_event(
            "WARNING",
            (
                "[macro_alert_guard_reason] "
                f"reason={str(alert.get('macro_alert_reason') or 'macro_risk_active')};"
                f"blocked_signals={macro_blocks}"
            ),
        )


def _log_external_signal_summary(state: dict | None = None) -> None:
    payload = state or load_bot_state()
    external_signal = dict(payload.get("external_signal", {}) or {})
    status = str(external_signal.get("last_status") or "DISABLED")
    reason = str(external_signal.get("last_reason") or "External signal webhook disabled.")
    source = str(external_signal.get("last_source") or "none")
    symbol = str(external_signal.get("last_symbol") or "none")
    side = str(external_signal.get("last_side") or "none")
    score = float(external_signal.get("last_score", 0.0) or 0.0)
    log_event(
        "INFO",
        (
            "[external_signal_summary] "
            f"enabled={1 if bool(external_signal.get('enabled', False)) else 0};"
            f"status={status.lower()};source={source};symbol={symbol};side={side};"
            f"score={score:.4f};audit_only=1;trade_authority=0"
        ),
    )
    if status in {"REJECTED", "EXPIRED", "DUPLICATE", "IGNORED", "DISABLED"}:
        log_event(
            "INFO",
            (
                "[external_signal_reject_reason] "
                f"status={status.lower()};reason={reason}"
            ),
        )


def _log_cycle_summary(*, action_text: str, market_data_status: dict | None, validation_report: dict) -> None:
    metrics = dict(validation_report.get("metrics", {}) or {})
    consistency = dict(validation_report.get("consistency", {}) or {})
    status = dict(market_data_status or {})
    message = (
        "[cycle_summary] "
        f"action={str(action_text or '').lower().replace(' ', '_')};"
        f"provider={str(status.get('provider_effective') or status.get('provider') or 'unknown').lower()};"
        f"feed={str(status.get('feed_status') or 'UNKNOWN').lower()};"
        f"positions={int(metrics.get('open_positions', 0) or 0)};"
        f"approved={int(metrics.get('signals_approved', 0) or 0)};"
        f"rejected={int(metrics.get('signals_rejected', 0) or 0)};"
        f"posture={str(consistency.get('operational_posture_label') or 'Indefinida').lower()}"
    )
    log_event("INFO", message)


def _maybe_send_final_validation_email(validation_report: dict, *, current_time: datetime) -> None:
    if final_report_path_reachable(validation_report):
        return
    if int(validation_report.get("validation_day_number", 0) or 0) < 10:
        return
    if not validation_report.get("final_validation_grade"):
        return

    state = load_bot_state()
    validation_state = state.get("validation", {}) or {}
    if bool(validation_state.get("final_email_sent", False)):
        return

    try:
        result = send_final_validation_email(validation_report)
    except Exception as exc:
        log_event("ERROR", f"Falha ao enviar email final da validacao swing: {exc}")
        return

    if result.get("sent"):
        update_validation_status(
            {
                "final_email_sent": True,
                "final_email_sent_at": current_time.isoformat(),
            }
        )
        log_event("INFO", "Email final da validacao swing enviado com sucesso.")
    else:
        log_event(
            "WARNING",
            f"Email final da validacao swing nao enviado: {result.get('reason') or 'motivo nao informado'}",
        )


def _maybe_send_reporting_emails(validation_report: dict, *, current_time: datetime) -> None:
    try:
        process_report_email_delivery(validation_report=validation_report, now=current_time)
    except Exception as exc:
        log_event("ERROR", f"Falha no envio best-effort dos relatorios por email: {exc}")


def _run_daily_retention_maintenance() -> None:
    state = load_bot_state()
    if not should_run_retention_job(state):
        return

    retention_state = state.get("retention", {}) or {}
    current_time = datetime.now(timezone.utc)

    try:
        summary = run_retention_job(
            now=current_time,
            retention_days=int(retention_state.get("retention_days") or 60),
            archive_trader_orders=bool(retention_state.get("archive_trader_orders", False)),
        )
        update_retention_status(summary)
        last_summary = summary.get("last_summary", {}) or {}
        log_event(
            "INFO",
            "Retencao executada com sucesso: "
            f"relatorios={int(last_summary.get('trade_reports_archived_rows', 0) or 0)}, "
            f"logs={int(last_summary.get('bot_logs_archived_rows', 0) or 0)}, "
            f"weeklies={int(last_summary.get('weekly_reports_generated', 0) or 0)}",
        )
    except Exception as exc:
        update_retention_status(
            {
                "last_run_at": current_time.isoformat(),
                "last_error": str(exc),
                "last_error_at": current_time.isoformat(),
            }
        )
        log_event("ERROR", f"Falha na retencao automatica: {exc}")


def _refresh_production_monitor(*, cycle_success: bool, exception_message: str = "") -> dict:
    current_time = datetime.now(timezone.utc)
    state = load_bot_state()
    previous_production = state.get("production", {}) or {}
    previous_health_level = str(previous_production.get("health_level") or "healthy").lower()

    updates = {
        "enabled": PRODUCTION_MODE,
        "alert_email_enabled": ALERT_EMAIL_ENABLED,
        "last_execution_at": current_time.isoformat(),
    }

    if cycle_success:
        updates.update(
            {
                "last_success_at": current_time.isoformat(),
                "consecutive_errors": 0,
                "last_error": "",
                "last_error_at": "",
                "last_exception": "",
            }
        )
    else:
        updates.update(
            {
                "consecutive_errors": max(0, int(previous_production.get("consecutive_errors", 0) or 0)) + 1,
                "last_error": exception_message,
                "last_error_at": current_time.isoformat(),
                "last_exception": exception_message,
            }
        )

    update_production_status(updates)
    updated_state = load_bot_state()
    health_payload = evaluate_production_health(updated_state, now=current_time)
    update_production_status(health_payload)
    monitored_state = load_bot_state()

    if exception_message:
        _send_health_alert(
            monitored_state,
            health_payload,
            alert_type="critical_exception",
            current_time=current_time,
        )
        return health_payload

    if health_payload.get("health_level") in {"warning", "critical"} and health_payload.get("health_reason") not in {
        "",
        "healthy",
    }:
        _send_health_alert(
            monitored_state,
            health_payload,
            alert_type=str(health_payload.get("health_reason") or "health_warning"),
            current_time=current_time,
        )
    elif previous_health_level in {"warning", "critical"} and health_payload.get("health_level") == "healthy":
        send_recovery_email(monitored_state, health_payload=health_payload, now=current_time)

    return health_payload


def worker_loop() -> None:
    emit_startup_marker("[worker_startup_marker] WORKER_BUILD_MARKER_20260422_A")
    emit_startup_marker("[worker_startup] worker starting")
    emit_startup_marker(
        (
            "[worker_startup_marker] build info loaded "
            f"build={MARKET_DATA_BUILD_LABEL};git_sha={str(RAILWAY_GIT_COMMIT_SHA or '') or 'na'};"
            f"build_timestamp={str(BUILD_TIMESTAMP or '') or 'na'};service={str(SERVICE_NAME or '') or 'na'}"
        )
    )
    emit_startup_marker(
        (
            "[worker_startup] provider configured "
            f"primary={str(MARKET_DATA_PROVIDER or '').lower()};"
            f"fallback={str(MARKET_DATA_FALLBACK_PROVIDER or '').lower()}"
        )
    )
    load_bot_state()
    emit_startup_marker("[worker_startup] state store ready")
    emit_startup_marker("[worker_startup_marker] worker loop started")

    while True:
        try:
            current_time = datetime.now(timezone.utc)
            refresh_daily_loss_guard()
            state = load_bot_state()

            if state.get("bot_status") != "RUNNING":
                update_runtime_state(
                    last_action="Robo pausado. Aguardando ativacao.",
                    next_run_delta_seconds=PAUSED_SLEEP_SECONDS,
                )
                _refresh_production_monitor(cycle_success=True)
                _run_daily_retention_maintenance()
                validation_report = refresh_swing_validation_cycle(now=current_time)
                _maybe_send_reporting_emails(validation_report, current_time=current_time)
                _maybe_send_final_validation_email(validation_report, current_time=current_time)
                time.sleep(PAUSED_SLEEP_SECONDS)
                continue

            result = run_trader_cycle(persist_market_data=False)
            action_text = build_action_text(result)

            persist_worker_cycle_state(
                last_action=action_text,
                next_run_delta_seconds=SLEEP_SECONDS,
                worker_status="online",
                market_data_payload=result.get("cycle_result", {}).get("market_data_status"),
                runtime_started_at=WORKER_RUNTIME_STARTED_AT,
                process_role=WORKER_PROCESS_ROLE,
            )

            log_event("INFO", action_text)
            health_payload = _refresh_production_monitor(cycle_success=True)
            _log_cycle_health(health_payload)
            _run_daily_retention_maintenance()
            validation_report = refresh_swing_validation_cycle(
                cycle_result=result.get("cycle_result", {}),
                now=current_time,
            )
            _log_cycle_summary(
                action_text=action_text,
                market_data_status=result.get("cycle_result", {}).get("market_data_status"),
                validation_report=validation_report,
            )
            _log_feed_quality_summary(result.get("cycle_result", {}).get("market_data_status"))
            _log_validation_cycle(validation_report)
            _log_signal_quality_summary(validation_report)
            _log_signal_rejection_summary(validation_report, result.get("cycle_result", {}))
            _log_calibration_preview_summary(validation_report)
            _log_strategy_bottleneck_summary(validation_report)
            _log_strategy_structure_audit_summary(validation_report)
            _log_market_structure_audit_summary(validation_report)
            _log_fib_alignment_audit_summary(validation_report)
            _log_multi_timeframe_intraday_fetch_summary(validation_report)
            _log_bos_pivot_trace_audit_summary(validation_report)
            _log_multi_timeframe_swing_audit_summary(validation_report)
            _log_feed_scope_reconciliation_summary(validation_report)
            _log_strategy_decision_bridge_trace_summary(validation_report)
            _log_no_setup_eligible_decomposition_summary(validation_report)
            _log_reversal_blocker_routing_audit_summary(validation_report)
            _log_setup_blocker_taxonomy_audit_summary(validation_report)
            _log_bos_confirmation_quality_audit_summary(validation_report)
            _log_h1_confirmation_after_h4_bos_audit_summary(validation_report)
            _log_post_10d_calibration_plan_summary(validation_report)
            _log_controlled_micro_adjustment_study_summary(validation_report)
            _log_shadow_decision_simulator_summary(validation_report)
            _log_phase2_fine_tune_summary(validation_report)
            _log_phase2_1_fine_tune_summary(validation_report)
            _log_macro_alert_summary(result.get("cycle_result", {}))
            _log_external_signal_summary(load_bot_state())
            _maybe_send_reporting_emails(validation_report, current_time=current_time)
            _maybe_send_final_validation_email(validation_report, current_time=current_time)

        except Exception as exc:
            error_msg = str(exc)
            log_event("ERROR", f"Erro no worker: {error_msg}")
            mark_error(error_msg)
            health_payload = _refresh_production_monitor(cycle_success=False, exception_message=error_msg)
            _log_cycle_health(health_payload)
            _run_daily_retention_maintenance()
            validation_report = refresh_swing_validation_cycle(now=datetime.now(timezone.utc))
            _maybe_send_reporting_emails(validation_report, current_time=datetime.now(timezone.utc))
            _maybe_send_final_validation_email(validation_report, current_time=datetime.now(timezone.utc))

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    worker_loop()
