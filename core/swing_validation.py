from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from core.config import (
    BOT_LOG_COLUMNS,
    BOT_LOG_FILE,
    CALIBRATION_PREVIEW_ENABLED,
    CALIBRATION_PREVIEW_MARGIN,
    CALIBRATION_PREVIEW_MAX_EXAMPLES,
    SWING_VALIDATION_RECOMMENDED_WATCHLIST,
    TRADER_ORDERS_COLUMNS,
    TRADER_ORDERS_FILE,
    VALIDATION_DEFAULT_MAX_OPEN_POSITIONS,
    VALIDATION_INITIAL_CAPITAL_BRL,
    VALIDATION_LIVE_TRADING_ENABLED,
    VALIDATION_MODE_DISPLAY,
    VALIDATION_TRADING_MODE,
)
from core.calibration_preview import build_calibration_preview, default_calibration_preview_state
from core.market_data import build_feed_quality_snapshot
from core.state_store import load_bot_state, read_storage_table, save_bot_state
from core.signal_rejection_analysis import (
    build_feed_rejection_consistency_diagnostic,
    build_rejection_scope_counters,
    rejection_dominant_message,
    rejection_reason_label,
    update_validation_rejection_state,
)
from core.strategy_bottleneck import default_strategy_bottleneck_state, summarize_strategy_bottlenecks
from core.trader_profiles import get_trader_profile_config, normalize_trader_profile
from core.trader_reports import (
    calculate_equity_curve_metrics,
    calculate_trade_report_metrics,
    generate_trade_suggestions,
    normalize_trade_reports_frame,
    read_trade_reports,
)


SWING_VALIDATION_MODE = "swing_10d"
SWING_VALIDATION_DAYS = 10
SWING_VALIDATION_PERIOD = "2y"
SWING_VALIDATION_INTERVAL = "1d"
SWING_VALIDATION_HISTORY_LIMIT = 420

PHASE_COLETA = "Coleta e observacao"
PHASE_VALIDACAO = "Validacao dos sinais"
PHASE_AJUSTE = "Ajuste controlado"
PHASE_CONSOLIDACAO = "Consolidacao"

VERDICT_MESSAGES = {
    "APROVADO": "Aprovado para continuar em paper trading com a configuracao atual.",
    "APROVADO_COM_AJUSTES": "Aprovado com ajustes. O sistema pode continuar apos aplicar recomendacoes.",
    "REPROVADO_ESTRATEGIA": "Reprovado por baixa qualidade estrategica.",
    "REPROVADO_INSTABILIDADE": "Reprovado por instabilidade operacional.",
}


SWING_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "Reservado": {
        "holding_minutes": 8 * 24 * 60,
        "max_open_positions": VALIDATION_DEFAULT_MAX_OPEN_POSITIONS,
        "min_signal_score": 0.80,
        "min_atr_pct": 0.005,
        "max_atr_pct": 0.045,
        "trailing_stop_atr_mult": 3.0,
        "rsi_entry_min": 48.0,
        "rsi_entry_max": 58.0,
        "pullback_min": -0.012,
        "pullback_max": 0.015,
        "momentum_min": 0.001,
        "momentum_8_min": 0.01,
        "breakout_min": -0.01,
        "ma20_slope_min": 0.0,
        "ma50_slope_min": 0.0,
        "reentry_cooldown_minutes": 3 * 24 * 60,
        "min_position_age_minutes": 2 * 24 * 60,
        "take_profit_rsi": 76.0,
        "weak_momentum_exit_threshold": -0.012,
        "trend_break_buffer_pct": 0.997,
        "hard_stop_loss_pct": 0.045,
        "holding_days_target": 8,
    },
    "Equilibrado": {
        "holding_minutes": 6 * 24 * 60,
        "max_open_positions": VALIDATION_DEFAULT_MAX_OPEN_POSITIONS,
        "min_signal_score": 0.74,
        "min_atr_pct": 0.004,
        "max_atr_pct": 0.055,
        "trailing_stop_atr_mult": 3.2,
        "rsi_entry_min": 47.0,
        "rsi_entry_max": 60.0,
        "pullback_min": -0.014,
        "pullback_max": 0.02,
        "momentum_min": 0.0,
        "momentum_8_min": 0.005,
        "breakout_min": -0.015,
        "ma20_slope_min": 0.0,
        "ma50_slope_min": 0.0,
        "reentry_cooldown_minutes": 2 * 24 * 60,
        "min_position_age_minutes": 24 * 60,
        "take_profit_rsi": 77.0,
        "weak_momentum_exit_threshold": -0.01,
        "trend_break_buffer_pct": 0.996,
        "hard_stop_loss_pct": 0.05,
        "holding_days_target": 6,
    },
    "Agressivo": {
        "holding_minutes": 4 * 24 * 60,
        "max_open_positions": VALIDATION_DEFAULT_MAX_OPEN_POSITIONS,
        "min_signal_score": 0.70,
        "min_atr_pct": 0.0035,
        "max_atr_pct": 0.07,
        "trailing_stop_atr_mult": 3.4,
        "rsi_entry_min": 45.0,
        "rsi_entry_max": 62.0,
        "pullback_min": -0.018,
        "pullback_max": 0.025,
        "momentum_min": -0.001,
        "momentum_8_min": 0.0,
        "breakout_min": -0.02,
        "ma20_slope_min": -0.0005,
        "ma50_slope_min": -0.001,
        "reentry_cooldown_minutes": 24 * 60,
        "min_position_age_minutes": 12 * 60,
        "take_profit_rsi": 79.0,
        "weak_momentum_exit_threshold": -0.014,
        "trend_break_buffer_pct": 0.994,
        "hard_stop_loss_pct": 0.055,
        "holding_days_target": 4,
    },
}


def _utc_now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_dt_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(df[column], errors="coerce", utc=True)


def _safe_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def _default_signal_counters() -> dict[str, Any]:
    return {
        "signals_total": 0,
        "signals_approved": 0,
        "signals_rejected": 0,
        "entries_against_trend": 0,
        "rejections": {
            "trend_not_confirmed": 0,
            "score_below_minimum": 0,
            "reversal_not_eligible": 0,
            "volatility_out_of_range": 0,
            "no_setup_eligible": 0,
            "breakout_not_confirmed": 0,
            "confidence_too_low": 0,
            "feed_quality_blocked": 0,
            "fallback_blocked": 0,
            "provider_unknown": 0,
            "context_blocked": 0,
            "cooldown_active": 0,
            "duplicate_signal_blocked": 0,
            "position_limit_reached": 0,
            "schedule_blocked": 0,
            "daily_loss_guard": 0,
            "macro_alert_guard": 0,
        },
        "assets_observed": {},
        "assets_approved": {},
        "context_status_counts": {
            "FAVORAVEL": 0,
            "NEUTRO": 0,
            "DESFAVORAVEL": 0,
            "CRITICO": 0,
        },
        "context_blocked_signals": 0,
    }


def default_validation_state() -> dict[str, Any]:
    return {
        "validation_mode": SWING_VALIDATION_MODE,
        "validation_mode_label": VALIDATION_MODE_DISPLAY,
        "validation_started_at": "",
        "validation_day_number": 1,
        "validation_phase": PHASE_COLETA,
        "validation_status": "running",
        "final_validation_grade": "",
        "final_validation_reason": "",
        "final_validation_generated_at": "",
        "final_email_sent": False,
        "final_email_sent_at": "",
        "timeframe": SWING_VALIDATION_INTERVAL,
        "timeframe_label": "Diario (1D)",
        "period_label": SWING_VALIDATION_PERIOD,
        "trading_mode": VALIDATION_TRADING_MODE,
        "live_trading_enabled": VALIDATION_LIVE_TRADING_ENABLED,
        "paper_only": True,
        "last_evaluated_at": "",
        "last_reset_at": "",
        "last_report": {},
        "rejection_top_reason": "",
        "rejection_top_layer": "",
        "rejection_top_strategy": "",
        "rejection_reason_breakdown": {},
        "rejection_layer_breakdown": {},
        "rejection_strategy_breakdown": {},
        "rejection_has_minimum_sample": False,
        "last_cycle_rejection_summary": {},
        "current_cycle_rejection_reason": "",
        "accumulated_rejection_reason": "",
        "fallback_rejection_current_cycle_count": 0,
        "fallback_rejection_accumulated_count": 0,
        "strategy_rejection_current_cycle_count": 0,
        "strategy_rejection_accumulated_count": 0,
        "guard_rejection_current_cycle_count": 0,
        "guard_rejection_accumulated_count": 0,
        "feed_rejection_consistency": {},
        "signal_counters": _default_signal_counters(),
        "last_signal_keys": {},
        "last_rejection_event_keys": {},
    }


def _phase_for_day(day_number: int) -> str:
    if day_number <= 3:
        return PHASE_COLETA
    if day_number <= 6:
        return PHASE_VALIDACAO
    if day_number <= 8:
        return PHASE_AJUSTE
    return PHASE_CONSOLIDACAO


def _validation_day_number(started_at: Any, now: datetime | None = None) -> int:
    started = _parse_iso(started_at)
    if started is None:
        return 1
    current_time = _utc_now(now)
    return max(1, min(SWING_VALIDATION_DAYS, int((current_time - started).total_seconds() // 86400) + 1))


def _report_window(started_at: Any, now: datetime | None = None) -> tuple[datetime, datetime]:
    current_time = _utc_now(now)
    started = _parse_iso(started_at) or current_time
    end_limit = started + timedelta(days=SWING_VALIDATION_DAYS)
    return started, min(current_time, end_limit)


def apply_swing_validation_overrides(profile_name: str, profile_config: dict[str, Any]) -> dict[str, Any]:
    normalized_name = str(profile_name or profile_config.get("name") or "Equilibrado").strip() or "Equilibrado"
    overrides = SWING_PROFILE_OVERRIDES.get(normalized_name, SWING_PROFILE_OVERRIDES["Equilibrado"])
    merged = dict(profile_config)
    merged.update(overrides)
    merged["validation_mode"] = SWING_VALIDATION_MODE
    merged["period"] = SWING_VALIDATION_PERIOD
    merged["interval"] = SWING_VALIDATION_INTERVAL
    merged["history_limit"] = SWING_VALIDATION_HISTORY_LIMIT
    merged["timeframe_label"] = "Diario (1D)"
    return merged


def _normalize_reasons(raw_reasons: Any) -> list[str]:
    if isinstance(raw_reasons, list):
        items = raw_reasons
    elif isinstance(raw_reasons, str):
        items = [chunk.strip() for chunk in raw_reasons.split(",")]
    else:
        items = []
    return [str(item).strip() for item in items if str(item).strip()]


def _update_signal_counters(validation_state: dict[str, Any], cycle_result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(cycle_result, dict):
        return validation_state

    signals = cycle_result.get("signals", []) or []
    counters = dict(validation_state.get("signal_counters", {}) or {})
    if not counters:
        counters = _default_signal_counters()
    counters["rejections"] = dict(counters.get("rejections", {}) or _default_signal_counters()["rejections"])
    counters["assets_observed"] = dict(counters.get("assets_observed", {}) or {})
    counters["assets_approved"] = dict(counters.get("assets_approved", {}) or {})
    counters["context_status_counts"] = dict(
        counters.get("context_status_counts", {}) or _default_signal_counters()["context_status_counts"]
    )
    last_signal_keys = dict(validation_state.get("last_signal_keys", {}) or {})

    for signal in signals:
        asset = str(signal.get("asset") or "").upper()
        signal_key = str(signal.get("signal_key") or "").strip()
        if not asset or not signal_key:
            continue
        if last_signal_keys.get(asset) == signal_key:
            continue

        last_signal_keys[asset] = signal_key
        counters["signals_total"] = int(counters.get("signals_total", 0) or 0) + 1
        counters["assets_observed"][asset] = int(counters["assets_observed"].get(asset, 0) or 0) + 1
        context_status = str(signal.get("context_status") or "NEUTRO").upper()
        if context_status not in counters["context_status_counts"]:
            counters["context_status_counts"][context_status] = 0
        counters["context_status_counts"][context_status] = int(
            counters["context_status_counts"].get(context_status, 0) or 0
        ) + 1

        if bool(signal.get("buy")):
            counters["signals_approved"] = int(counters.get("signals_approved", 0) or 0) + 1
            counters["assets_approved"][asset] = int(counters["assets_approved"].get(asset, 0) or 0) + 1
            if str(signal.get("trend_alignment") or "").lower() == "countertrend":
                counters["entries_against_trend"] = int(counters.get("entries_against_trend", 0) or 0) + 1
            continue

        counters["signals_rejected"] = int(counters.get("signals_rejected", 0) or 0) + 1
        for reason in _normalize_reasons(signal.get("rejection_reasons")):
            if reason not in counters["rejections"]:
                counters["rejections"][reason] = 0
            counters["rejections"][reason] = int(counters["rejections"].get(reason, 0) or 0) + 1
        if "context_blocked" in _normalize_reasons(signal.get("rejection_reasons")):
            counters["context_blocked_signals"] = int(counters.get("context_blocked_signals", 0) or 0) + 1

    validation_state["signal_counters"] = counters
    validation_state["last_signal_keys"] = last_signal_keys
    validation_state = update_validation_rejection_state(validation_state, cycle_result)
    return validation_state


def _filter_timeframe(df: pd.DataFrame, column: str, start_at: datetime, end_at: datetime) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df.copy()
    working = df.copy()
    working[column] = pd.to_datetime(working[column], errors="coerce", utc=True)
    return working[(working[column] >= start_at) & (working[column] < end_at)].reset_index(drop=True)


def _asset_rankings(reports_df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if reports_df.empty:
        return [], []
    grouped = (
        reports_df.assign(realized_pnl=_safe_numeric(reports_df.get("realized_pnl")).fillna(0.0))
        .groupby("asset", dropna=False)
        .agg(
            trades=("asset", "size"),
            pnl=("realized_pnl", "sum"),
            win_rate=("realized_pnl", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
        )
        .reset_index()
    )
    best = grouped.sort_values(["pnl", "win_rate"], ascending=[False, False]).head(3)
    worst = grouped.sort_values(["pnl", "win_rate"], ascending=[True, True]).head(3)
    return best.to_dict(orient="records"), worst.to_dict(orient="records")


def _most_used_assets(orders_df: pd.DataFrame) -> list[dict[str, Any]]:
    if orders_df.empty:
        return []
    grouped = (
        orders_df.assign(side=orders_df["side"].astype("string").str.upper())
        .groupby("asset", dropna=False)
        .agg(
            events=("asset", "size"),
            buys=("side", lambda s: int((s == "BUY").sum())),
            sells=("side", lambda s: int((s == "SELL").sum())),
        )
        .reset_index()
        .sort_values(["events", "buys"], ascending=[False, False])
        .head(5)
    )
    return grouped.to_dict(orient="records")


def _early_and_late_trades(reports_df: pd.DataFrame) -> tuple[int, int]:
    if reports_df.empty:
        return 0, 0
    durations = _safe_numeric(reports_df.get("duration_minutes"))
    holding_limits = _safe_numeric(reports_df.get("holding_minutes_limit"))
    early_threshold = holding_limits.fillna(24 * 60).apply(lambda value: max(12 * 60, float(value) * 0.25))
    late_threshold = holding_limits.fillna(4 * 24 * 60).apply(lambda value: max(2 * 24 * 60, float(value) * 1.15))
    early_closed = int(((durations.notna()) & (durations <= early_threshold)).sum())
    late_held = int(((durations.notna()) & (durations >= late_threshold)).sum())
    return early_closed, late_held


def _parse_cycle_health_logs(logs_df: pd.DataFrame) -> pd.DataFrame:
    if logs_df.empty or "message" not in logs_df.columns:
        return pd.DataFrame(columns=["timestamp", "fallback", "health_level", "feed_status"])

    rows: list[dict[str, Any]] = []
    for row in logs_df.to_dict(orient="records"):
        message = str(row.get("message") or "")
        if not message.startswith("[cycle_health] "):
            continue
        payload: dict[str, Any] = {"timestamp": row.get("timestamp")}
        for chunk in message[len("[cycle_health] ") :].split(";"):
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            payload[key.strip()] = value.strip()
        rows.append(payload)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "fallback", "health_level", "feed_status"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame


def _fallback_pct(logs_df: pd.DataFrame) -> float | None:
    cycle_health = _parse_cycle_health_logs(logs_df)
    if cycle_health.empty:
        return None
    total_cycles = int(len(cycle_health))
    if total_cycles <= 0:
        return None
    fallback_cycles = int(cycle_health["fallback"].astype("string").fillna("0").isin(["1", "true", "True"]).sum())
    return round((fallback_cycles / total_cycles) * 100.0, 2)


def _comparison_payload(reports_df: pd.DataFrame, started_at: datetime, now: datetime) -> dict[str, Any]:
    if reports_df.empty:
        return {}
    split_at = started_at + timedelta(days=6)
    before_df = reports_df[reports_df["closed_at"] < split_at].copy()
    after_df = reports_df[reports_df["closed_at"] >= split_at].copy()
    if before_df.empty and after_df.empty:
        return {}

    before_metrics = calculate_trade_report_metrics(before_df)
    after_metrics = calculate_trade_report_metrics(after_df)
    improvement = []
    if before_metrics["payoff"] is not None and after_metrics["payoff"] is not None:
        if float(after_metrics["payoff"]) > float(before_metrics["payoff"]):
            improvement.append("Payoff melhorou na fase final.")
        elif float(after_metrics["payoff"]) < float(before_metrics["payoff"]):
            improvement.append("Payoff piorou apos a fase de ajustes.")
    if float(after_metrics["win_rate"] or 0.0) > float(before_metrics["win_rate"] or 0.0):
        improvement.append("Win rate melhorou na fase final.")
    elif float(after_metrics["win_rate"] or 0.0) < float(before_metrics["win_rate"] or 0.0):
        improvement.append("Win rate piorou na fase final.")

    return {
        "before": {
            "trades": int(before_metrics["total_trades"] or 0),
            "pnl": float(before_metrics["total_pnl"] or 0.0),
            "win_rate": float(before_metrics["win_rate"] or 0.0),
            "payoff": before_metrics["payoff"],
        },
        "after": {
            "trades": int(after_metrics["total_trades"] or 0),
            "pnl": float(after_metrics["total_pnl"] or 0.0),
            "win_rate": float(after_metrics["win_rate"] or 0.0),
            "payoff": after_metrics["payoff"],
        },
        "notes": improvement,
    }


def _phase_conclusion(day_number: int, report: dict[str, Any]) -> str:
    metrics = report.get("metrics", {}) or {}
    if day_number <= 3:
        if float(metrics.get("fallback_cycle_pct") or 0.0) >= 30.0 or int(metrics.get("operational_errors") or 0) >= 3:
            return "instavel"
        if int(metrics.get("signals_approved") or 0) <= 1:
            return "poucos sinais"
        if int(metrics.get("signals_approved") or 0) >= 12:
            return "sinais excessivos"
        return "base inicial coletada"
    if day_number <= 6:
        if int(metrics.get("against_trend_entries") or 0) > 0:
            return "filtro fraco"
        if int(metrics.get("signals_approved") or 0) >= 18:
            return "excesso de sinais"
        return "entradas coerentes"
    if day_number <= 8:
        notes = (report.get("before_after_comparison", {}) or {}).get("notes", []) or []
        return "melhora observada" if any("melhorou" in str(note).lower() for note in notes) else "ajuste controlado"
    return "consolidacao em andamento"


def _build_errors(report: dict[str, Any]) -> list[str]:
    metrics = report.get("metrics", {}) or {}
    errors: list[str] = []
    if float(metrics.get("fallback_cycle_pct") or 0.0) > 30.0:
        errors.append("Feed instavel por fallback acima de 30% dos ciclos.")
    if int(metrics.get("operational_errors") or 0) >= 3:
        errors.append("Ocorreram muitos erros operacionais no periodo.")
    if int(metrics.get("against_trend_entries") or 0) > 0:
        errors.append("Foram detectadas entradas contra a tendencia dominante.")
    if int(metrics.get("signals_approved") or 0) >= 20:
        errors.append("O robo aprovou sinais em excesso para uma rotina swing profissional.")
    if int(metrics.get("early_closed_trades") or 0) >= 2:
        errors.append("Ha trades encerrados cedo demais, indicando timing de saida fraco.")
    if int(metrics.get("late_held_trades") or 0) >= 2:
        errors.append("Ha posicoes mantidas tempo demais, travando capital em papel.")
    if int(metrics.get("context_blocked_signals") or 0) >= 3:
        errors.append("Parte dos sinais de cripto foi bloqueada por contexto desfavoravel ou critico.")
    if (report.get("performance", {}) or {}).get("payoff") is not None and float(report["performance"]["payoff"]) < 1.0:
        errors.append("Payoff abaixo de 1.0, com risco maior que o retorno medio.")
    return errors


def _build_successes(report: dict[str, Any]) -> list[str]:
    metrics = report.get("metrics", {}) or {}
    performance = report.get("performance", {}) or {}
    successes: list[str] = []
    if float(metrics.get("fallback_cycle_pct") or 0.0) <= 20.0:
        successes.append("Estabilidade operacional consistente, com fallback sob controle.")
    if float(performance.get("pnl_total") or 0.0) > 0:
        successes.append("PnL total positivo durante o ciclo de validacao.")
    if performance.get("payoff") is not None and float(performance.get("payoff") or 0.0) >= 1.2:
        successes.append("Boa relacao risco/retorno, com payoff igual ou superior a 1.2.")
    if float(performance.get("win_rate") or 0.0) >= 0.5:
        successes.append("Win rate coerente para a proposta de swing mais seletiva.")
    if int(metrics.get("against_trend_entries") or 0) == 0:
        successes.append("Nao foram registradas entradas contra a tendencia dominante.")
    if int(metrics.get("context_blocked_signals") or 0) > 0:
        successes.append("O filtro de contexto cripto evitou entradas fracas em ambiente menos favoravel.")
    if report.get("best_assets"):
        successes.append("Ja existem ativos com comportamento mais confiavel dentro do ciclo.")
    return successes


def _build_suggestions(report: dict[str, Any]) -> list[dict[str, Any]]:
    suggestions = list(generate_trade_suggestions(report.get("reports_df", pd.DataFrame())))
    metrics = report.get("metrics", {}) or {}
    if int(metrics.get("signals_approved") or 0) >= 15:
        suggestions.append(
            {
                "severity": "warning",
                "message": "Aumentar min_signal_score para reduzir sinais demais no modo swing.",
                "proposed_adjustments": {"min_signal_score_delta": 0.03},
            }
        )
    if int(metrics.get("against_trend_entries") or 0) > 0:
        suggestions.append(
            {
                "severity": "warning",
                "message": "Reforcar filtro de tendencia para evitar entradas contra a estrutura principal.",
                "proposed_adjustments": {"ma20_slope_min_delta": 0.002, "ma50_slope_min_delta": 0.002},
            }
        )
    if float(metrics.get("fallback_cycle_pct") or 0.0) > 30.0:
        suggestions.append(
            {
                "severity": "warning",
                "message": "Instabilidade de feed relevante; manter o robo em paper e revisar o provider antes de escalar.",
                "proposed_adjustments": {},
            }
        )
    if int(metrics.get("context_blocked_signals") or 0) >= 2:
        suggestions.append(
            {
                "severity": "info",
                "message": "O contexto cripto bloqueou sinais fracos; manter o filtro contextual ativo nesta fase.",
                "proposed_adjustments": {},
            }
        )
    if int(metrics.get("late_held_trades") or 0) >= 2:
        suggestions.append(
            {
                "severity": "info",
                "message": "Revisar max_open_positions e trailing stop para destravar capital preso por tempo demais.",
                "proposed_adjustments": {"max_open_positions_delta": -1},
            }
        )

    unique: list[dict[str, Any]] = []
    seen_messages: set[str] = set()
    for suggestion in suggestions:
        message = str(suggestion.get("message") or "").strip()
        if not message or message in seen_messages:
            continue
        seen_messages.add(message)
        unique.append(suggestion)
    return unique


def _build_context_impact_estimate(report: dict[str, Any]) -> str:
    metrics = report.get("metrics", {}) or {}
    current_context = report.get("current_market_context", {}) or {}
    status = str(current_context.get("market_context_status") or "NEUTRO").upper()
    blocked_signals = int(metrics.get("context_blocked_signals", 0) or 0)
    status_counts = dict(metrics.get("context_status_counts", {}) or {})
    adverse_cycles = int(status_counts.get("DESFAVORAVEL", 0) or 0) + int(status_counts.get("CRITICO", 0) or 0)

    if blocked_signals >= 3:
        return (
            f"O filtro de contexto barrou {blocked_signals} sinais cripto fracos em ambiente "
            f"{status}, reduzindo a chance de entradas ruins."
        )
    if blocked_signals > 0:
        return f"O filtro de contexto conteve {blocked_signals} sinais cripto de qualidade insuficiente."
    if adverse_cycles > 0:
        return (
            f"O contexto passou {adverse_cycles} leituras em modo desfavoravel/critico, "
            "mas sem bloquear sinais fortes nesta janela."
        )
    if status == "FAVORAVEL":
        return "O contexto cripto ficou favoravel na maior parte do periodo, sem necessidade de endurecer sinais."
    return "O contexto permaneceu neutro ou pouco relevante, sem impacto material nas entradas desta janela."


def _normalized_watchlist(values: Any) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        asset = str(value).strip().upper()
        if not asset or asset in seen:
            continue
        seen.add(asset)
        normalized.append(asset)
    return normalized


def _load_runtime_equity_curve_metrics() -> dict[str, float | None]:
    try:
        from engines.quant_bridge import read_paper_equity

        return calculate_equity_curve_metrics(read_paper_equity(limit=1000))
    except Exception:
        return {
            "current_equity": None,
            "peak_equity": None,
            "max_drawdown_brl": None,
            "max_drawdown_pct": None,
        }


def _sample_quality_payload(total_trades: int) -> dict[str, str]:
    if total_trades < 3:
        return {
            "label": "Baixa",
            "message": "A amostra ainda e pequena demais para concluir qualidade estrategica com confianca.",
        }
    if total_trades < 8:
        return {
            "label": "Parcial",
            "message": "A amostra ja permite leitura inicial, mas ainda pede mais ciclos antes de ajuste fino.",
        }
    return {
        "label": "Aceitavel",
        "message": "A amostra ja sustenta uma leitura operacional mais confiavel desta fase.",
    }


def _operational_posture_payload(
    *,
    approval_rate: float | None,
    signals_total: int,
    effective_profile: dict[str, Any],
) -> dict[str, str]:
    score_floor = float(effective_profile.get("min_signal_score", 0.0) or 0.0)
    max_open_positions = int(effective_profile.get("max_open_positions", VALIDATION_DEFAULT_MAX_OPEN_POSITIONS) or 0)

    if approval_rate is not None and signals_total >= 10:
        if approval_rate <= 0.08 and score_floor >= 0.74:
            return {
                "label": "Conservador",
                "message": "O filtro esta bem seletivo para esta fase, com poucas aprovacoes por janela.",
            }
        if approval_rate >= 0.35 or score_floor <= 0.66 or max_open_positions >= 4:
            return {
                "label": "Permissivo",
                "message": "O conjunto atual tende a aceitar sinais demais para uma validacao swing disciplinada.",
            }

    return {
        "label": "Equilibrado",
        "message": "O robo esta seletivo sem aparentar permissividade excessiva para esta fase.",
    }


def _signal_quality_payload(
    *,
    approval_rate: float | None,
    signals_total: int,
    sample_quality: dict[str, str],
    posture: dict[str, str],
    watchlist_phase_aligned: bool,
    fallback_cycle_pct: float,
) -> dict[str, str]:
    if signals_total <= 0 or sample_quality.get("label") == "Inicial":
        return {
            "label": "Baixa",
            "message": "Ainda ha pouca amostra para concluir a qualidade dos sinais desta fase.",
        }
    if not watchlist_phase_aligned or fallback_cycle_pct >= 20.0:
        return {
            "label": "Baixa",
            "message": "A leitura de sinal ainda esta contaminada por desalinhamento da watchlist ou qualidade insuficiente do feed.",
        }
    if posture.get("label") == "Conservador" and approval_rate is not None and approval_rate <= 0.12:
        return {
            "label": "Moderada",
            "message": "O filtro parece seletivo demais para a amostra atual; vale observar mais ciclos antes de ajustar.",
        }
    if sample_quality.get("label") == "Aceitavel" and posture.get("label") == "Equilibrado":
        return {
            "label": "Boa",
            "message": "Os sinais ja sustentam uma leitura operacional mais confiavel desta fase.",
        }
    return {
        "label": "Moderada",
        "message": "Os sinais ja permitem leitura inicial, mas ainda pedem mais amostra antes de ajuste fino.",
    }


def _validation_reading_payload(
    *,
    sample_quality: dict[str, str],
    posture: dict[str, str],
    watchlist_alignment: dict[str, Any],
) -> dict[str, Any]:
    messages: list[str] = []

    if sample_quality.get("label") == "Inicial":
        messages.append("Ainda ha pouca amostra para leitura estrategica desta fase.")
    elif sample_quality.get("label") == "Parcial":
        messages.append("A amostra ja permite leitura inicial, mas ainda nao sustenta ajuste fino.")
    else:
        messages.append("Ja existe base minima para avaliar pequenos ajustes sem precipitar a estrategia.")

    if posture.get("label") == "Conservador":
        messages.append("O filtro parece seletivo demais para esta janela.")
    elif posture.get("label") == "Permissivo":
        messages.append("O filtro esta permissivo demais para uma validacao swing disciplinada.")
    else:
        messages.append("O filtro atual segue equilibrado para a fase.")

    if watchlist_alignment.get("phase_aligned"):
        messages.append("A watchlist segue coerente com a fase cripto-only.")
    else:
        messages.append("A watchlist ainda nao esta totalmente coerente com a fase atual.")

    fine_tuning_ready = bool(
        sample_quality.get("label") == "Aceitavel"
        and posture.get("label") == "Equilibrado"
        and bool(watchlist_alignment.get("phase_aligned"))
    )
    if fine_tuning_ready:
        messages.append("Ja existe base minima para observar ajustes finos, sem mudar a estrategia base agora.")
    else:
        messages.append("Ainda nao ha base minima suficiente para ajuste fino com seguranca.")

    return {
        "message": " ".join(messages),
        "fine_tuning_ready": fine_tuning_ready,
    }


def _rejection_quality_payload(validation_state: dict[str, Any]) -> dict[str, Any]:
    reason_breakdown = dict(validation_state.get("rejection_reason_breakdown", {}) or {})
    layer_breakdown = dict(validation_state.get("rejection_layer_breakdown", {}) or {})
    strategy_breakdown = dict(validation_state.get("rejection_strategy_breakdown", {}) or {})
    total_events = int(sum(int(value or 0) for value in reason_breakdown.values()))
    top_reason = str(validation_state.get("rejection_top_reason") or "")
    top_layer = str(validation_state.get("rejection_top_layer") or "")
    top_strategy = str(validation_state.get("rejection_top_strategy") or "")
    scope_counts = build_rejection_scope_counters(
        dict(validation_state.get("last_cycle_rejection_summary", {}) or {}),
        {
            "total_rejection_events": total_events,
            "reason_breakdown": reason_breakdown,
            "layer_breakdown": layer_breakdown,
            "strategy_breakdown": strategy_breakdown,
            "top_reason": top_reason,
            "top_layer": top_layer,
            "top_strategy": top_strategy,
        },
    )

    top_reasons: list[dict[str, Any]] = []
    if total_events > 0:
        for reason_code, count in sorted(reason_breakdown.items(), key=lambda item: int(item[1] or 0), reverse=True)[:5]:
            pct = (int(count or 0) / total_events) if total_events > 0 else 0.0
            top_reasons.append(
                {
                    "reason_code": str(reason_code),
                    "human_reason": rejection_reason_label(str(reason_code)),
                    "count": int(count or 0),
                    "pct": round(pct, 4),
                }
            )

    return {
        "total_rejection_events": total_events,
        "top_reason": top_reason,
        "top_reason_label": rejection_reason_label(top_reason),
        "top_layer": top_layer,
        "top_layer_message": rejection_dominant_message(top_layer) if top_layer else "",
        "top_strategy": top_strategy,
        "reason_breakdown": reason_breakdown,
        "layer_breakdown": layer_breakdown,
        "strategy_breakdown": strategy_breakdown,
        "top_reasons": top_reasons,
        "has_minimum_sample": bool(validation_state.get("rejection_has_minimum_sample", False)),
        **scope_counts,
    }


def _watchlist_phase_alignment_payload(state: dict[str, Any]) -> dict[str, Any]:
    current_watchlist = _normalized_watchlist((state.get("trader", {}) or {}).get("watchlist", []))
    recommended_watchlist = _normalized_watchlist(SWING_VALIDATION_RECOMMENDED_WATCHLIST)
    non_recommended_assets = [asset for asset in current_watchlist if asset not in recommended_watchlist]
    missing_recommended_assets = [asset for asset in recommended_watchlist if asset not in current_watchlist]
    open_positions = [
        str(position.get("asset") or "").strip().upper()
        for position in (state.get("positions", []) or [])
        if str(position.get("module") or "").upper() == "TRADER" and str(position.get("status") or "").upper() == "OPEN"
    ]
    managed_assets_outside_watchlist = [asset for asset in open_positions if asset and asset not in current_watchlist]
    phase_aligned = not non_recommended_assets and bool(current_watchlist)

    if phase_aligned and not managed_assets_outside_watchlist:
        message = "A watchlist ativa segue coerente com a fase cripto-only atual."
    elif non_recommended_assets:
        message = (
            "Ha ativos fora da watchlist recomendada da fase: "
            f"{', '.join(non_recommended_assets)}."
        )
    else:
        message = (
            "Existem posicoes abertas fora da watchlist ativa: "
            f"{', '.join(managed_assets_outside_watchlist)}."
        )

    return {
        "phase_aligned": phase_aligned,
        "current_watchlist": current_watchlist,
        "recommended_watchlist": recommended_watchlist,
        "non_recommended_assets": non_recommended_assets,
        "missing_recommended_assets": missing_recommended_assets,
        "managed_assets_outside_watchlist": managed_assets_outside_watchlist,
        "message": message,
    }


def _build_operational_consistency(
    report: dict[str, Any],
    state: dict[str, Any],
    equity_metrics: dict[str, float | None],
) -> dict[str, Any]:
    metrics = dict(report.get("metrics", {}) or {})
    trader_state = dict(state.get("trader", {}) or {})
    current_profile = normalize_trader_profile(trader_state.get("profile"))
    base_profile = get_trader_profile_config(
        current_profile,
        base_ticket_value=float(trader_state.get("ticket_value", 100.0) or 100.0),
        base_holding_minutes=int(trader_state.get("holding_minutes", 60) or 60),
        base_max_open_positions=int(trader_state.get("max_open_positions", VALIDATION_DEFAULT_MAX_OPEN_POSITIONS) or VALIDATION_DEFAULT_MAX_OPEN_POSITIONS),
    )
    effective_profile = apply_swing_validation_overrides(current_profile, base_profile)

    signals_total = int(metrics.get("signals_total", 0) or 0)
    signals_approved = int(metrics.get("signals_approved", 0) or 0)
    approval_rate = (signals_approved / signals_total) if signals_total > 0 else None
    sample_quality = _sample_quality_payload(int(metrics.get("trades_closed", 0) or 0))
    rejection_quality = _rejection_quality_payload(dict(state.get("validation", {}) or {}))
    posture = _operational_posture_payload(
        approval_rate=approval_rate,
        signals_total=signals_total,
        effective_profile=effective_profile,
    )
    watchlist_alignment = _watchlist_phase_alignment_payload(state)
    signal_quality = _signal_quality_payload(
        approval_rate=approval_rate,
        signals_total=signals_total,
        sample_quality=sample_quality,
        posture=posture,
        watchlist_phase_aligned=bool(watchlist_alignment["phase_aligned"]),
        fallback_cycle_pct=float(metrics.get("fallback_cycle_pct", 0.0) or 0.0),
    )
    validation_reading = _validation_reading_payload(
        sample_quality=sample_quality,
        posture=posture,
        watchlist_alignment=watchlist_alignment,
    )
    calibration_preview = dict(state.get("calibration_preview", {}) or default_calibration_preview_state())
    strategy_bottleneck = dict(state.get("strategy_bottleneck", {}) or default_strategy_bottleneck_state())
    runtime_capital = float(state.get("wallet_value", 0.0) or 0.0)
    capital_phase_aligned = abs(runtime_capital - float(VALIDATION_INITIAL_CAPITAL_BRL)) < 0.01

    capital_message = (
        "Capital do runtime alinhado ao capital-base recomendado da fase."
        if capital_phase_aligned
        else (
            "O capital atual do runtime nao bate com o capital-base recomendado da fase "
            f"({runtime_capital:,.2f} vs {float(VALIDATION_INITIAL_CAPITAL_BRL):,.2f})."
        )
    )

    return {
        "sample_quality_label": sample_quality["label"],
        "sample_quality_message": sample_quality["message"],
        "signal_approval_rate": None if approval_rate is None else round(float(approval_rate), 4),
        "operational_posture_label": posture["label"],
        "operational_posture_message": posture["message"],
        "signal_quality_label": signal_quality["label"],
        "signal_quality_message": signal_quality["message"],
        "rejection_top_reason": rejection_quality["top_reason"],
        "rejection_top_reason_label": rejection_quality["top_reason_label"],
        "rejection_top_layer": rejection_quality["top_layer"],
        "rejection_top_layer_message": rejection_quality["top_layer_message"],
        "rejection_top_strategy": rejection_quality["top_strategy"],
        "rejection_has_minimum_sample": rejection_quality["has_minimum_sample"],
        "watchlist_phase_aligned": bool(watchlist_alignment["phase_aligned"]),
        "watchlist_message": watchlist_alignment["message"],
        "current_watchlist": watchlist_alignment["current_watchlist"],
        "recommended_watchlist": watchlist_alignment["recommended_watchlist"],
        "non_recommended_assets": watchlist_alignment["non_recommended_assets"],
        "managed_assets_outside_watchlist": watchlist_alignment["managed_assets_outside_watchlist"],
        "capital_phase_aligned": capital_phase_aligned,
        "capital_message": capital_message,
        "effective_profile_name": current_profile,
        "effective_ticket_value": round(float(effective_profile.get("ticket_value", 0.0) or 0.0), 2),
        "effective_holding_minutes": int(effective_profile.get("holding_minutes", 0) or 0),
        "effective_max_open_positions": int(
            effective_profile.get("max_open_positions", VALIDATION_DEFAULT_MAX_OPEN_POSITIONS) or VALIDATION_DEFAULT_MAX_OPEN_POSITIONS
        ),
        "effective_min_signal_score": round(float(effective_profile.get("min_signal_score", 0.0) or 0.0), 4),
        "validation_reading_message": validation_reading["message"],
        "fine_tuning_ready": bool(validation_reading["fine_tuning_ready"]),
        "max_drawdown_brl": equity_metrics.get("max_drawdown_brl"),
        "max_drawdown_pct": equity_metrics.get("max_drawdown_pct"),
        "calibration_preview_mode": calibration_preview.get("mode", "PREVIEW_ONLY"),
        "calibration_preview_near_approved_count": int(calibration_preview.get("near_approved_count", 0) or 0),
        "calibration_preview_near_approved_rate": float(calibration_preview.get("near_approved_rate", 0.0) or 0.0),
        "calibration_preview_recommendation": str(calibration_preview.get("recommendation") or "observe_more"),
        "strategy_bottleneck_mode": strategy_bottleneck.get("mode", "DIAGNOSTIC_ONLY"),
        "strategy_bottleneck_dominant": str(strategy_bottleneck.get("dominant_bottleneck") or ""),
        "strategy_bottleneck_recommendation": str(strategy_bottleneck.get("recommendation") or "observe_more"),
    }


def _final_verdict(report: dict[str, Any], state: dict[str, Any]) -> tuple[str, str]:
    metrics = report.get("metrics", {}) or {}
    performance = report.get("performance", {}) or {}
    production = state.get("production", {}) or {}

    fallback_pct = float(metrics.get("fallback_cycle_pct") or 0.0)
    operational_errors = int(metrics.get("operational_errors") or 0)
    consecutive_errors = int(production.get("consecutive_errors") or 0)
    against_trend_entries = int(metrics.get("against_trend_entries") or 0)
    payoff = performance.get("payoff")
    pnl_total = float(performance.get("pnl_total") or 0.0)
    win_rate = float(performance.get("win_rate") or 0.0)

    if fallback_pct > 30.0 or operational_errors >= 5 or consecutive_errors >= 3:
        return "REPROVADO_INSTABILIDADE", (
            "O ciclo terminou com instabilidade operacional relevante, especialmente em feed/falhas consecutivas."
        )
    if (payoff is not None and float(payoff) < 1.0) or against_trend_entries >= 2 or (pnl_total < 0 and win_rate < 0.4):
        return "REPROVADO_ESTRATEGIA", (
            "A estrategia swing nao entregou qualidade suficiente nas entradas e saidas durante o ciclo."
        )
    if pnl_total > 0 and (payoff is None or float(payoff) >= 1.2) and fallback_pct <= 20.0 and against_trend_entries == 0:
        return "APROVADO", "O robo se manteve estavel e entregou consistencia suficiente para continuar em paper."
    return "APROVADO_COM_AJUSTES", "O ciclo foi aceitavel, mas ainda exige ajustes controlados antes de escalar."


def build_swing_validation_report(state: dict | None = None, now: datetime | None = None) -> dict[str, Any]:
    payload = state or load_bot_state()
    validation_state = dict(payload.get("validation", {}) or {})
    current_time = _utc_now(now)

    started_at, window_end = _report_window(validation_state.get("validation_started_at"), current_time)
    day_number = _validation_day_number(started_at.isoformat(), current_time)
    phase = _phase_for_day(day_number)

    reports_df = read_trade_reports()
    reports_df = normalize_trade_reports_frame(reports_df)
    reports_df = _filter_timeframe(reports_df, "closed_at", started_at, window_end)

    orders_df = read_storage_table(TRADER_ORDERS_FILE, columns=TRADER_ORDERS_COLUMNS)
    orders_df = _filter_timeframe(orders_df, "timestamp", started_at, window_end)

    logs_df = read_storage_table(BOT_LOG_FILE, columns=BOT_LOG_COLUMNS)
    logs_df = _filter_timeframe(logs_df, "timestamp", started_at, window_end)

    signal_counters = dict(validation_state.get("signal_counters", {}) or _default_signal_counters())
    signal_rejections = dict(signal_counters.get("rejections", {}) or {})
    rejection_summary = _rejection_quality_payload(validation_state)
    feed_quality = build_feed_quality_snapshot(dict(payload.get("market_data", {}) or {}))
    feed_rejection_consistency = build_feed_rejection_consistency_diagnostic(
        feed_quality=feed_quality,
        current_cycle_summary=dict(validation_state.get("last_cycle_rejection_summary", {}) or {}),
        accumulated_summary=rejection_summary,
    )

    performance_metrics = calculate_trade_report_metrics(reports_df)
    early_closed, late_held = _early_and_late_trades(reports_df)
    fallback_cycle_pct = _fallback_pct(logs_df)
    best_assets, worst_assets = _asset_rankings(reports_df)
    most_used_assets = _most_used_assets(orders_df)
    operational_errors = int(
        (logs_df["level"].astype("string").str.upper() == "ERROR").sum()
    ) if not logs_df.empty and "level" in logs_df.columns else 0

    metrics = {
        "trades_opened": int((orders_df["side"].astype("string").str.upper() == "BUY").sum()) if not orders_df.empty else 0,
        "trades_closed": int(performance_metrics["total_trades"] or 0),
        "signals_total": int(signal_counters.get("signals_total", 0) or 0),
        "signals_approved": int(signal_counters.get("signals_approved", 0) or 0),
        "signals_rejected": int(signal_counters.get("signals_rejected", 0) or 0),
        "against_trend_entries": int(signal_counters.get("entries_against_trend", 0) or 0),
        "signal_rejections": signal_rejections,
        "rejection_top_reason": rejection_summary["top_reason"],
        "rejection_top_layer": rejection_summary["top_layer"],
        "rejection_top_strategy": rejection_summary["top_strategy"],
        "rejection_reason_breakdown": dict(rejection_summary["reason_breakdown"]),
        "rejection_layer_breakdown": dict(rejection_summary["layer_breakdown"]),
        "rejection_strategy_breakdown": dict(rejection_summary["strategy_breakdown"]),
        "rejection_has_minimum_sample": bool(rejection_summary["has_minimum_sample"]),
        "current_cycle_rejection_reason": rejection_summary["current_cycle_rejection_reason"],
        "accumulated_rejection_reason": rejection_summary["accumulated_rejection_reason"],
        "fallback_rejection_current_cycle_count": int(rejection_summary["fallback_rejection_current_cycle_count"]),
        "fallback_rejection_accumulated_count": int(rejection_summary["fallback_rejection_accumulated_count"]),
        "strategy_rejection_current_cycle_count": int(rejection_summary["strategy_rejection_current_cycle_count"]),
        "strategy_rejection_accumulated_count": int(rejection_summary["strategy_rejection_accumulated_count"]),
        "guard_rejection_current_cycle_count": int(rejection_summary["guard_rejection_current_cycle_count"]),
        "guard_rejection_accumulated_count": int(rejection_summary["guard_rejection_accumulated_count"]),
        "feed_rejection_consistency": dict(feed_rejection_consistency),
        "context_status_counts": dict(signal_counters.get("context_status_counts", {}) or {}),
        "context_blocked_signals": int(signal_counters.get("context_blocked_signals", 0) or 0),
        "open_positions": len(payload.get("positions", []) or []),
        "avg_duration_minutes": performance_metrics["avg_duration"],
        "fallback_cycle_pct": fallback_cycle_pct,
        "operational_errors": operational_errors,
        "worker_status": str(payload.get("worker_status") or "offline"),
        "heartbeat_age_seconds": payload.get("production", {}).get("heartbeat_age_seconds"),
        "consecutive_errors": int(payload.get("production", {}).get("consecutive_errors", 0) or 0),
        "early_closed_trades": early_closed,
        "late_held_trades": late_held,
    }

    performance = {
        "pnl_total": float(performance_metrics["total_pnl"] or 0.0),
        "win_rate": float(performance_metrics["win_rate"] or 0.0),
        "payoff": performance_metrics["payoff"],
        "avg_win": performance_metrics["avg_win"],
        "avg_loss": performance_metrics["avg_loss"],
        "best_trade": performance_metrics["best_trade"],
        "worst_trade": performance_metrics["worst_trade"],
    }

    equity_metrics = _load_runtime_equity_curve_metrics()
    metrics["max_drawdown_brl"] = equity_metrics.get("max_drawdown_brl")
    metrics["max_drawdown_pct"] = equity_metrics.get("max_drawdown_pct")

    report = {
        "validation_mode": SWING_VALIDATION_MODE,
        "paper_only": True,
        "timeframe": SWING_VALIDATION_INTERVAL,
        "timeframe_label": "Diario (1D)",
        "period": SWING_VALIDATION_PERIOD,
        "current_market_context": dict(payload.get("market_context", {}) or {}),
        "validation_started_at": started_at.isoformat(),
        "validation_window_end_at": window_end.isoformat(),
        "validation_day_number": day_number,
        "validation_phase": phase,
        "validation_status": "completed" if day_number >= SWING_VALIDATION_DAYS else "running",
        "metrics": metrics,
        "performance": performance,
        "rejection_quality": rejection_summary,
        "feed_rejection_consistency": feed_rejection_consistency,
        "calibration_preview": dict(payload.get("calibration_preview", {}) or default_calibration_preview_state()),
        "strategy_bottleneck": dict(payload.get("strategy_bottleneck", {}) or default_strategy_bottleneck_state()),
        "most_used_assets": most_used_assets,
        "best_assets": best_assets,
        "worst_assets": worst_assets,
        "reports_df": reports_df,
    }
    report["before_after_comparison"] = _comparison_payload(reports_df, started_at, window_end)
    report["suggestions"] = _build_suggestions(report)
    report["errors"] = _build_errors(report)
    report["successes"] = _build_successes(report)
    report["context_impact_estimate"] = _build_context_impact_estimate(report)
    report["phase_conclusion"] = _phase_conclusion(day_number, report)
    report["consistency"] = _build_operational_consistency(report, payload, equity_metrics)

    if day_number >= SWING_VALIDATION_DAYS:
        verdict, reason = _final_verdict(report, payload)
        report["final_validation_grade"] = verdict
        report["final_validation_reason"] = reason
        report["verdict_message"] = VERDICT_MESSAGES.get(verdict, "")
        report["final_validation_generated_at"] = current_time.isoformat()
    else:
        report["final_validation_grade"] = ""
        report["final_validation_reason"] = ""
        report["verdict_message"] = ""
        report["final_validation_generated_at"] = ""

    return report


def refresh_swing_validation_cycle(
    *,
    cycle_result: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current_time = _utc_now(now)
    state = load_bot_state()
    validation_state = dict(state.get("validation", {}) or {})
    defaults = default_validation_state()
    for key, value in defaults.items():
        validation_state.setdefault(key, value)

    if not validation_state.get("validation_started_at"):
        validation_state["validation_started_at"] = current_time.isoformat()
    validation_state["validation_day_number"] = _validation_day_number(validation_state.get("validation_started_at"), current_time)
    validation_state["validation_phase"] = _phase_for_day(int(validation_state["validation_day_number"]))
    validation_state["validation_status"] = "completed" if int(validation_state["validation_day_number"]) >= SWING_VALIDATION_DAYS else "running"
    validation_state["validation_mode"] = SWING_VALIDATION_MODE
    validation_state["paper_only"] = True
    validation_state["timeframe"] = SWING_VALIDATION_INTERVAL
    validation_state["timeframe_label"] = "Diario (1D)"
    validation_state["period_label"] = SWING_VALIDATION_PERIOD
    validation_state["last_evaluated_at"] = current_time.isoformat()

    validation_state = _update_signal_counters(validation_state, cycle_result)
    if isinstance(cycle_result, dict):
        state["calibration_preview"] = build_calibration_preview(
            signals=list(cycle_result.get("signals", []) or []),
            state=state,
            paper_state=dict(cycle_result.get("state", {}) or {}),
            market_data_status=dict(cycle_result.get("market_data_status", {}) or {}),
            macro_alert=dict(cycle_result.get("macro_alert", {}) or {}),
            enabled=CALIBRATION_PREVIEW_ENABLED,
            margin=CALIBRATION_PREVIEW_MARGIN,
            max_examples=CALIBRATION_PREVIEW_MAX_EXAMPLES,
        )
        state["strategy_bottleneck"] = summarize_strategy_bottlenecks(
            signals=list(cycle_result.get("signals", []) or []),
            enabled=True,
            max_candidates=10,
        )
    state["validation"] = validation_state
    save_bot_state(state)

    updated_state = load_bot_state()
    report = build_swing_validation_report(updated_state, now=current_time)
    sanitized_report = {key: value for key, value in report.items() if key != "reports_df"}
    updated_state["validation"] = dict(updated_state.get("validation", {}) or {})
    updated_state["validation"].update(
        {
            "validation_mode": SWING_VALIDATION_MODE,
            "validation_started_at": sanitized_report.get("validation_started_at", validation_state.get("validation_started_at", "")),
            "validation_day_number": sanitized_report.get("validation_day_number", validation_state.get("validation_day_number", 1)),
            "validation_phase": sanitized_report.get("validation_phase", validation_state.get("validation_phase", PHASE_COLETA)),
            "validation_status": sanitized_report.get("validation_status", validation_state.get("validation_status", "running")),
            "final_validation_grade": sanitized_report.get("final_validation_grade", ""),
            "final_validation_reason": sanitized_report.get("final_validation_reason", ""),
            "final_validation_generated_at": sanitized_report.get("final_validation_generated_at", ""),
            "last_report": sanitized_report,
            "last_evaluated_at": current_time.isoformat(),
            "paper_only": True,
            "timeframe": SWING_VALIDATION_INTERVAL,
            "timeframe_label": "Diario (1D)",
            "period_label": SWING_VALIDATION_PERIOD,
            "rejection_top_reason": sanitized_report.get("metrics", {}).get("rejection_top_reason", validation_state.get("rejection_top_reason", "")),
            "rejection_top_layer": sanitized_report.get("metrics", {}).get("rejection_top_layer", validation_state.get("rejection_top_layer", "")),
            "rejection_top_strategy": sanitized_report.get("metrics", {}).get("rejection_top_strategy", validation_state.get("rejection_top_strategy", "")),
            "rejection_reason_breakdown": sanitized_report.get("metrics", {}).get(
                "rejection_reason_breakdown",
                validation_state.get("rejection_reason_breakdown", {}),
            ),
            "rejection_layer_breakdown": sanitized_report.get("metrics", {}).get(
                "rejection_layer_breakdown",
                validation_state.get("rejection_layer_breakdown", {}),
            ),
            "rejection_strategy_breakdown": sanitized_report.get("metrics", {}).get(
                "rejection_strategy_breakdown",
                validation_state.get("rejection_strategy_breakdown", {}),
            ),
            "rejection_has_minimum_sample": sanitized_report.get("metrics", {}).get(
                "rejection_has_minimum_sample",
                validation_state.get("rejection_has_minimum_sample", False),
            ),
            "last_cycle_rejection_summary": validation_state.get("last_cycle_rejection_summary", {}),
            "current_cycle_rejection_reason": sanitized_report.get("metrics", {}).get(
                "current_cycle_rejection_reason",
                validation_state.get("current_cycle_rejection_reason", ""),
            ),
            "accumulated_rejection_reason": sanitized_report.get("metrics", {}).get(
                "accumulated_rejection_reason",
                validation_state.get("accumulated_rejection_reason", ""),
            ),
            "fallback_rejection_current_cycle_count": sanitized_report.get("metrics", {}).get(
                "fallback_rejection_current_cycle_count",
                validation_state.get("fallback_rejection_current_cycle_count", 0),
            ),
            "fallback_rejection_accumulated_count": sanitized_report.get("metrics", {}).get(
                "fallback_rejection_accumulated_count",
                validation_state.get("fallback_rejection_accumulated_count", 0),
            ),
            "strategy_rejection_current_cycle_count": sanitized_report.get("metrics", {}).get(
                "strategy_rejection_current_cycle_count",
                validation_state.get("strategy_rejection_current_cycle_count", 0),
            ),
            "strategy_rejection_accumulated_count": sanitized_report.get("metrics", {}).get(
                "strategy_rejection_accumulated_count",
                validation_state.get("strategy_rejection_accumulated_count", 0),
            ),
            "guard_rejection_current_cycle_count": sanitized_report.get("metrics", {}).get(
                "guard_rejection_current_cycle_count",
                validation_state.get("guard_rejection_current_cycle_count", 0),
            ),
            "guard_rejection_accumulated_count": sanitized_report.get("metrics", {}).get(
                "guard_rejection_accumulated_count",
                validation_state.get("guard_rejection_accumulated_count", 0),
            ),
            "feed_rejection_consistency": sanitized_report.get(
                "feed_rejection_consistency",
                sanitized_report.get("metrics", {}).get(
                    "feed_rejection_consistency",
                    validation_state.get("feed_rejection_consistency", {}),
                ),
            ),
        }
    )
    updated_state["calibration_preview"] = dict(
        sanitized_report.get("calibration_preview")
        or updated_state.get("calibration_preview", {})
        or default_calibration_preview_state()
    )
    updated_state["strategy_bottleneck"] = dict(
        sanitized_report.get("strategy_bottleneck")
        or updated_state.get("strategy_bottleneck", {})
        or default_strategy_bottleneck_state()
    )
    save_bot_state(updated_state)
    return sanitized_report


def reset_swing_validation_cycle(now: datetime | None = None) -> dict[str, Any]:
    current_time = _utc_now(now)
    state = load_bot_state()
    validation_state = default_validation_state()
    validation_state["validation_started_at"] = current_time.isoformat()
    validation_state["last_reset_at"] = current_time.isoformat()
    state["validation"] = validation_state
    save_bot_state(state)
    return refresh_swing_validation_cycle(now=current_time)
