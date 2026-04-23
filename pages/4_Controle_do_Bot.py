from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from core.alerts import send_email_alert
from core.auth.guards import render_auth_toolbar, require_admin
from core.broker import broker_status_label, probe_broker_status
from core.config import ALERT_EMAIL_ENABLED, ALERT_EMAIL_FROM, ALERT_EMAIL_PROVIDER, PRODUCTION_MODE, SMTP_USERNAME
from core.market_data import build_feed_quality_snapshot, classify_feed_status, format_market_timestamp, legacy_market_status
from core.production_monitor import evaluate_production_health
from core.signal_rejection_analysis import rejection_dominant_message, rejection_layer_label, rejection_reason_label
from core.state_store import (
    load_bot_state,
    log_event,
    resolve_market_data_views,
    reset_state,
    save_bot_state,
    update_broker_status,
    update_production_status,
)
from core.swing_validation import refresh_swing_validation_cycle, reset_swing_validation_cycle
from engines.trader_engine import run_trader_cycle


def market_data_status_label(raw_status: str | None) -> str:
    payload = raw_status if isinstance(raw_status, dict) else {"status": raw_status}
    return classify_feed_status(
        status=payload.get("feed_status") or payload.get("status"),
        last_source=payload.get("last_source"),
        source_breakdown=payload.get("source_breakdown"),
    )


def market_data_provider_label(raw_status: dict | None) -> str:
    payload = raw_status if isinstance(raw_status, dict) else {"provider": raw_status}
    provider = str(payload.get("provider") or "unknown").strip().lower()
    labels = {
        "twelvedata": "Twelve Data",
        "yahoo": "Yahoo",
        "synthetic": "Fallback sintetico",
        "mixed": "Twelve Data + Yahoo",
        "unknown": "Desconhecido",
    }
    return labels.get(provider, provider.upper())


def market_data_source_label(raw_source: str | None) -> str:
    payload = raw_source if isinstance(raw_source, dict) else {"last_source": raw_source}
    labels = {
        "market": "Mercado ao vivo",
        "cached": "Cache reaproveitado",
        "fallback": "Fallback sintetico",
        "mixed": "Misto",
        "unknown": "Desconhecido",
    }
    source = str(payload.get("last_source") or "unknown").strip().lower()
    base_label = labels.get(source, str(source or "Desconhecido"))
    provider_label = market_data_provider_label(payload)
    if source == "fallback":
        return base_label
    return f"{base_label} via {provider_label}"


def twelvedata_diagnostic_payload(raw_status: dict | None) -> dict:
    payload = raw_status if isinstance(raw_status, dict) else {}
    diagnostics = payload.get("provider_diagnostics", {}) or {}
    diagnostic = diagnostics.get("twelvedata", {}) if isinstance(diagnostics, dict) else {}
    return dict(diagnostic or {})


def audit_display_value(value: object) -> str:
    if value is None:
        return "NAO REGISTRADO NO ESTADO ATUAL"
    if isinstance(value, str) and not value.strip():
        return "NAO REGISTRADO NO ESTADO ATUAL"
    if isinstance(value, bool):
        return "Sim" if value else "Nao"
    return str(value)


def symbol_list_label(symbols: list[str] | None) -> str:
    values = [str(item).upper() for item in (symbols or []) if str(item)]
    return ", ".join(values) if values else "Nenhum"


def pct_label(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value or 0.0) * 100:.1f}%"


def worker_instrumentation_confirmed(payload: dict | None) -> bool:
    data = payload or {}
    return (
        str(data.get("process_role") or "").strip().lower() == "worker"
        and str(data.get("state_writer") or "").strip().lower() == "worker"
        and bool(str(data.get("build_active") or "").strip())
        and bool(str(data.get("last_stage") or "").strip())
    )


def market_data_legacy_label(raw_status: dict | None) -> str:
    payload = raw_status or {}
    return legacy_market_status(
        status=payload.get("status_legacy") or payload.get("status"),
        last_source=payload.get("last_source"),
        source_breakdown=payload.get("source_breakdown"),
    )


def broker_mode_label(raw_mode: str | None) -> str:
    labels = {
        "paper": "Simulado",
        "live": "Real",
    }
    return labels.get(str(raw_mode or "paper").strip().lower(), str(raw_mode or "paper").title())


def bot_status_label(raw_status: str | None) -> str:
    labels = {
        "RUNNING": "Ligado",
        "PAUSED": "Pausado",
        "STOPPED": "Parado",
    }
    return labels.get(str(raw_status or "").upper(), str(raw_status or "Desconhecido"))


def health_level_label(raw_level: str | None) -> str:
    labels = {
        "healthy": "Saudavel",
        "warning": "Atencao",
        "critical": "Critico",
    }
    return labels.get(str(raw_level or "healthy").strip().lower(), str(raw_level or "healthy").title())


def market_context_label(raw_status: str | None) -> str:
    labels = {
        "FAVORAVEL": "Favoravel",
        "NEUTRO": "Neutro",
        "DESFAVORAVEL": "Desfavoravel",
        "CRITICO": "Critico",
    }
    return labels.get(str(raw_status or "NEUTRO").strip().upper(), str(raw_status or "NEUTRO").title())


current_user = require_admin()
render_auth_toolbar()

st.title("Controle Operacional")
st.caption(f"Painel administrativo do trader. Usuario: {current_user['username']}")
st.code("BUILD_UI_MARKER_20260422_A")

state = load_bot_state()
security_state = state.get("security", {}) or {}
operational_market_state, chart_market_state = resolve_market_data_views(state)
broker_state = update_broker_status(probe_broker_status(security_state, requested_by="admin_panel"))
state = load_bot_state()
production_state = update_production_status(evaluate_production_health(state))
state = load_bot_state()
production_state = state.get("production", {}) or {}
operational_market_state, chart_market_state = resolve_market_data_views(state)
broker_state = state.get("broker", {}) or broker_state
validation_report = refresh_swing_validation_cycle()
state = load_bot_state()
validation_state = state.get("validation", {}) or {}
market_context_state = state.get("market_context", {}) or {}
risk_state = state.get("risk", {}) or {}
daily_loss_limit_brl = float(risk_state.get("daily_loss_limit_brl", 0.0) or 0.0)
daily_loss_consumed_brl = float(risk_state.get("daily_loss_consumed_brl", 0.0) or 0.0)
daily_loss_remaining_brl = float(risk_state.get("daily_loss_remaining_brl", 0.0) or 0.0)
daily_realized_pnl_brl = float(risk_state.get("daily_realized_pnl_brl", 0.0) or 0.0)
daily_loss_day_key = str(risk_state.get("daily_loss_day_key") or "-")
daily_loss_block_active = bool(risk_state.get("daily_loss_block_active", False))
daily_loss_blocked_at = str(risk_state.get("daily_loss_blocked_at") or "")
daily_loss_block_reason = str(risk_state.get("daily_loss_block_reason") or "")

if bool(security_state.get("real_mode_enabled", False)):
    st.warning("Real trading enabled")

st.subheader("Modo producao")
st.caption("Monitoramento de saude, alertas e diagnostico operacional. PAPER TRADING permanece como padrao nesta etapa.")

production_mode_text = "Ativo" if PRODUCTION_MODE else "Inativo"
alert_mode_text = "Ativo" if ALERT_EMAIL_ENABLED else "Inativo"
alert_provider_text = str(production_state.get("alert_provider") or ALERT_EMAIL_PROVIDER or "smtp").upper()
configured_sender = ALERT_EMAIL_FROM or SMTP_USERNAME or "Sem registro"
health_level = str(production_state.get("health_level") or "healthy").lower()
health_message = str(production_state.get("health_message") or "Sem mensagem.")

if health_level == "healthy":
    st.success(f"{health_level_label(health_level)}: {health_message}")
elif health_level == "warning":
    st.warning(f"{health_level_label(health_level)}: {health_message}")
else:
    st.error(f"{health_level_label(health_level)}: {health_message}")

prod_c1, prod_c2, prod_c3, prod_c4 = st.columns(4)
prod_c1.metric("Status geral", health_level_label(health_level))
prod_c2.metric("Heartbeat age (s)", str(production_state.get("heartbeat_age_seconds") or 0))
prod_c3.metric("Falhas consecutivas", str(production_state.get("consecutive_errors") or 0))
prod_c4.metric("Alertas por email", alert_mode_text)

prod_c5, prod_c6, prod_c7, prod_c8 = st.columns(4)
prod_c5.metric("Modo producao", production_mode_text)
prod_c6.metric("Ultimo heartbeat", state.get("worker_heartbeat") or "Sem registro")
prod_c7.metric("Ultima execucao", production_state.get("last_execution_at") or "Sem registro")
prod_c8.metric("Ultimo sucesso", production_state.get("last_success_at") or "Sem registro")

prod_c9, prod_c10, prod_c11, prod_c12 = st.columns(4)
prod_c9.metric("Feed monitorado", market_data_status_label(production_state.get("feed_status")))
prod_c10.metric("Broker monitorado", broker_status_label(production_state.get("broker_status")))
prod_c11.metric("Ultimo alerta", production_state.get("last_alert_sent_at") or "Nenhum")
prod_c12.metric("Proximo alerta elegivel", production_state.get("next_alert_eligible_at") or "Agora")

st.caption(
    f"Provider de alerta: {alert_provider_text} | Remetente configurado: {configured_sender}"
)

st.subheader("Trava de perda diária (paper)")
st.caption(
    "Bloqueio operacional explícito: ao atingir o limite diário de perda, o robô não abre novas entradas. "
    "Posições já abertas continuam sob gestão normal."
)
if daily_loss_block_active:
    st.error("Trava diária ativa: novas entradas bloqueadas por perda diária.")
else:
    st.success("Trava diária pronta: novas entradas liberadas neste momento.")

risk_c1, risk_c2, risk_c3, risk_c4 = st.columns(4)
risk_c1.metric("Estado", "Bloqueado" if daily_loss_block_active else "Liberado")
risk_c2.metric("Limite diário", f"R$ {daily_loss_limit_brl:,.2f}")
risk_c3.metric("Perda consumida", f"R$ {daily_loss_consumed_brl:,.2f}")
risk_c4.metric("Limite restante", f"R$ {daily_loss_remaining_brl:,.2f}")
st.caption(
    f"Dia operacional UTC: {daily_loss_day_key} | "
    f"PnL realizado do dia (base da trava): R$ {daily_realized_pnl_brl:,.2f}"
)
if daily_loss_block_active:
    st.caption(
        f"Bloqueio ativado em: {format_market_timestamp(daily_loss_blocked_at)} | "
        f"Motivo: {daily_loss_block_reason or 'Limite diário atingido.'}"
    )
if risk_state.get("daily_loss_reset_at"):
    st.caption(
        f"Último reset automático na virada do dia UTC: "
        f"{format_market_timestamp(risk_state.get('daily_loss_reset_at'))}"
    )

act_c1, act_c2 = st.columns(2)
with act_c1:
    if st.button("Testar email", use_container_width=True):
        result = send_email_alert(
            "[Trade Ops Desk] Teste de alerta",
            "Email de teste do modo producao.\nNenhuma ordem real foi habilitada.\nBroker atual: PAPER.",
            alert_type="manual_test",
            force=True,
        )
        if result.get("sent"):
            st.success("Email de teste enviado.")
        else:
            st.warning(f"Email de teste nao enviado: {result.get('reason') or 'bloqueado'}")

with act_c2:
    if st.button("Recalcular saude", use_container_width=True):
        production_state = update_production_status(evaluate_production_health(load_bot_state()))
        st.info(f"Saude recalculada: {health_level_label(production_state.get('health_level'))}")

st.subheader("Validacao swing 10 dias")
st.caption("Camada de validacao paper para swing profissional. Nenhuma ordem real e liberada nesta etapa.")

validation_status = str(validation_report.get("validation_status") or "running")
validation_phase = str(validation_report.get("validation_phase") or validation_state.get("validation_phase") or "Coleta e observacao")
final_grade = str(validation_report.get("final_validation_grade") or validation_state.get("final_validation_grade") or "").strip()
verdict_message = str(validation_report.get("verdict_message") or "").strip()
phase_conclusion = str(validation_report.get("phase_conclusion") or "Sem conclusao")

if final_grade == "APROVADO":
    st.success(verdict_message or "Ciclo aprovado.")
elif final_grade == "APROVADO_COM_AJUSTES":
    st.warning(verdict_message or "Ciclo aprovado com ajustes.")
elif final_grade in {"REPROVADO_ESTRATEGIA", "REPROVADO_INSTABILIDADE"}:
    st.error(verdict_message or "Ciclo reprovado.")
else:
    st.info(f"Fase atual: {validation_phase} | Conclusao parcial: {phase_conclusion}")

val_metrics = dict(validation_report.get("metrics", {}) or {})
val_perf = dict(validation_report.get("performance", {}) or {})
val_consistency = dict(validation_report.get("consistency", {}) or {})

val_c1, val_c2, val_c3, val_c4 = st.columns(4)
val_c1.metric("Dia atual", str(int(validation_report.get("validation_day_number", 1) or 1)))
val_c2.metric("Fase", validation_phase)
val_c3.metric("Status do ciclo", "Finalizado" if validation_status == "completed" else "Em validacao")
val_c4.metric("Timeframe", str(validation_report.get("timeframe_label") or validation_state.get("timeframe_label") or "Diario (1D)"))

val_c5, val_c6, val_c7, val_c8 = st.columns(4)
val_c5.metric("Trades fechados", str(int(val_metrics.get("trades_closed", 0) or 0)))
val_c6.metric("Win rate", f"{float(val_perf.get('win_rate', 0.0) or 0.0) * 100:.2f}%")
val_c7.metric("Payoff", "-" if val_perf.get("payoff") is None else f"{float(val_perf.get('payoff') or 0.0):.2f}")
val_c8.metric("PnL total", f"R$ {float(val_perf.get('pnl_total', 0.0) or 0.0):,.2f}")

val_c9, val_c10, val_c11, val_c12 = st.columns(4)
val_c9.metric("Sinais aprovados", str(int(val_metrics.get("signals_approved", 0) or 0)))
val_c10.metric("Sinais rejeitados", str(int(val_metrics.get("signals_rejected", 0) or 0)))
val_c11.metric(
    "Fallback do ciclo",
    "-" if val_metrics.get("fallback_cycle_pct") is None else f"{float(val_metrics.get('fallback_cycle_pct') or 0.0):.2f}%",
)
val_c12.metric("Erros operacionais", str(int(val_metrics.get("operational_errors", 0) or 0)))

cons_c1, cons_c2, cons_c3, cons_c4 = st.columns(4)
cons_c1.metric("Amostra atual", str(val_consistency.get("sample_quality_label") or "Sem leitura"))
cons_c2.metric("Postura", str(val_consistency.get("operational_posture_label") or "Indefinida"))
cons_c3.metric(
    "Drawdown max",
    "-"
    if val_metrics.get("max_drawdown_pct") is None
    else f"{float(val_metrics.get('max_drawdown_pct') or 0.0) * 100:.2f}%",
)
cons_c4.metric(
    "Watchlist da fase",
    "Coerente" if bool(val_consistency.get("watchlist_phase_aligned")) else "Fora da fase",
)
st.caption(
    "Aprovacao de sinais no ciclo: "
    + (
        "-"
        if val_consistency.get("signal_approval_rate") is None
        else f"{float(val_consistency.get('signal_approval_rate') or 0.0) * 100:.1f}%"
    )
)
if val_consistency.get("sample_quality_message"):
    st.caption(f"Amostra: {val_consistency.get('sample_quality_message')}")
if val_consistency.get("watchlist_message"):
    st.caption(f"Watchlist: {val_consistency.get('watchlist_message')}")
if val_consistency.get("capital_phase_aligned") is False:
    st.warning(
        "O capital atual do runtime nao coincide com o capital-base recomendado da fase. "
        "Para um novo ciclo limpo, use o reset operacional do trader."
    )

st.subheader("Qualidade de sinal da FASE 2")
signal_c1, signal_c2, signal_c3, signal_c4, signal_c5, signal_c6 = st.columns(6)
signal_c1.metric("Sinais aprovados", str(int(val_metrics.get("signals_approved", 0) or 0)))
signal_c2.metric("Sinais rejeitados", str(int(val_metrics.get("signals_rejected", 0) or 0)))
signal_c3.metric(
    "Taxa de aprovacao",
    "-"
    if val_consistency.get("signal_approval_rate") is None
    else f"{float(val_consistency.get('signal_approval_rate') or 0.0) * 100:.1f}%",
)
signal_c4.metric("Amostra do ciclo", str(val_consistency.get("sample_quality_label") or "Sem leitura"))
signal_c5.metric("Postura atual", str(val_consistency.get("operational_posture_label") or "Indefinida"))
signal_c6.metric("Leitura simples", str(val_consistency.get("signal_quality_label") or "Baixa"))

signal_d1, signal_d2 = st.columns(2)
with signal_d1:
    st.write(
        f"Consistencia da watchlist: "
        f"{'Coerente com a fase' if bool(val_consistency.get('watchlist_phase_aligned')) else 'Fora da fase'}"
    )
    if val_consistency.get("signal_quality_message"):
        st.caption(f"Sinal: {val_consistency.get('signal_quality_message')}")
with signal_d2:
    st.write(
        f"Ajuste fino: "
        f"{'Ja existe base minima' if bool(val_consistency.get('fine_tuning_ready')) else 'Ainda nao'}"
    )
    if val_consistency.get("validation_reading_message"):
        st.caption(f"Validacao: {val_consistency.get('validation_reading_message')}")

rejection_quality = dict(validation_report.get("rejection_quality", {}) or {})
rejection_top_reasons = rejection_quality.get("top_reasons", []) or []
st.subheader("Qualidade de rejeicao de sinal")
rej_c1, rej_c2, rej_c3, rej_c4 = st.columns(4)
rej_c1.metric(
    "Motivo dominante",
    rejection_reason_label(rejection_quality.get("top_reason")) if rejection_quality.get("top_reason") else "Sem leitura",
)
rej_c2.metric(
    "Camada dominante",
    rejection_layer_label(rejection_quality.get("top_layer")) if rejection_quality.get("top_layer") else "Sem leitura",
)
rej_c3.metric(
    "Setup mais bloqueado",
    str(rejection_quality.get("top_strategy") or "Sem leitura"),
)
rej_c4.metric(
    "Base minima para ajuste",
    "Sim" if bool(rejection_quality.get("has_minimum_sample")) else "Nao",
)

rej_d1, rej_d2 = st.columns(2)
with rej_d1:
    st.write("**Top 5 motivos de rejeicao**")
    if rejection_top_reasons:
        for item in rejection_top_reasons:
            st.caption(
                f"- {item.get('human_reason')}: {int(item.get('count', 0) or 0)} "
                f"({pct_label(item.get('pct'))})"
            )
    else:
        st.caption("Sem rejeicoes suficientes registradas ate o momento.")
with rej_d2:
    layer_breakdown = dict(rejection_quality.get("layer_breakdown", {}) or {})
    strategy_breakdown = dict(rejection_quality.get("strategy_breakdown", {}) or {})
    st.write(
        f"**Leitura curta:** "
        f"{rejection_dominant_message(rejection_quality.get('top_layer')) if rejection_quality.get('top_layer') else 'Sem leitura consolidada.'}"
    )
    if layer_breakdown:
        layer_lines = [
            f"{rejection_layer_label(layer)}={pct_label((int(count or 0) / max(int(rejection_quality.get('total_rejection_events', 0) or 0), 1)))}"
            for layer, count in sorted(layer_breakdown.items(), key=lambda item: int(item[1] or 0), reverse=True)
        ]
        st.caption("Camadas: " + " | ".join(layer_lines))
    if strategy_breakdown:
        top_strategy, top_strategy_count = sorted(
            strategy_breakdown.items(),
            key=lambda item: int(item[1] or 0),
            reverse=True,
        )[0]
        st.caption(f"Estrategia mais bloqueada: {top_strategy} ({int(top_strategy_count or 0)})")

val_actions1, val_actions2 = st.columns(2)
with val_actions1:
    if st.button("Recalcular validacao swing", use_container_width=True):
        validation_report = refresh_swing_validation_cycle()
        st.info("Validacao swing recalculada.")
with val_actions2:
    if st.button("Reiniciar ciclo swing 10 dias", use_container_width=True):
        validation_report = reset_swing_validation_cycle()
        st.warning("Novo ciclo swing iniciado a partir de agora.")

validation_state = load_bot_state().get("validation", {}) or {}
st.caption(
    f"Inicio do ciclo: {validation_report.get('validation_started_at') or 'Sem registro'} | "
    f"Email final enviado: {'Sim' if validation_state.get('final_email_sent') else 'Nao'}"
)

if validation_report.get("final_validation_reason"):
    st.caption(f"Motivo final: {validation_report.get('final_validation_reason')}")

panel_c1, panel_c2 = st.columns(2)
with panel_c1:
    st.write("**Erros identificados**")
    errors = validation_report.get("errors", []) or []
    if errors:
        for item in errors:
            st.caption(f"- {item}")
    else:
        st.caption("Sem erros relevantes ate aqui.")

    st.write("**Ativos em destaque**")
    best_assets = pd.DataFrame(validation_report.get("best_assets", []) or [])
    if best_assets.empty:
        st.info("Sem ativos consistentes suficientes ate o momento.")
    else:
        st.dataframe(best_assets, use_container_width=True)

with panel_c2:
    st.write("**Acertos identificados**")
    successes = validation_report.get("successes", []) or []
    if successes:
        for item in successes:
            st.caption(f"- {item}")
    else:
        st.caption("Sem acertos destacados ate aqui.")

    st.write("**Ativos problematicos**")
    worst_assets = pd.DataFrame(validation_report.get("worst_assets", []) or [])
    if worst_assets.empty:
        st.info("Sem ativos problematicos suficientes ate o momento.")
    else:
        st.dataframe(worst_assets, use_container_width=True)

st.write("**Sugestoes analiticas**")
suggestions = validation_report.get("suggestions", []) or []
if suggestions:
    for suggestion in suggestions:
        st.caption(f"- {suggestion.get('message')}")
else:
    st.caption("Sem sugestoes novas neste momento.")

before_after = validation_report.get("before_after_comparison", {}) or {}
if before_after:
    before_after_c1, before_after_c2 = st.columns(2)
    before_payload = before_after.get("before", {}) or {}
    after_payload = before_after.get("after", {}) or {}
    with before_after_c1:
        st.write("**Antes da fase de ajuste**")
        st.caption(
            f"Trades: {int(before_payload.get('trades', 0) or 0)} | "
            f"Win rate: {float(before_payload.get('win_rate', 0.0) or 0.0) * 100:.2f}% | "
            f"PnL: R$ {float(before_payload.get('pnl', 0.0) or 0.0):,.2f}"
        )
    with before_after_c2:
        st.write("**Depois da fase de ajuste**")
        st.caption(
            f"Trades: {int(after_payload.get('trades', 0) or 0)} | "
            f"Win rate: {float(after_payload.get('win_rate', 0.0) or 0.0) * 100:.2f}% | "
            f"PnL: R$ {float(after_payload.get('pnl', 0.0) or 0.0):,.2f}"
        )
    notes = before_after.get("notes", []) or []
    if notes:
        for note in notes:
            st.caption(f"- {note}")

st.subheader("Contexto de mercado cripto")
st.caption("Filtro auxiliar de PAPER TRADING. O contexto nao dispara ordens; ele apenas endurece sinais fracos de cripto.")

context_c1, context_c2, context_c3, context_c4 = st.columns(4)
context_c1.metric("Status atual", market_context_label(market_context_state.get("market_context_status")))
context_c2.metric("Score", f"{float(market_context_state.get('market_context_score', 50.0) or 50.0):.1f}")
context_c3.metric(
    "Sinais bloqueados",
    str(int(val_metrics.get("context_blocked_signals", 0) or 0)),
)
context_c4.metric("PAPER", "Ativo")

st.caption(f"Motivo: {market_context_state.get('market_context_reason') or 'Sem motivo registrado.'}")
st.caption(f"Impacto no robo: {market_context_state.get('market_context_impact') or 'Sem impacto adicional.'}")
if validation_report.get("context_impact_estimate"):
    st.caption(f"Impacto estimado no periodo: {validation_report.get('context_impact_estimate')}")

context_status_counts = dict(val_metrics.get("context_status_counts", {}) or {})
if context_status_counts:
    st.caption(
        "Contexto por periodo: "
        f"FAVORAVEL={int(context_status_counts.get('FAVORAVEL', 0) or 0)} | "
        f"NEUTRO={int(context_status_counts.get('NEUTRO', 0) or 0)} | "
        f"DESFAVORAVEL={int(context_status_counts.get('DESFAVORAVEL', 0) or 0)} | "
        f"CRITICO={int(context_status_counts.get('CRITICO', 0) or 0)}"
    )
if market_context_state.get("market_context_regime"):
    watchlist_consistency = market_context_state.get("watchlist_consistency")
    watchlist_consistency_label = (
        "-"
        if watchlist_consistency is None
        else f"{float(watchlist_consistency or 0.0) * 100:.1f}%"
    )
    st.caption(
        "Regime observado: "
        f"{str(market_context_state.get('market_context_regime') or 'indefinido').capitalize()} | "
        f"Consistencia da watchlist: {watchlist_consistency_label}"
    )

feed_status_label = market_data_status_label(operational_market_state)
if feed_status_label == "LIVE":
    st.success(
        f"Feed classificado como LIVE via {market_data_provider_label(operational_market_state)}."
    )
elif feed_status_label == "DELAYED":
    st.warning(
        f"Feed classificado como DELAYED: usando {market_data_source_label(operational_market_state)}. "
        "A UI deixa isso explicito e o worker nao trata dado atrasado como se fosse ao vivo."
    )
elif feed_status_label == "FALLBACK":
    st.error(
        f"Feed classificado como FALLBACK via {market_data_provider_label(operational_market_state)}. "
        "O worker continua online, mas evita operar com dado nao confiavel."
    )
else:
    st.info("Status do feed ainda nao determinado.")

info_c1, info_c2, info_c3, info_c4, info_c5 = st.columns(5)
info_c1.metric("Status do bot", bot_status_label(state.get("bot_status")))
info_c2.metric("Provider de dados", market_data_provider_label(operational_market_state))
info_c3.metric("Status do feed", feed_status_label)
info_c4.metric("Fonte atual", market_data_source_label(operational_market_state))
info_c5.metric("Modo do broker", broker_mode_label(broker_state.get("mode")))

diag_c1, diag_c2 = st.columns(2)
with diag_c1:
    st.caption(
        f"Ultimo sync do feed: {format_market_timestamp(operational_market_state.get('last_sync_at'))}"
    )
    st.caption(
        f"Ultimo sucesso: {format_market_timestamp(operational_market_state.get('last_success_at'))}"
    )
with diag_c2:
    st.caption(f"Broker provider: {str(broker_state.get('provider', 'paper')).upper()}")
    st.caption(f"Status do broker: {broker_status_label(broker_state.get('status'))}")

if operational_market_state.get("last_error"):
    st.caption(f"Ultimo alerta de mercado: {operational_market_state.get('last_error')}")
if broker_state.get("warning"):
    st.caption(f"Observacao do broker: {broker_state.get('warning')}")
if production_state.get("last_alert_error"):
    st.caption(f"Ultima falha no envio de alerta: {production_state.get('last_alert_error')}")

worker_feed_quality = build_feed_quality_snapshot(operational_market_state)
chart_feed_quality = build_feed_quality_snapshot(chart_market_state)
st.subheader("Qualidade do feed")
st.caption("Separacao explicita entre o feed operacional do worker e o ultimo feed visual usado pelo grafico do Trader.")
feed_c1, feed_c2, feed_c3, feed_c4 = st.columns(4)
feed_c1.metric("Sucesso Twelve Data", pct_label(worker_feed_quality.get("twelvedata_success_rate")))
feed_c2.metric("Ultimo sucesso real", format_market_timestamp(worker_feed_quality.get("last_success_at")))
feed_c3.metric("Ativos live no ciclo", f"{int(worker_feed_quality.get('live_count') or 0)}/{int(worker_feed_quality.get('total_symbols') or 0)}")
feed_c4.metric("Ativos em fallback", str(int(worker_feed_quality.get("fallback_count") or 0)))

feed_detail_c1, feed_detail_c2 = st.columns(2)
with feed_detail_c1:
    st.write("**Feed operacional do worker**")
    st.write(
        f"Status: {market_data_status_label(operational_market_state)} | "
        f"Fonte: {market_data_source_label(operational_market_state)}"
    )
    st.write(f"Ativos live: {symbol_list_label(worker_feed_quality.get('live_symbols'))}")
    st.write(f"Ativos em fallback: {symbol_list_label(worker_feed_quality.get('fallback_symbols'))}")
    if worker_feed_quality.get("fallback_reason"):
        st.caption(f"Motivo do fallback operacional: {worker_feed_quality.get('fallback_reason')}")
    if worker_feed_quality.get("quality_message"):
        st.caption(worker_feed_quality.get("quality_message"))
with feed_detail_c2:
    st.write("**Feed do grafico do Trader**")
    st.write(
        f"Status: {market_data_status_label(chart_market_state)} | "
        f"Fonte: {market_data_source_label(chart_market_state)}"
    )
    st.write(f"Ativos live: {symbol_list_label(chart_feed_quality.get('live_symbols'))}")
    st.write(f"Ativos em fallback: {symbol_list_label(chart_feed_quality.get('fallback_symbols'))}")
    if chart_feed_quality.get("fallback_reason"):
        st.caption(f"Motivo do fallback visual: {chart_feed_quality.get('fallback_reason')}")
    st.caption("Fallback apenas visual do grafico nao altera o feed operacional do worker.")

with st.expander("Diagnostico do feed"):
    st.write("**Contexto operacional (worker):**")
    st.write(f"Classificacao atual: {market_data_status_label(operational_market_state)}")
    st.write(f"Taxonomia legada: {market_data_legacy_label(operational_market_state)}")
    st.write(f"Solicitado por: {operational_market_state.get('requested_by') or 'Sem registro'}")
    st.write(f"Ativos monitorados: {', '.join(operational_market_state.get('symbols', []) or []) or 'Sem registro'}")
    st.write(f"Ultimo sync: {format_market_timestamp(operational_market_state.get('last_sync_at'))}")
    st.code(
        json.dumps(operational_market_state.get("source_breakdown", {}) or {}, ensure_ascii=False, indent=2),
        language="json",
    )
    td_diag = twelvedata_diagnostic_payload(operational_market_state)
    if td_diag:
        st.write("**Diagnostico Twelve Data (worker):**")
        st.write(f"Build ativo: {td_diag.get('build_label') or 'Sem registro'}")
        st.write(f"Servico: {td_diag.get('service_name') or 'Sem registro'}")
        st.write(f"API key lida pelo processo: {'Sim' if td_diag.get('api_key_present') else 'Nao'}")
        st.write(f"Tamanho da chave: {int(td_diag.get('api_key_length') or 0)}")
        st.write(f"Base URL: {td_diag.get('api_base') or 'Sem registro'}")
        st.write(f"Host resolvido: {td_diag.get('api_base_host') or 'Sem registro'}")
        st.write(f"Base URL valida: {'Sim' if td_diag.get('api_base_valid') else 'Nao'}")
        st.write(f"Simbolo amostra: {td_diag.get('sample_symbol') or 'Sem registro'}")
        st.write(f"Simbolo normalizado: {td_diag.get('sample_normalized_symbol') or 'Sem registro'}")
        st.write(f"Request montado: {'Sim' if td_diag.get('request_built') else 'Nao'}")
        st.write(f"Request saiu do processo: {'Sim' if td_diag.get('request_attempted') else 'Nao'}")
        st.write(f"Resposta recebida: {'Sim' if td_diag.get('response_received') else 'Nao'}")
        st.write(f"Sucessos Twelve Data no ciclo: {int(td_diag.get('success_count') or 0)}")
        st.write(f"Ultimo estagio: {td_diag.get('last_stage') or 'Sem registro'}")
        if td_diag.get("http_statuses"):
            st.write(f"HTTP status observados: {', '.join(str(item) for item in (td_diag.get('http_statuses') or []))}")
        if td_diag.get("payload_codes"):
            st.write(f"Codigos retornados: {', '.join(str(item) for item in (td_diag.get('payload_codes') or []))}")
        if td_diag.get("last_error"):
            st.write(f"Ultimo erro Twelve Data: {td_diag.get('last_error')}")
        st.code(json.dumps(td_diag, ensure_ascii=False, indent=2), language="json")
    if chart_market_state:
        st.write("**Ultimo contexto visual (Trader):**")
        st.write(f"Classificacao atual: {market_data_status_label(chart_market_state)}")
        st.write(f"Taxonomia legada: {market_data_legacy_label(chart_market_state)}")
        st.write(f"Solicitado por: {chart_market_state.get('requested_by') or 'Sem registro'}")
        st.write(f"Ativos monitorados: {', '.join(chart_market_state.get('symbols', []) or []) or 'Sem registro'}")
        st.write(f"Ultimo sync: {format_market_timestamp(chart_market_state.get('last_sync_at'))}")
        st.code(
            json.dumps(chart_market_state.get("source_breakdown", {}) or {}, ensure_ascii=False, indent=2),
            language="json",
        )

st.markdown("### AUDITORIA DO WORKER")
st.write("controle_bot_ui_version: audit_v2")
worker_confirmed = worker_instrumentation_confirmed(operational_market_state)
if worker_confirmed:
    st.success("WORKER INSTRUMENTADO CONFIRMADO")
else:
    st.error("WORKER INSTRUMENTADO NAO COMPROVADO")

audit_c1, audit_c2 = st.columns(2)
with audit_c1:
    st.write(f"ui_audit_probe: {audit_display_value(operational_market_state.get('ui_audit_probe'))}")
    st.write(f"Build ativo: {audit_display_value(operational_market_state.get('build_active'))}")
    st.write(f"Git SHA: {audit_display_value(operational_market_state.get('git_sha'))}")
    st.write(f"Build timestamp: {audit_display_value(operational_market_state.get('build_timestamp'))}")
    st.write(f"Runtime iniciado em: {audit_display_value(operational_market_state.get('runtime_started_at'))}")
    st.write(f"Servico: {audit_display_value(operational_market_state.get('service_name'))}")
    st.write(f"Papel do processo: {audit_display_value(operational_market_state.get('process_role'))}")
    st.write(f"Ultima gravacao do estado: {audit_display_value(operational_market_state.get('state_written_at'))}")
    st.write(f"Writer do estado: {audit_display_value(operational_market_state.get('state_writer'))}")
    st.write(f"Build SHA do estado: {audit_display_value(operational_market_state.get('state_build_sha'))}")
    st.write(f"Schema do estado: {audit_display_value(operational_market_state.get('state_schema_version'))}")
with audit_c2:
    st.write(f"API key presente: {audit_display_value(operational_market_state.get('api_key_present'))}")
    st.write(f"Request preparado: {audit_display_value(operational_market_state.get('request_prepared'))}")
    st.write(f"Request tentado: {audit_display_value(operational_market_state.get('request_attempted'))}")
    st.write(f"Resposta recebida: {audit_display_value(operational_market_state.get('response_received'))}")
    st.write(f"Status code: {audit_display_value(operational_market_state.get('response_status_code'))}")
    st.write(f"Ultimo estagio: {audit_display_value(operational_market_state.get('last_stage'))}")
    st.write(f"Ultimo erro: {audit_display_value(operational_market_state.get('last_error'))}")
    requested_symbols = operational_market_state.get("requested_symbols") or operational_market_state.get("symbols") or []
    st.write(
        "Simbolos solicitados: "
        f"{audit_display_value(', '.join(str(item) for item in requested_symbols) if requested_symbols else '')}"
    )
    st.write(f"Provider efetivo: {audit_display_value(operational_market_state.get('provider_effective') or operational_market_state.get('provider'))}")

with st.expander("Diagnostico do broker"):
    st.write(f"**Provider:** {str(broker_state.get('provider', 'paper')).upper()}")
    st.write(f"**Status:** {broker_status_label(broker_state.get('status'))}")
    st.write(f"**Modo configurado:** {broker_mode_label(broker_state.get('configured_mode'))}")
    st.write(f"**Modo efetivo:** {broker_mode_label(broker_state.get('effective_mode'))}")
    st.write(f"**Conta:** {broker_state.get('account_id') or 'Sem registro'}")
    st.write(f"**Base URL:** {broker_state.get('base_url') or 'Sem registro'}")
    st.write(f"**API key configurada:** {'Sim' if broker_state.get('api_key_configured') else 'Nao'}")
    st.write(f"**API secret configurada:** {'Sim' if broker_state.get('api_secret_configured') else 'Nao'}")
    st.write(f"**Pode enviar ordens agora:** {'Sim' if broker_state.get('can_submit_orders') else 'Nao'}")
    st.write(f"**Execucao real habilitada nesta etapa:** {'Sim' if broker_state.get('execution_enabled') else 'Nao'}")

status_options = ["RUNNING", "PAUSED", "STOPPED"]
mode_options = ["Automatico", "Semi-automatico"]

status_index = status_options.index(state["bot_status"]) if state.get("bot_status") in status_options else 1
mode_index = mode_options.index(state["bot_mode"]) if state.get("bot_mode") in mode_options else 0

status = st.radio("Status do bot", status_options, index=status_index, format_func=bot_status_label, horizontal=True)
mode = st.selectbox("Modo do bot", mode_options, index=mode_index)

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("Salvar status", use_container_width=True):
        state["bot_status"] = status
        state["bot_mode"] = mode
        save_bot_state(state)
        log_event("INFO", f"Status atualizado para {status} / {mode}")
        st.success("Status salvo.")

with c2:
    if st.button("Executar ciclo agora", use_container_width=True):
        result = run_trader_cycle()
        cycle = result.get("cycle_result", {}) or {}
        risk = result.get("risk", {}) or {}
        blocked = bool(risk.get("daily_loss_block_active", False))
        st.success(
            f"Ciclo executado. Trades feitos: {int(cycle.get('trades_executed', 0) or 0)} | "
            f"Feed: {market_data_status_label(cycle.get('market_data_status'))} | "
            f"Trava diaria: {'BLOQUEADA' if blocked else 'LIBERADA'}"
        )

with c3:
    if st.button("Pausar entradas", use_container_width=True):
        state["bot_status"] = "PAUSED"
        save_bot_state(state)
        st.warning("Bot pausado.")

with c4:
    if st.button("Reset geral", use_container_width=True):
        reset_state()
        st.error("Estado resetado.")

st.info("Ligado abre e gerencia posicoes. Pausado nao abre novas posicoes. Parado paralisa tudo.")
