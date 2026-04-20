from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.config import (
    BROKER_ACCOUNT_ID,
    BROKER_API_KEY,
    BROKER_API_SECRET,
    BROKER_BASE_URL,
    BROKER_HTTP_HEALTHCHECK,
    BROKER_MODE,
    BROKER_PROVIDER,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_provider(raw_provider: str | None) -> str:
    provider = str(raw_provider or "paper").strip().lower()
    return provider or "paper"


def _normalize_mode(raw_mode: str | None) -> str:
    mode = str(raw_mode or "paper").strip().lower()
    return mode if mode in {"paper", "live"} else "paper"


def _default_broker_base_url(provider: str, mode: str) -> str:
    if provider == "alpaca":
        return "https://paper-api.alpaca.markets" if mode == "paper" else "https://api.alpaca.markets"
    if provider == "binance":
        return "https://testnet.binance.vision" if mode == "paper" else "https://api.binance.com"
    return ""


def _mask_value(value: str, visible: int = 4) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if len(raw) <= visible:
        return "*" * len(raw)
    return f"{'*' * max(len(raw) - visible, 0)}{raw[-visible:]}"


def broker_status_label(raw_status: str | None) -> str:
    labels = {
        "paper": "Simulado",
        "configured": "Configurado",
        "ready": "Pronto",
        "missing_credentials": "Credenciais ausentes",
        "guarded": "Bloqueado pelo modo real",
        "error": "Erro",
        "unknown": "Desconhecido",
    }
    return labels.get(str(raw_status or "unknown").strip().lower(), str(raw_status or "Desconhecido"))


def build_broker_runtime_config(security_state: dict | None = None) -> dict[str, Any]:
    security = security_state or {}
    provider = _normalize_provider(BROKER_PROVIDER)
    configured_mode = _normalize_mode(BROKER_MODE)
    real_mode_enabled = bool(security.get("real_mode_enabled", False))
    live_requested = configured_mode == "live" or real_mode_enabled
    effective_mode = "paper" if provider == "paper" else ("live" if real_mode_enabled else configured_mode)
    api_key_configured = bool(str(BROKER_API_KEY or "").strip())
    api_secret_configured = bool(str(BROKER_API_SECRET or "").strip())
    base_url = str(BROKER_BASE_URL or "").strip() or _default_broker_base_url(provider, configured_mode)

    return {
        "provider": provider,
        "configured_mode": configured_mode,
        "effective_mode": effective_mode,
        "real_mode_enabled": real_mode_enabled,
        "live_requested": live_requested,
        "account_id": str(BROKER_ACCOUNT_ID or "").strip(),
        "account_id_masked": _mask_value(BROKER_ACCOUNT_ID),
        "base_url": base_url,
        "api_key_configured": api_key_configured,
        "api_secret_configured": api_secret_configured,
        "http_healthcheck_enabled": bool(BROKER_HTTP_HEALTHCHECK),
    }


def probe_broker_status(security_state: dict | None = None, requested_by: str = "runtime") -> dict[str, Any]:
    runtime = build_broker_runtime_config(security_state)
    provider = runtime["provider"]
    configured_mode = runtime["configured_mode"]
    effective_mode = runtime["effective_mode"]
    real_mode_enabled = bool(runtime["real_mode_enabled"])
    live_requested = bool(runtime["live_requested"])
    api_key_configured = bool(runtime["api_key_configured"])
    api_secret_configured = bool(runtime["api_secret_configured"])

    status = "unknown"
    last_error = ""
    warning = ""
    can_submit_orders = False
    execution_enabled = False

    if provider == "paper":
        status = "paper"
        warning = "Broker em modo simulado. Nenhuma ordem real sera enviada."
        can_submit_orders = False
        execution_enabled = False
    elif not api_key_configured or not api_secret_configured:
        status = "missing_credentials"
        last_error = "Credenciais da corretora nao configuradas por ambiente."
        warning = "Configure BROKER_API_KEY e BROKER_API_SECRET antes de prosseguir para integracao real."
    elif live_requested and not real_mode_enabled:
        status = "guarded"
        warning = "Credenciais configuradas, mas o modo real continua bloqueado pela camada de seguranca."
    else:
        status = "configured" if effective_mode == "paper" else "ready"
        warning = (
            "Integracao preparada; envio real de ordens ainda nao foi habilitado nesta etapa."
            if effective_mode == "live"
            else "Credenciais e provider preparados para evoluir o fluxo sem sair do modo paper."
        )

    return {
        "provider": provider,
        "mode": effective_mode,
        "status": status,
        "last_sync_at": _utc_now_iso(),
        "last_error": last_error,
        "account_id": runtime["account_id_masked"] or runtime["account_id"],
        "requested_by": requested_by,
        "configured_mode": configured_mode,
        "effective_mode": effective_mode,
        "base_url": runtime["base_url"],
        "api_key_configured": api_key_configured,
        "api_secret_configured": api_secret_configured,
        "execution_enabled": execution_enabled,
        "can_submit_orders": can_submit_orders,
        "warning": warning,
    }
