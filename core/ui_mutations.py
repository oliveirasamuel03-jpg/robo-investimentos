"""Conservative mutation feedback shared by Streamlit controls.

No automatic retry, overwrite or credential-bearing exception is rendered. The
pending marker records only an action label; it never resubmits an operation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

PENDING_INTENT_KEY = "atlas_pending_operator_intent"


@dataclass(frozen=True)
class MutationResult:
    confirmed: bool
    value: Any = None
    error_code: str = ""


def _remember(ui, action: str, status: str) -> None:
    ui.session_state[PENDING_INTENT_KEY] = {"action": action, "status": status}


def run_ui_mutation(operation: Callable[[], Any], *, ui, action: str) -> MutationResult:
    """Execute once and report whether the call completed; success is caller-owned.

    The operation must return only after its persistence contract has completed.
    A returned stop receipt needs the additional persisted/acknowledged checks in
    request_stop_from_ui; returning from a request is not proof of effective STOP.
    """
    from core.persistence import PersistenceConflictError

    try:
        value = operation()
    except PersistenceConflictError:
        _remember(ui, action, "UNCONFIRMED")
        ui.error(
            f"{action}: o estado mudou. A operação não foi confirmada. "
            "Recarregue os dados e revise sua intenção antes de tentar novamente. "
            "Nenhuma repetição automática foi executada."
        )
        return MutationResult(False, error_code="CONFLICT")
    except Exception as exc:
        _remember(ui, action, "UNCONFIRMED")
        code = "LOCK_TIMEOUT" if getattr(exc, "sqlstate", None) == "55P03" else "WRITE_FAILED"
        ui.error(
            f"{action}: operação não confirmada por falha ou contenção. "
            "O resultado persistido pode ser desconhecido. Recarregue e verifique "
            "o estado antes de tentar novamente. Não houve repetição automática."
        )
        return MutationResult(False, error_code=code)
    if ui.session_state.get(PENDING_INTENT_KEY, {}).get("action") == action:
        ui.session_state.pop(PENDING_INTENT_KEY, None)
    return MutationResult(True, value)


def render_pending_intent(ui) -> None:
    pending = ui.session_state.get(PENDING_INTENT_KEY)
    if pending:
        ui.warning(
            f"Intenção pendente: {pending['action']}. "
            "Confira o estado persistido e revise a operação. "
            "Esta indicação não reenvia comandos e não confirma a parada."
        )


def request_stop_from_ui(*, ui, requested_by: str) -> MutationResult:
    from core.stop_control import request_stop

    _remember(ui, "Parar robô", "RECEIVED")
    ui.info("Solicitação de parada recebida. Aguardando confirmação de persistência.")
    result = run_ui_mutation(
        lambda: request_stop(requested_by=requested_by), ui=ui, action="Parar robô"
    )
    if not result.confirmed:
        ui.error("Parada NÃO confirmada. Não considere o robô parado.")
        return result
    receipt = result.value
    if not receipt.persisted:
        _remember(ui, "Parar robô", "UNCONFIRMED")
        ui.error(
            "Parada NÃO confirmada por falha ou contenção. "
            "Recarregue e verifique o protocolo; não considere o robô parado."
        )
        return MutationResult(False, receipt, receipt.error_code or "STOP_UNCONFIRMED")
    if receipt.acknowledged:
        ui.success(f"Parada efetivamente confirmada pelo protocolo. Solicitação: {receipt.request_id}.")
    else:
        _remember(ui, "Parar robô", "PERSISTED")
        ui.warning(
            f"Solicitação persistida: {receipt.request_id}. "
            "A parada efetiva ainda aguarda reconhecimento do worker. "
            "Pode haver ciclo em andamento; não há cancelamento retroativo de commits."
        )
    return MutationResult(True, receipt)


def render_stop_status(*, ui, state: dict) -> None:
    from core.stop_control import stop_control_status

    control = stop_control_status(state)
    status = control.get("status")
    activity = control.get("cycle_activity", {}) or {}
    if activity.get("phase") == "PREPARING":
        ui.info(
            "Último registro do worker: aquisição de dados de um ciclo. "
            "Esse registro não comprova que o processo ainda esteja ativo; "
            "a confirmação de parada é apresentada separadamente."
        )
    if operational_status_label(state) == "Parada confirmada":
        if ui.session_state.get(PENDING_INTENT_KEY, {}).get("action") == "Parar robô":
            ui.session_state.pop(PENDING_INTENT_KEY, None)
        ui.success(
            f"Parada confirmada pelo worker: {control.get('request_id') or '-'} "
            f"em {control.get('acknowledged_at') or '-'}."
        )
    elif control.get("persisted") and control.get("stopped"):
        ui.warning(
            f"Parada solicitada e persistida: {control.get('request_id') or '-'}. "
            "Reconhecimento do worker pendente; ciclo pode estar em andamento."
        )
    elif state.get("bot_status") == "STOPPED":
        ui.warning("STOPPED persistido; reconhecimento da parada pelo protocolo ainda não verificado.")


def visible_control_snapshot(*, ui, state: dict, key: str):
    """Keep the state actually shown before a click-triggered Streamlit rerun.

    START validates this prior control intent rather than silently adopting a
    STOP received while the page was open. This marker never replays a command.
    """
    session_key = f"atlas_control_snapshot_{key}"
    previous = ui.session_state.get(session_key, state)
    ui.session_state[session_key] = deepcopy(state)
    return previous


def classify_cycle_feedback(result: dict) -> str:
    """A later STOP cannot turn an already committed cycle into an aborted one."""
    committed = result.get("committed")
    if committed is True:
        return "COMMITTED_BEFORE_STOP" if result.get("status") == "STOPPED" else "COMMITTED"
    if committed is False or result.get("status") == "STOPPED":
        return "NOT_COMMITTED"
    return "UNKNOWN"


def operational_status_label(state: dict) -> str:
    """Describe recorded control evidence without treating STOPPED as an ACK."""
    status = str(state.get("bot_status") or "").upper()
    if status == "RUNNING":
        return "Ligado"
    if status == "PAUSED":
        return "Pausado"
    if status == "STOPPED":
        control = state.get("stop_control", {}) or {}
        if (isinstance(control, dict) and control.get("status") == "ACKNOWLEDGED"
                and isinstance(control.get("request_id"), str) and control["request_id"].strip()
                and isinstance(control.get("acknowledged_at"), str) and control["acknowledged_at"].strip()):
            return "Parada confirmada"
        return "Parada pendente de confirmação"
    return "Estado desconhecido"
