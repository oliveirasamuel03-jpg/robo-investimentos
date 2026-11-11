"""Conservative operator feedback: failures never become successful commands."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest
from tests.conftest import load_module

ROOT = Path(__file__).resolve().parents[1]


class UI:
    def __init__(self):
        self.session_state = {}
        self.messages = []

    def error(self, message):
        self.messages.append(("error", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def info(self, message):
        self.messages.append(("info", message))

    def success(self, message):
        self.messages.append(("success", message))


def test_conflict_does_not_retry_or_claim_success(isolated_storage):
    persistence = load_module("core.persistence")
    helper = load_module("core.ui_mutations")
    ui, calls = UI(), []
    def save():
        calls.append(True)
        raise persistence.PersistenceConflictError("private failure detail")
    result = helper.run_ui_mutation(save, ui=ui, action="Salvar configurações")
    assert not result.confirmed and result.error_code == "CONFLICT"
    assert len(calls) == 1
    assert ui.session_state[helper.PENDING_INTENT_KEY]["action"] == "Salvar configurações"
    message = " ".join(text for _, text in ui.messages)
    assert "estado mudou" in message and "Recarregue" in message and "revise" in message
    assert "private failure detail" not in message
    assert not any(kind == "success" for kind, _ in ui.messages)


@pytest.mark.parametrize("sqlstate", [None, "55P03"])
def test_failure_records_label_without_credentials_or_retry(isolated_storage, sqlstate):
    helper = load_module("core.ui_mutations")
    ui, calls = UI(), []
    class DBFailure(RuntimeError):
        pass
    error = DBFailure("postgresql://secret-user:secret-password@private-host")
    error.sqlstate = sqlstate
    def fail():
        calls.append(True)
        raise error
    result = helper.run_ui_mutation(fail, ui=ui, action="Iniciar robô")
    assert not result.confirmed
    assert result.error_code == ("LOCK_TIMEOUT" if sqlstate else "WRITE_FAILED")
    assert len(calls) == 1
    helper.render_pending_intent(ui)
    assert len(calls) == 1
    assert "secret-password" not in str(ui.messages)
    assert "secret-password" not in str(ui.session_state)
    assert not any(kind == "success" for kind, _ in ui.messages)


def test_success_clears_only_its_pending_intent(isolated_storage):
    helper = load_module("core.ui_mutations")
    ui = UI()
    ui.session_state[helper.PENDING_INTENT_KEY] = {"action": "Salvar configurações", "status": "UNCONFIRMED"}
    result = helper.run_ui_mutation(lambda: {"saved": True}, ui=ui, action="Salvar configurações")
    assert result.confirmed and result.value == {"saved": True}
    assert helper.PENDING_INTENT_KEY not in ui.session_state
    assert ui.messages == []  # Only the caller may announce success after checking result.


@pytest.mark.parametrize("status,persisted,acknowledged", [
    ("UNCONFIRMED", False, False), ("PERSISTED", True, False), ("ACKNOWLEDGED", True, True),
])
def test_stop_receipt_never_confuses_persisted_with_acknowledged(
    isolated_storage, monkeypatch, status, persisted, acknowledged
):
    control = load_module("core.stop_control")
    helper = load_module("core.ui_mutations")
    ui, calls = UI(), []
    receipt = SimpleNamespace(request_id="synthetic-request", status=status,
                              persisted=persisted, acknowledged=acknowledged, error_code="")
    def request(**kwargs):
        calls.append(kwargs)
        return receipt
    monkeypatch.setattr(control, "request_stop", request)
    result = helper.request_stop_from_ui(ui=ui, requested_by="synthetic-admin")
    assert len(calls) == 1
    assert result.confirmed == persisted
    assert any("recebida" in text for _, text in ui.messages)
    assert any(kind == "success" for kind, _ in ui.messages) == acknowledged
    if not acknowledged:
        assert helper.PENDING_INTENT_KEY in ui.session_state
    if persisted and not acknowledged:
        assert any("aguarda reconhecimento" in text for _, text in ui.messages)
    if not persisted:
        assert any("Parada NÃO confirmada" in text for _, text in ui.messages)


def test_stop_exception_has_specific_unconfirmed_feedback(isolated_storage, monkeypatch):
    control = load_module("core.stop_control")
    helper = load_module("core.ui_mutations")
    ui, calls = UI(), []
    def request(**kwargs):
        calls.append(True)
        raise OSError("synthetic outage")
    monkeypatch.setattr(control, "request_stop", request)
    assert not helper.request_stop_from_ui(ui=ui, requested_by="synthetic-admin").confirmed
    assert len(calls) == 1
    assert any("Parada NÃO confirmada" in message for _, message in ui.messages)
    assert not any(kind == "success" for kind, _ in ui.messages)


def test_prior_render_snapshot_is_not_replaced_by_current_stop_before_start(isolated_storage):
    helper = load_module("core.ui_mutations")
    ui = UI()
    old = {"bot_status": "RUNNING", "stop_control": {}}
    first = helper.visible_control_snapshot(ui=ui, state=old, key="test")
    assert first == old
    old["bot_status"] = "PAUSED"  # Session marker must be a separate snapshot.
    new = {"bot_status": "STOPPED", "stop_control": {"request_id": "new-stop"}}
    expected = helper.visible_control_snapshot(ui=ui, state=new, key="test")
    assert expected["bot_status"] == "RUNNING"
    assert expected["stop_control"] == {}
    assert ui.session_state["atlas_control_snapshot_test"] == new


@pytest.mark.parametrize("page", ["pages/1_Trader.py", "pages/4_Controle_do_Bot.py"])
def test_actual_stop_button_failure_does_not_display_confirmed_stop(isolated_storage, monkeypatch, page):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    users = load_module("core.auth.users_store")
    user = users.create_user("fixture-stop-admin", "fixture-password", "admin")
    control = load_module("core.stop_control")
    calls = []
    def request(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(request_id="synthetic-failed-stop", status="UNCONFIRMED",
                               persisted=False, acknowledged=False, error_code="SYNTHETIC_FAILURE")
    monkeypatch.setattr(control, "request_stop", request)
    at = AppTest.from_file(str(ROOT / page), default_timeout=30)
    at.session_state["auth_user"] = {**user, "auth_mode": "session"}
    at.session_state["auth_last_activity"] = datetime.now(timezone.utc).isoformat()
    at.run()
    assert not at.exception
    stop = next(button for button in at.button if button.label == "⏹ Parar robô")
    stop.click().run()
    assert not at.exception
    assert len(calls) == 1
    assert any("Parada NÃO confirmada" in item.value for item in at.error)
    assert not any("Parada efetivamente confirmada" in item.value for item in at.success)
    at.run()  # A refresh shows the pending intention, and never resends it.
    assert len(calls) == 1


@pytest.mark.parametrize("page,button_label,success_text", [
    ("pages/1_Trader.py", "Salvar configuração avançada", "Configuração salva."),
    ("pages/4_Controle_do_Bot.py", "Salvar status", "Status persistido."),
])
def test_actual_save_control_conflict_keeps_intent_without_success(
    isolated_storage, monkeypatch, page, button_label, success_text
):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    users = load_module("core.auth.users_store")
    user = users.create_user("fixture-conflict-admin", "fixture-password", "admin")
    store = load_module("core.state_store")
    persistence = load_module("core.persistence")
    at = AppTest.from_file(str(ROOT / page), default_timeout=30)
    at.session_state["auth_user"] = {**user, "auth_mode": "session"}
    at.session_state["auth_last_activity"] = datetime.now(timezone.utc).isoformat()
    at.run()
    assert not at.exception
    before = store.load_bot_state()
    calls = []
    def conflict(state):
        calls.append(True)
        raise persistence.PersistenceConflictError("synthetic concurrent state change")
    monkeypatch.setattr(store, "save_bot_state", conflict)
    next(button for button in at.button if button.label == button_label).click().run()
    assert not at.exception
    assert len(calls) == 1
    assert any("estado mudou" in item.value for item in at.error)
    assert not any(success_text in item.value for item in at.success)
    after = store.load_bot_state()
    assert after["trader"] == before["trader"]
    assert after["bot_status"] == before["bot_status"]


def test_start_click_from_page_before_stop_cannot_adopt_new_stop_silently(isolated_storage):
    store = load_module("core.state_store")
    control = load_module("core.stop_control")
    helper = load_module("core.ui_mutations")
    state = store.load_bot_state()
    ui = UI()
    helper.visible_control_snapshot(ui=ui, state=state, key="stale-start")
    receipt = control.request_stop(requested_by="fixture-operator")
    assert receipt.persisted
    control.acknowledge_stop(worker_id="fixture-worker")
    stopped = store.load_bot_state()
    previous_render = helper.visible_control_snapshot(ui=ui, state=stopped, key="stale-start")
    result = helper.run_ui_mutation(
        lambda: control.start_robot(previous_render, requested_by="fixture-operator"),
        ui=ui, action="Iniciar robô",
    )
    assert not result.confirmed and result.error_code == "CONFLICT"
    assert store.load_bot_state()["bot_status"] == "STOPPED"


def test_stop_status_requires_acknowledgement_evidence_and_stopped_state(isolated_storage):
    helper = load_module("core.ui_mutations")
    ui = UI()
    state = {"bot_status": "STOPPED", "stop_control": {
        "request_id": "synthetic-stop", "status": "ACKNOWLEDGED", "acknowledged_at": "",
    }}
    helper.render_stop_status(ui=ui, state=state)
    assert not any(kind == "success" for kind, _ in ui.messages)
    state["stop_control"]["acknowledged_at"] = "2026-11-11T00:00:00+00:00"
    ui.session_state[helper.PENDING_INTENT_KEY] = {"action": "Parar robô", "status": "PERSISTED"}
    helper.render_stop_status(ui=ui, state=state)
    assert any(kind == "success" for kind, _ in ui.messages)
    assert helper.PENDING_INTENT_KEY not in ui.session_state
    ui.messages.clear()
    state["bot_status"] = "RUNNING"
    helper.render_stop_status(ui=ui, state=state)
    assert not any(kind == "success" for kind, _ in ui.messages)


@pytest.mark.parametrize("result,expected", [
    ({"status": "STOPPED", "committed": True}, "COMMITTED_BEFORE_STOP"),
    ({"status": "RUNNING", "committed": True}, "COMMITTED"),
    ({"status": "STOPPED", "committed": False}, "NOT_COMMITTED"),
    ({"status": "STOPPED"}, "NOT_COMMITTED"),
    ({"status": "RUNNING"}, "UNKNOWN"),
])
def test_later_stop_never_relabels_committed_cycle_as_aborted(isolated_storage, result, expected):
    helper = load_module("core.ui_mutations")
    assert helper.classify_cycle_feedback(result) == expected


@pytest.mark.parametrize("page,button_label", [
    ("pages/1_Trader.py", "🔁 Rodar agora"),
    ("pages/4_Controle_do_Bot.py", "Executar ciclo agora"),
])
def test_actual_cycle_feedback_preserves_commit_before_stop(
    isolated_storage, monkeypatch, page, button_label
):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    users = load_module("core.auth.users_store")
    user = users.create_user("fixture-commit-admin", "fixture-password", "admin")
    at = AppTest.from_file(str(ROOT / page), default_timeout=30)
    at.session_state["auth_user"] = {**user, "auth_mode": "session"}
    at.session_state["auth_last_activity"] = datetime.now(timezone.utc).isoformat()
    at.run()
    assert not at.exception
    import engines.trader_engine as trader
    calls = []
    def committed_then_stopped():
        calls.append(True)
        return {"status": "STOPPED", "committed": True, "cycle_result": {"trades_executed": 1}}
    monkeypatch.setattr(trader, "run_trader_cycle", committed_then_stopped)
    next(button for button in at.button if button.label == button_label).click().run()
    assert not at.exception
    assert len(calls) == 1
    assert any("Ciclo PAPER confirmado antes" in item.value for item in at.warning)
    assert any("não cancela esse commit retroativamente" in item.value for item in at.warning)
    assert not any("nenhuma nova entrada" in item.value for item in at.info)
