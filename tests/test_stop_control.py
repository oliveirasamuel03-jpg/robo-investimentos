"""Operator request persistence is distinct from worker acknowledgement."""
from concurrent.futures import ThreadPoolExecutor
import importlib
from threading import Barrier

import pytest


def _modules():
    return (importlib.import_module("core.state_store"),
            importlib.import_module("core.stop_control"),
            importlib.import_module("core.persistence"))


def test_stop_receipt_requires_worker_acknowledgement(isolated_storage):
    store, stop, _ = _modules()
    receipt = stop.request_stop("fixture-operator")
    assert receipt.status == "PERSISTED" and receipt.persisted and not receipt.acknowledged
    state = store.load_bot_state()
    assert state["bot_status"] == "STOPPED"
    assert state["stop_control"]["request_id"] == receipt.request_id
    assert state["stop_control"]["requested_at"]
    assert state["stop_control"]["persisted_at"]
    assert not state["stop_control"]["acknowledged_at"]
    ack = stop.acknowledge_stop("fixture-worker")
    assert ack.status == "ACKNOWLEDGED" and ack.persisted and ack.acknowledged
    assert ack.request_id == receipt.request_id
    state = store.load_bot_state()
    assert state["stop_control"]["worker_id"] == "fixture-worker"
    assert [event["event"] for event in state["stop_control"]["events"]] == ["PERSISTED", "ACKNOWLEDGED"]


def test_repeated_stop_and_ack_keep_identity_and_no_duplicate_audit(isolated_storage):
    store, stop, _ = _modules()
    first = stop.request_stop()
    assert stop.request_stop().request_id == first.request_id
    stop.acknowledge_stop()
    assert stop.request_stop().request_id == first.request_id
    stop.acknowledge_stop()
    assert len(store.load_bot_state()["stop_control"]["events"]) == 2


@pytest.mark.parametrize("phase", ["PERSISTED", "ACKNOWLEDGED"])
def test_reset_preserves_stopped_and_its_audit(isolated_storage, phase):
    store, stop, _ = _modules()
    receipt = stop.request_stop()
    if phase == "ACKNOWLEDGED":
        stop.acknowledge_stop()
    before = store.load_bot_state()["stop_control"]
    reset = store.reset_state()
    assert reset["bot_status"] == "STOPPED"
    assert reset["stop_control"] == before
    assert reset["stop_control"]["request_id"] == receipt.request_id


def test_legacy_stopped_reset_and_restart_are_preserved(isolated_storage):
    store, stop, _ = _modules()
    state = store.load_bot_state()
    state["bot_status"] = "STOPPED"
    store.save_bot_state(state)
    store.reset_state()
    # Importing a new worker/state module never changes the persisted intention.
    store = importlib.reload(store)
    assert store.load_bot_state()["bot_status"] == "STOPPED"
    assert stop.start_robot(store.load_bot_state())["bot_status"] == "RUNNING"


@pytest.mark.parametrize("action", ["save", "update", "remove_control"])
def test_generic_saves_cannot_clear_a_confirmed_stop(isolated_storage, action):
    store, stop, p = _modules()
    stop.request_stop()
    stop.acknowledge_stop()
    state = store.load_bot_state()
    if action == "remove_control":
        state.pop("stop_control")
        operation = lambda: store.save_bot_state(state)
    elif action == "save":
        state["bot_status"] = "RUNNING"
        operation = lambda: store.save_bot_state(state)
    else:
        operation = lambda: store.update_bot_state(lambda row: {**row, "bot_status": "PAUSED"})
    with pytest.raises(p.PersistenceConflictError):
        operation()
    assert store.load_bot_state()["bot_status"] == "STOPPED"


def test_start_requires_latest_control_revision_and_ack(isolated_storage):
    store, stop, p = _modules()
    old = store.load_bot_state()
    stop.request_stop()
    with pytest.raises(p.PersistenceConflictError, match="changed"):
        stop.start_robot(old)
    with pytest.raises(p.PersistenceConflictError, match="pending"):
        stop.start_robot(store.load_bot_state())
    before_ack = store.load_bot_state()
    stop.acknowledge_stop()
    with pytest.raises(p.PersistenceConflictError, match="changed"):
        stop.start_robot(before_ack)
    result = stop.start_robot(store.load_bot_state(), "fixture", bot_mode="Semi-automatico")
    assert result["bot_status"] == "RUNNING" and result["bot_mode"] == "Semi-automatico"
    assert result["stop_control"]["status"] == "RESUMED"
    assert result["stop_control"]["events"][-1]["event"] == "RESUMED"


def test_heartbeat_does_not_invalidate_control_revision_start(isolated_storage):
    store, stop, _ = _modules()
    stop.request_stop()
    stop.acknowledge_stop()
    displayed = store.load_bot_state()
    store.update_worker_heartbeat("online")
    result = stop.start_robot(displayed)
    assert result["bot_status"] == "RUNNING" and result["worker_status"] == "online"


def test_stop_failure_is_unconfirmed_and_has_no_retry(isolated_storage, monkeypatch):
    store, stop, _ = _modules()
    before = store.load_bot_state()
    attempts = []
    def fail(*args, **kwargs):
        attempts.append(True)
        raise OSError("synthetic failure; must not be copied into receipt")
    monkeypatch.setattr(store, "update_bot_state", fail)
    result = stop.request_stop()
    assert result.status == "UNCONFIRMED" and not result.persisted and not result.acknowledged
    assert result.error_code == "OSError" and len(attempts) == 1
    assert store.load_bot_state() == before


def test_readback_failure_does_not_claim_persistence(isolated_storage, monkeypatch):
    store, stop, _ = _modules()
    store.load_bot_state()
    original = store.load_bot_state
    def fail():
        raise OSError("synthetic failed verification")
    monkeypatch.setattr(store, "load_bot_state", fail)
    receipt = stop.request_stop()
    assert receipt.status == "UNCONFIRMED" and not receipt.persisted
    assert original()["bot_status"] == "STOPPED"  # A failure is not proof of rollback either.


def test_stop_and_heartbeat_are_serialized_without_loss(isolated_storage):
    store, stop, _ = _modules()
    state = store.load_bot_state()
    state["cash"] = 1000
    store.save_bot_state(state)
    barrier = Barrier(2)
    def request():
        barrier.wait(5)
        return stop.request_stop()
    def heartbeat():
        barrier.wait(5)
        store.update_worker_heartbeat("online")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(request), pool.submit(heartbeat)
        assert first.result(10).persisted
        second.result(10)
    result = store.load_bot_state()
    assert result["bot_status"] == "STOPPED" and result["worker_status"] == "online"
    assert result["cash"] == 1000


def test_stop_wins_over_stale_config_snapshot(isolated_storage):
    store, stop, p = _modules()
    old = store.load_bot_state()
    old["fixture_configuration"] = "old-form"
    stop.request_stop()
    with pytest.raises(p.PersistenceConflictError):
        store.save_bot_state(old)
    result = store.load_bot_state()
    assert result["bot_status"] == "STOPPED" and "fixture_configuration" not in result


def test_ack_candidate_has_no_io_or_effect_before_save(isolated_storage):
    store, stop, _ = _modules()
    stop.request_stop()
    state = store.load_bot_state()
    candidate = stop.mutate_acknowledgement(state, "fixture-worker")
    assert state["stop_control"]["status"] == "PERSISTED"
    assert candidate["stop_control"]["status"] == "ACKNOWLEDGED"
    assert store.load_bot_state()["stop_control"]["status"] == "PERSISTED"
    store.save_bot_state(candidate)
    assert store.load_bot_state()["stop_control"]["status"] == "ACKNOWLEDGED"


def test_status_exposes_cycle_phase_without_claiming_liveness(isolated_storage):
    store, stop, _ = _modules()
    store.update_bot_state(lambda state: {**state, "cycle_activity": {"cycle_id": "fixture", "phase": "PREPARING"}})
    stop.request_stop()
    status = stop.stop_control_status(store.load_bot_state())
    assert status["cycle_activity"]["phase"] == "PREPARING"
    assert status["persisted"] and not status["acknowledged"]


@pytest.mark.parametrize("action", ["save", "update"])
def test_raw_persistence_cannot_resume_confirmed_stop(isolated_storage, action):
    store, stop, p = _modules()
    stop.request_stop()
    stop.acknowledge_stop()
    current = store.load_bot_state()
    if action == "save":
        current["bot_status"] = "RUNNING"
        operation = lambda: p.save_json_state("bot_state", current, store.BOT_STATE_FILE)
    else:
        operation = lambda: p.update_json_state("bot_state", dict, store.BOT_STATE_FILE,
                                                lambda row: {**row, "bot_status": "RUNNING"})
    with pytest.raises(p.PersistenceConflictError):
        operation()
    assert store.load_bot_state()["bot_status"] == "STOPPED"
    assert store.load_bot_state()["stop_control"]["status"] == "ACKNOWLEDGED"


def test_raw_persistence_cannot_clear_stop_audit_on_reset(isolated_storage):
    store, stop, p = _modules()
    stop.request_stop()
    before = store.load_bot_state()["stop_control"]
    with pytest.raises(p.PersistenceConflictError):
        p.update_json_state("bot_state", dict, store.BOT_STATE_FILE,
                            lambda row: {"bot_status": "STOPPED"})
    assert store.load_bot_state()["stop_control"] == before


def test_start_with_selected_mode_rejects_stale_mode(isolated_storage):
    store, stop, p = _modules()
    stop.request_stop()
    stop.acknowledge_stop()
    displayed = store.load_bot_state()
    assert displayed["bot_mode"] == "Automatico"
    store.update_bot_state(lambda row: {**row, "bot_mode": "Semi-automatico"})
    with pytest.raises(p.PersistenceConflictError, match="Bot mode changed"):
        stop.start_robot(displayed, bot_mode="Automatico")
    latest = store.load_bot_state()
    assert latest["bot_status"] == "STOPPED" and latest["bot_mode"] == "Semi-automatico"
