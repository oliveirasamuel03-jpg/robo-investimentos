from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from tests.conftest import load_module


def _simultaneous(*actions):
    barrier = Barrier(len(actions))

    def run(action):
        barrier.wait(timeout=10)
        try:
            return action()
        except ValueError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=len(actions)) as executor:
        return list(executor.map(run, actions))


def _users(isolated_storage, monkeypatch):
    store = load_module("core.auth.users_store")
    monkeypatch.setattr(store, "hash_password", lambda password: "fixture-hash")
    return store


@pytest.mark.parametrize("payload", [
    {},
    {"users": "invalid"},
    {"users": [None]},
    {"users": [{"username": "admin", "password_hash": ""}]},
    {"users": [{"username": "admin", "password_hash": "hash", "role": "owner"}]},
    {"users": [{"username": "admin", "password_hash": "hash", "disabled": "false"}]},
    {"users": [{"username": "admin", "password_hash": 123}]},
    {"users": [{"username": "admin", "password_hash": "hash"},
               {"username": " ADMIN ", "password_hash": "other-hash"}]},
])
def test_semantic_auth_corruption_blocks_reads_writes_and_bootstrap(isolated_storage, monkeypatch, payload):
    store = _users(isolated_storage, monkeypatch)
    store.AUTH_USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps(payload)
    store.AUTH_USERS_FILE.write_text(original)
    monkeypatch.setenv("ADMIN_USERNAME", "fixture-bootstrap")
    monkeypatch.setenv("ADMIN_PASSWORD", "fixture-password")
    for action in (store.list_users, store.ensure_bootstrap_admin,
                   lambda: store.create_user("new", "fixture-password")):
        with pytest.raises(RuntimeError, match="recovery required"):
            action()
        assert store.AUTH_USERS_FILE.read_text() == original


def test_parallel_user_creation_preserves_both_records(isolated_storage, monkeypatch):
    store = _users(isolated_storage, monkeypatch)
    results = _simultaneous(
        lambda: store.create_user("first", "fixture-password"),
        lambda: store.create_user("second", "fixture-password"),
    )
    assert all(isinstance(result, dict) for result in results)
    assert [user["username"] for user in store.list_users()] == ["first", "second"]


def test_parallel_duplicate_user_is_rejected(isolated_storage, monkeypatch):
    store = _users(isolated_storage, monkeypatch)
    results = _simultaneous(
        lambda: store.create_user("same", "fixture-password"),
        lambda: store.create_user(" SAME ", "fixture-password"),
    )
    assert sum(isinstance(result, ValueError) for result in results) == 1
    assert len(store.list_users()) == 1


def test_parallel_admin_mutations_preserve_one_enabled_admin(isolated_storage, monkeypatch):
    store = _users(isolated_storage, monkeypatch)
    store.create_user("first", "fixture-password", "admin")
    store.create_user("second", "fixture-password", "admin")
    results = _simultaneous(
        lambda: store.set_user_disabled("first", True),
        lambda: store.update_user_role("second", "user"),
    )
    assert sum(isinstance(result, ValueError) for result in results) == 1
    assert sum(user["role"] == "admin" and not user["disabled"] for user in store.list_users()) == 1


def test_parallel_bootstrap_rechecks_empty_store_under_lock(isolated_storage, monkeypatch):
    store = _users(isolated_storage, monkeypatch)
    monkeypatch.setenv("ADMIN_USERNAME", "fixture-bootstrap")
    monkeypatch.setenv("ADMIN_PASSWORD", "fixture-password")
    original_list = store.list_users
    barrier = Barrier(2)

    def both_observe_empty():
        users = original_list()
        barrier.wait(timeout=10)
        return users

    monkeypatch.setattr(store, "list_users", both_observe_empty)
    results = _simultaneous(store.ensure_bootstrap_admin, store.ensure_bootstrap_admin)
    assert sum(result is None for result in results) == 1
    assert len(original_list()) == 1


def test_parallel_status_updates_preserve_stopped_and_financial_state(isolated_storage):
    store = load_module("core.state_store")
    state = store.load_bot_state()
    state["bot_status"] = "STOPPED"
    state["positions"] = [{"fixture_position": "unchanged"}]
    state["cash"] = 4321.0
    store.save_bot_state(state)
    _simultaneous(
        lambda: store.update_validation_status({"fixture_marker": "validation"}),
        lambda: store.update_production_status({"fixture_marker": "production"}),
        lambda: store.update_worker_heartbeat("fixture"),
        lambda: store.update_market_data_status({"requested_by": "trader_chart", "provider": "fixture"}),
    )
    after = store.load_bot_state()
    assert after["bot_status"] == "STOPPED"
    assert after["cash"] == 4321.0
    assert after["positions"] == [{"fixture_position": "unchanged"}]
    assert after["validation"]["fixture_marker"] == "validation"
    assert after["production"]["fixture_marker"] == "production"
    assert after["worker_status"] == "fixture"
    assert after["market_data"]["contexts"]["trader_chart"]["provider"] == "fixture"
    assert after["security"]["real_mode_enabled"] is False


def test_explicit_reset_uses_atomic_update(isolated_storage):
    store = load_module("core.state_store")
    state = store.load_bot_state()
    state["fixture_old_state"] = True
    store.save_bot_state(state)
    reset = store.reset_state()
    assert "fixture_old_state" not in reset
    assert "fixture_old_state" not in store.load_bot_state()
    assert reset["security"]["real_mode_enabled"] is False


def test_parallel_csv_initialization_never_overwrites_appends(isolated_storage):
    """Repeated startup races must preserve every successfully appended record."""
    store = load_module("core.state_store")
    # This path starts absent; initialization and append compete for the same file.
    path = isolated_storage / "fixture-concurrent-init.csv"
    columns = ["id", "description"]
    actions = []
    for index in range(16):
        actions.append(lambda: store._ensure_csv(path, columns))
        actions.append(lambda index=index: store.append_table_row(
            path, {"id": index, "description": "fixture, concurrent startup"}, columns=columns,
        ))
    results = _simultaneous(*actions)
    assert all(result is None for result in results)
    rows = store.read_table(path, columns=columns)
    assert len(rows) == 16
    assert sorted(rows["id"].tolist()) == list(range(16))
    assert set(rows["description"]) == {"fixture, concurrent startup"}
