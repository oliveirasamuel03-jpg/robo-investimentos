from __future__ import annotations

from tests.conftest import load_module


def test_users_store_json_fallback_creates_hashed_user(isolated_storage):
    users_store = load_module("core.auth.users_store")

    created = users_store.create_user("admin", "senha-super-forte", "admin")
    stored = users_store.get_user("admin")
    listed = users_store.list_users()

    assert created["username"] == "admin"
    assert stored is not None
    assert stored["role"] == "admin"
    assert stored["password_hash"] != "senha-super-forte"
    assert any(user["username"] == "admin" for user in listed)
    assert users_store.AUTH_USERS_FILE.exists()


def test_bootstrap_admin_uses_env_when_store_is_empty(isolated_storage, monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "bootstrap")
    monkeypatch.setenv("ADMIN_PASSWORD", "senha-bootstrap")
    users_store = load_module("core.auth.users_store")

    created = users_store.ensure_bootstrap_admin()

    assert created is not None
    assert created["username"] == "bootstrap"
