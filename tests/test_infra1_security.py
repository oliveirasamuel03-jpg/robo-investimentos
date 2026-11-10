from datetime import datetime, timedelta, timezone
from pathlib import Path
import importlib

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import load_module

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def reset_streamlit_page_discovery(monkeypatch):
    # Streamlit caches this process-wide. Each AppTest has a different entrypoint.
    from streamlit.runtime.pages_manager import PagesManager
    monkeypatch.setattr(PagesManager, "uses_pages_directory", None)


@pytest.mark.parametrize("value", [None, "", "typo", "true", "1"])
def test_auth_defaults_fail_closed(monkeypatch, value):
    from core.auth.session import auth_required
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    if value is not None:
        monkeypatch.setenv("AUTH_REQUIRED", value)
    assert auth_required()


@pytest.mark.parametrize("key,value", [("APP_ENV", "production"), ("ENVIRONMENT", "prod"),
    ("RAILWAY_PROJECT_ID", "test-project"), ("RAILWAY_ENVIRONMENT", "production")])
def test_production_rejects_auth_bypass(monkeypatch, key, value):
    from core.auth.session import auth_required
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    monkeypatch.setenv(key, value)
    assert auth_required()


def test_local_false_cannot_create_anonymous_admin(monkeypatch):
    import core.auth.session as session
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    monkeypatch.setattr(session.st, "session_state", {})
    assert session.auth_required()
    assert session.get_current_user() is None
    session.st.session_state["auth_user"] = {
        "username": "local-dev", "role": "admin", "auth_mode": "dev-bypass",
    }
    assert session.get_current_user() is None


def test_empty_store_cannot_register_admin_from_login(isolated_storage, monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    at = AppTest.from_file(str(ROOT / "pages/0_Login.py")).run()
    assert not at.exception
    assert not at.text_input
    assert not at.button
    assert "administrador inicial" in at.error[0].value
    assert load_module("core.auth.users_store").list_users() == []


def test_configured_bootstrap_shows_login_without_authenticating(isolated_storage, monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("ADMIN_USERNAME", "fixture-admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "fixture-only-password")
    at = AppTest.from_file(str(ROOT / "pages/0_Login.py")).run()
    assert not at.exception
    assert len(at.text_input) == 2
    assert "auth_user" not in at.session_state
    at.text_input[0].set_value("fixture-admin")
    at.text_input[1].set_value("wrong-password")
    at.button[0].click().run()
    assert not at.exception
    assert "auth_user" not in at.session_state
    assert at.error


def test_session_expiry_and_nonadmin_guard(isolated_storage, monkeypatch):
    import core.auth.session as session
    import core.auth.guards as guards
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    store = load_module("core.auth.users_store")
    store.create_user("fixture-user", "fixture-password", "user")
    monkeypatch.setattr(session.st, "session_state", {})
    session.login_user(store.get_user("fixture-user"))
    assert session.get_current_user()["role"] == "user"
    assert not guards.is_admin()
    session.st.session_state[session.SESSION_LAST_ACTIVITY_KEY] = (
        datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    assert session.get_current_user() is None


@pytest.mark.parametrize("provider", ["paper", "alpaca", "binance"])
@pytest.mark.parametrize("mode", ["paper", "live"])
@pytest.mark.parametrize("real_enabled", [False, True])
def test_broker_never_authorizes_orders(isolated_storage, monkeypatch, provider, mode, real_enabled):
    monkeypatch.setenv("BROKER_PROVIDER", provider)
    monkeypatch.setenv("BROKER_MODE", mode)
    monkeypatch.setenv("BROKER_API_KEY", "fixture-key")
    monkeypatch.setenv("BROKER_API_SECRET", "fixture-secret")
    load_module("core.config")
    broker = load_module("core.broker")
    status = broker.probe_broker_status({"real_mode_enabled": real_enabled})
    assert status["execution_enabled"] is False
    assert status["can_submit_orders"] is False


def test_worker_import_and_simulated_heartbeat(isolated_storage):
    import sys
    sys.modules.pop("workers.trader_worker", None)
    worker = importlib.import_module("workers.trader_worker")
    worker.update_runtime_state("offline-test", 60)
    state = worker.load_bot_state()
    assert state["worker_status"] == "online"
    assert state["worker_heartbeat"]
    assert state["security"]["real_mode_enabled"] is False


@pytest.mark.parametrize("page", ["app.py", "pages/1_Trader.py", "pages/4_Controle_do_Bot.py", "pages/5_Historico.py"])
def test_authenticated_pages_render_offline(isolated_storage, monkeypatch, page):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    store = load_module("core.auth.users_store")
    user = store.create_user("fixture-admin", "fixture-password", "admin")
    at = AppTest.from_file(str(ROOT / page), default_timeout=20)
    at.session_state["auth_user"] = {**user, "auth_mode": "session"}
    at.session_state["auth_last_activity"] = datetime.now(timezone.utc).isoformat()
    at.run()
    assert not at.exception
    assert not any(button.key == "toggle_real_mode" and not button.disabled for button in at.button)
