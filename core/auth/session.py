from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import streamlit as st

SESSION_USER_KEY = "auth_user"
SESSION_LAST_ACTIVITY_KEY = "auth_last_activity"
SESSION_TIMEOUT_MINUTES = 30


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def auth_required() -> bool:
    explicit = os.getenv("AUTH_REQUIRED")
    if explicit not in (None, ""):
        return _truthy(explicit)

    for key in ("APP_ENV", "ENVIRONMENT", "STREAMLIT_ENV"):
        if str(os.getenv(key, "")).strip().lower() in {"prod", "production"}:
            return True

    if str(os.getenv("RAILWAY_ENVIRONMENT", "")).strip():
        return True
    if str(os.getenv("RAILWAY_PROJECT_ID", "")).strip():
        return True

    return False


def _sanitize_user(user: dict | None) -> dict | None:
    if not isinstance(user, dict):
        return None

    return {
        "username": str(user.get("username", "")).strip(),
        "role": str(user.get("role", "user")).strip().lower() or "user",
        "disabled": bool(user.get("disabled", False)),
        "auth_mode": str(user.get("auth_mode", "session")).strip() or "session",
    }


def _dev_user() -> dict:
    return {
        "username": "local-dev",
        "role": "admin",
        "disabled": False,
        "auth_mode": "dev-bypass",
    }


def _session_timed_out() -> bool:
    last_seen = _parse_iso(st.session_state.get(SESSION_LAST_ACTIVITY_KEY))
    if last_seen is None:
        return False

    timeout_at = last_seen + timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    return datetime.now(timezone.utc) > timeout_at


def login_user(user: dict) -> dict:
    safe_user = _sanitize_user(user)
    if not safe_user or not safe_user["username"]:
        raise ValueError("Invalid user.")
    if safe_user["disabled"]:
        raise ValueError("This user is disabled.")

    safe_user["auth_mode"] = "session"
    st.session_state[SESSION_USER_KEY] = safe_user
    st.session_state[SESSION_LAST_ACTIVITY_KEY] = _utc_now_iso()
    return safe_user


def logout_user() -> None:
    st.session_state.pop(SESSION_USER_KEY, None)
    st.session_state.pop(SESSION_LAST_ACTIVITY_KEY, None)


def has_active_session() -> bool:
    session_user = _sanitize_user(st.session_state.get(SESSION_USER_KEY))
    if not session_user:
        return False

    if _session_timed_out():
        logout_user()
        return False

    st.session_state[SESSION_LAST_ACTIVITY_KEY] = _utc_now_iso()
    return True


def get_current_user() -> dict | None:
    if has_active_session():
        session_user = _sanitize_user(st.session_state.get(SESSION_USER_KEY))
        if session_user and session_user.get("auth_mode") == "session":
            try:
                from .users_store import get_user

                persisted_user = get_user(session_user["username"])
            except Exception:
                persisted_user = None

            if not persisted_user or persisted_user.get("disabled", False):
                logout_user()
                return None if auth_required() else _dev_user()

            session_user["role"] = str(persisted_user.get("role", session_user["role"])).lower()
            st.session_state[SESSION_USER_KEY] = session_user

        return session_user

    if auth_required():
        return None

    return _dev_user()


def is_authenticated() -> bool:
    return get_current_user() is not None
