from __future__ import annotations

import os
from datetime import datetime, timezone

from core.config import AUTH_USERS_FILE
from core.persistence import load_json_state, save_json_state

from .security import hash_password

AUTH_USERS_NAMESPACE = "auth_users_store"
VALID_ROLES = {"admin", "user"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_payload() -> dict:
    return {"users": []}


def _normalize_username(username: str) -> str:
    value = str(username or "").strip().lower()
    if not value:
        raise ValueError("Username cannot be empty.")
    return value


def _normalize_role(role: str) -> str:
    value = str(role or "user").strip().lower()
    if value not in VALID_ROLES:
        raise ValueError("Role must be 'admin' or 'user'.")
    return value


def _load_users_payload() -> dict:
    payload = load_json_state(AUTH_USERS_NAMESPACE, _default_payload, AUTH_USERS_FILE)
    users = payload.get("users", [])
    if not isinstance(users, list):
        users = []

    normalized_users = []
    for raw_user in users:
        if not isinstance(raw_user, dict):
            continue
        username = str(raw_user.get("username", "")).strip().lower()
        password_hash = str(raw_user.get("password_hash", "")).strip()
        role = str(raw_user.get("role", "user")).strip().lower()
        if not username or not password_hash or role not in VALID_ROLES:
            continue

        normalized_users.append(
            {
                "username": username,
                "password_hash": password_hash,
                "role": role,
                "disabled": bool(raw_user.get("disabled", False)),
                "created_at": str(raw_user.get("created_at", "") or ""),
            }
        )

    return {"users": normalized_users}


def _save_users_payload(payload: dict) -> None:
    AUTH_USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_json_state(AUTH_USERS_NAMESPACE, payload, AUTH_USERS_FILE)


def _sanitize_user(user: dict) -> dict:
    return {
        "username": str(user.get("username", "")),
        "role": str(user.get("role", "user")),
        "disabled": bool(user.get("disabled", False)),
        "created_at": str(user.get("created_at", "") or ""),
    }


def _enabled_admin_count(payload: dict) -> int:
    users = payload.get("users", [])
    return sum(1 for user in users if user.get("role") == "admin" and not user.get("disabled", False))


def create_user(username: str, password: str, role: str = "user") -> dict:
    normalized_username = _normalize_username(username)
    normalized_role = _normalize_role(role)
    payload = _load_users_payload()

    if any(user.get("username") == normalized_username for user in payload["users"]):
        raise ValueError("User already exists.")

    user = {
        "username": normalized_username,
        "password_hash": hash_password(password),
        "role": normalized_role,
        "disabled": False,
        "created_at": _utc_now_iso(),
    }
    payload["users"].append(user)
    payload["users"] = sorted(payload["users"], key=lambda row: row["username"])
    _save_users_payload(payload)
    return _sanitize_user(user)


def get_user(username: str) -> dict | None:
    normalized_username = _normalize_username(username)
    payload = _load_users_payload()
    for user in payload["users"]:
        if user.get("username") == normalized_username:
            return dict(user)
    return None


def list_users() -> list[dict]:
    payload = _load_users_payload()
    return [_sanitize_user(user) for user in payload["users"]]


def update_user_role(username: str, role: str) -> dict:
    normalized_username = _normalize_username(username)
    normalized_role = _normalize_role(role)
    payload = _load_users_payload()

    for user in payload["users"]:
        if user.get("username") != normalized_username:
            continue

        if user.get("role") == "admin" and normalized_role != "admin" and _enabled_admin_count(payload) <= 1 and not user.get("disabled", False):
            raise ValueError("At least one enabled admin user is required.")

        user["role"] = normalized_role
        _save_users_payload(payload)
        return _sanitize_user(user)

    raise ValueError("User not found.")


def set_user_disabled(username: str, disabled: bool) -> dict:
    normalized_username = _normalize_username(username)
    payload = _load_users_payload()

    for user in payload["users"]:
        if user.get("username") != normalized_username:
            continue

        if user.get("role") == "admin" and not disabled and user.get("disabled", False):
            user["disabled"] = False
            _save_users_payload(payload)
            return _sanitize_user(user)

        if user.get("role") == "admin" and disabled and _enabled_admin_count(payload) <= 1 and not user.get("disabled", False):
            raise ValueError("At least one enabled admin user is required.")

        user["disabled"] = bool(disabled)
        _save_users_payload(payload)
        return _sanitize_user(user)

    raise ValueError("User not found.")


def ensure_bootstrap_admin() -> dict | None:
    if list_users():
        return None

    username = str(os.getenv("ADMIN_USERNAME") or os.getenv("AUTH_BOOTSTRAP_USERNAME") or "").strip()
    password = str(os.getenv("ADMIN_PASSWORD") or os.getenv("AUTH_BOOTSTRAP_PASSWORD") or "")

    if not username or not password:
        return None

    return create_user(username=username, password=password, role="admin")
