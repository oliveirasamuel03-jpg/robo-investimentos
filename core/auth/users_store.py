from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Callable

from core.config import AUTH_USERS_FILE
from core.persistence import PersistenceDataError, load_json_state, update_json_state

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


def _normalize_users_payload(payload: dict) -> dict:
    """Reject corrupt credentials instead of silently deleting users on the next write.

    Existing username/role normalization is retained. Hash algorithms are not
    migrated here; a nonempty hash follows the existing authentication contract.
    """
    users = payload.get("users")
    if not isinstance(users, list):
        raise PersistenceDataError("Invalid users structure; recovery required.")

    normalized_users = []
    usernames = set()
    for raw_user in users:
        if not isinstance(raw_user, dict):
            raise PersistenceDataError("Invalid user record; recovery required.")
        username = str(raw_user.get("username", "")).strip().lower()
        password_hash = raw_user.get("password_hash", "")
        role = str(raw_user.get("role", "user")).strip().lower()
        disabled = raw_user.get("disabled", False)
        if (not username or not isinstance(password_hash, str) or not password_hash.strip()
                or role not in VALID_ROLES or not isinstance(disabled, bool)):
            raise PersistenceDataError("Invalid user credentials or permissions; recovery required.")
        if username in usernames:
            raise PersistenceDataError("Duplicate username; recovery required.")
        usernames.add(username)
        normalized_users.append({
            **raw_user,
            "username": username,
            "password_hash": password_hash.strip(),
            "role": role,
            "disabled": disabled,
            "created_at": str(raw_user.get("created_at", "") or ""),
        })
    return {**payload, "users": normalized_users}


def _load_users_payload() -> dict:
    return _normalize_users_payload(
        load_json_state(AUTH_USERS_NAMESPACE, _default_payload, AUTH_USERS_FILE)
    )


def _update_users_payload(updater: Callable[[dict], dict]) -> dict:
    return update_json_state(
        AUTH_USERS_NAMESPACE,
        _default_payload,
        AUTH_USERS_FILE,
        lambda payload: updater(_normalize_users_payload(payload)),
    )


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


def _new_user(username: str, password: str, role: str) -> dict:
    # Hash outside the storage lock: bcrypt can be expensive.
    return {
        "username": username,
        "password_hash": hash_password(password),
        "role": role,
        "disabled": False,
        "created_at": _utc_now_iso(),
    }


def create_user(username: str, password: str, role: str = "user") -> dict:
    normalized_username = _normalize_username(username)
    normalized_role = _normalize_role(role)
    user = _new_user(normalized_username, password, normalized_role)

    def create(payload: dict) -> dict:
        if any(row.get("username") == normalized_username for row in payload["users"]):
            raise ValueError("User already exists.")
        payload["users"].append(user)
        payload["users"] = sorted(payload["users"], key=lambda row: row["username"])
        return payload

    _update_users_payload(create)
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

    def update(payload: dict) -> dict:
        for user in payload["users"]:
            if user.get("username") != normalized_username:
                continue
            if user.get("role") == "admin" and normalized_role != "admin" and _enabled_admin_count(payload) <= 1 and not user.get("disabled", False):
                raise ValueError("At least one enabled admin user is required.")
            user["role"] = normalized_role
            return payload
        raise ValueError("User not found.")

    payload = _update_users_payload(update)
    return _sanitize_user(next(user for user in payload["users"] if user["username"] == normalized_username))


def set_user_disabled(username: str, disabled: bool) -> dict:
    normalized_username = _normalize_username(username)

    def update(payload: dict) -> dict:
        for user in payload["users"]:
            if user.get("username") != normalized_username:
                continue
            if user.get("role") == "admin" and disabled and _enabled_admin_count(payload) <= 1 and not user.get("disabled", False):
                raise ValueError("At least one enabled admin user is required.")
            user["disabled"] = bool(disabled)
            return payload
        raise ValueError("User not found.")

    payload = _update_users_payload(update)
    return _sanitize_user(next(user for user in payload["users"] if user["username"] == normalized_username))


def ensure_bootstrap_admin() -> dict | None:
    if list_users():
        return None

    username = str(os.getenv("ADMIN_USERNAME") or os.getenv("AUTH_BOOTSTRAP_USERNAME") or "").strip()
    password = str(os.getenv("ADMIN_PASSWORD") or os.getenv("AUTH_BOOTSTRAP_PASSWORD") or "")
    if not username or not password:
        return None

    user = _new_user(_normalize_username(username), password, "admin")
    created = False

    def bootstrap(payload: dict) -> dict:
        nonlocal created
        # Recheck while locked: a different process may have bootstrapped already.
        if not payload["users"]:
            payload["users"].append(user)
            created = True
        return payload

    _update_users_payload(bootstrap)
    return _sanitize_user(user) if created else None
