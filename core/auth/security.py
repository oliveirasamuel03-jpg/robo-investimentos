from __future__ import annotations

try:
    import bcrypt
except Exception:
    bcrypt = None


def _require_bcrypt() -> None:
    if bcrypt is None:
        raise RuntimeError("The 'bcrypt' dependency is required for authentication.")


def hash_password(password: str) -> str:
    _require_bcrypt()

    if not isinstance(password, str) or not password.strip():
        raise ValueError("Password cannot be empty.")

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    _require_bcrypt()

    if not isinstance(password, str) or not isinstance(hashed, str):
        return False

    password_bytes = password.encode("utf-8")
    hashed_bytes = hashed.encode("utf-8")

    try:
        return bool(bcrypt.checkpw(password_bytes, hashed_bytes))
    except Exception:
        return False
