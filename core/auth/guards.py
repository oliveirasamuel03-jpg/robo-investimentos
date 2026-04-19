from __future__ import annotations

import streamlit as st

from .session import (
    SESSION_TIMEOUT_MINUTES,
    auth_required,
    get_current_user,
    has_active_session,
    logout_user,
)


def switch_to_page(page_path: str) -> None:
    if hasattr(st, "switch_page"):
        st.switch_page(page_path)
    st.stop()


def is_admin(user: dict | None = None) -> bool:
    active_user = user or get_current_user()
    if not isinstance(active_user, dict):
        return False
    return str(active_user.get("role", "")).lower() == "admin" and not bool(active_user.get("disabled", False))


def require_auth() -> dict:
    user = get_current_user()
    if user:
        return user

    st.warning("Sua sessao expirou ou voce precisa fazer login.")
    switch_to_page("pages/0_Login.py")
    raise RuntimeError("Authentication redirect failed.")


def require_admin() -> dict:
    user = require_auth()
    if is_admin(user):
        return user

    st.error("Acesso restrito a administradores.")
    st.stop()
    raise RuntimeError("Admin access denied.")


def render_auth_toolbar() -> None:
    user = get_current_user()
    if not user:
        return

    with st.sidebar:
        st.markdown("### Sessao")
        st.caption(f"Usuario: {user['username']}")
        st.caption(f"Perfil: {user['role']}")
        st.caption(f"Timeout: {SESSION_TIMEOUT_MINUTES} min")

        if user.get("auth_mode") == "dev-bypass":
            st.info("Modo dev ativo. Autenticacao opcional.")
        else:
            st.success("Sessao autenticada.")

        if auth_required() or has_active_session():
            if st.button("Logout", key="auth_logout_sidebar", use_container_width=True):
                logout_user()
                switch_to_page("pages/0_Login.py")
