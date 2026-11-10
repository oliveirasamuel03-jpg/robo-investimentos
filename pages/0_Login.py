from __future__ import annotations

import streamlit as st

from core.auth.guards import switch_to_page
from core.auth.security import verify_password
from core.auth.session import auth_required, has_active_session, login_user
from core.auth.users_store import ensure_bootstrap_admin, get_user, list_users


st.set_page_config(page_title="Login", layout="centered")


def go_to_app() -> None:
    switch_to_page("pages/1_Trader.py")


ensure_bootstrap_admin()

st.title("Login")
st.caption("Acesso protegido da plataforma de trading.")

if has_active_session():
    st.success("Sessao ja autenticada.")
    go_to_app()

auth_is_required = auth_required()
existing_users = list_users()

if not auth_is_required:
    st.info("Modo desenvolvimento: autenticacao opcional. Em producao o login sera obrigatorio.")
    if st.button("Continuar em modo dev", use_container_width=True):
        go_to_app()

st.markdown("---")

if not existing_users:
    st.error("Acesso indisponivel: administrador inicial ainda nao configurado.")
    st.info("O responsavel deve configurar o bootstrap administrativo pelo ambiente protegido do servidor.")
    st.stop()
else:
    with st.form("login_form"):
        username = st.text_input("Usuario")
        password = st.text_input("Senha", type="password")
        submit = st.form_submit_button("Entrar", use_container_width=True)

    if submit:
        try:
            user = get_user(username)
            if not user:
                raise ValueError("Usuario ou senha invalidos.")
            if user.get("disabled", False):
                raise ValueError("Este usuario esta desabilitado.")
            if not verify_password(password, str(user.get("password_hash", ""))):
                raise ValueError("Usuario ou senha invalidos.")

            login_user(user)
            st.success("Login realizado com sucesso.")
            go_to_app()
        except Exception as exc:
            st.error(str(exc))
