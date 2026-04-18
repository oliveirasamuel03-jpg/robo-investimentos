from __future__ import annotations

import pandas as pd
import streamlit as st

from core.state_store import load_bot_state, save_bot_state
from portfolio.wallet import compute_wallet_snapshot

st.title("Carteira")
state = load_bot_state()
snapshot = compute_wallet_snapshot(state)

c1, c2 = st.columns(2)
with c1:
    new_wallet = st.number_input("Valor total da carteira (R$)", min_value=10.0, value=float(state["wallet_value"]), step=10.0)
with c2:
    reserve_pct = st.slider("Reserva de caixa (%)", min_value=0.0, max_value=50.0, value=float(state.get("reserve_cash_pct", 0.10)) * 100, step=1.0)

if st.button("Salvar carteira", use_container_width=True):
    delta = float(new_wallet) - float(state["wallet_value"])
    state["wallet_value"] = float(new_wallet)
    state["cash"] = round(float(state["cash"]) + delta, 2)
    state["reserve_cash_pct"] = float(reserve_pct) / 100.0
    save_bot_state(state)
    st.success("Carteira atualizada.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Carteira", f"R$ {snapshot['wallet_value']:,.2f}")
k2.metric("Caixa", f"R$ {snapshot['cash']:,.2f}")
k3.metric("Capital em uso", f"R$ {snapshot['capital_in_use']:,.2f}")
k4.metric("PnL realizado", f"R$ {snapshot['realized_pnl']:,.2f}")

st.subheader("Posições da Plataforma")
positions = state.get("positions", [])
if positions:
    st.dataframe(pd.DataFrame(positions), use_container_width=True)
else:
    st.info("Sem posições registradas.")
