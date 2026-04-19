# -*- coding: utf-8 -*-
import streamlit as st
from paper_trading_engine import run_paper_cycle

st.set_page_config(page_title="Robô Trader", layout="wide")

st.title("🤖 Robô de Investimentos")

if st.button("Rodar ciclo"):
    result = run_paper_cycle()

    st.subheader("Resultado")
    st.write(result["report"])

    st.subheader("Equity")
    st.line_chart(result["equity"])

    st.subheader("Trades")
    st.write(result["trades"])
