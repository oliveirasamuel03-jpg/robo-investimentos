import streamlit as st

st.set_page_config(
    page_title="Robô Investimentos Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

dashboard = st.Page("pages/1_Dashboard.py", title="Dashboard", icon="📊")
investimentos = st.Page("pages/2_Investimentos.py", title="Investimentos", icon="💼")
trader = st.Page("pages/3_Trader.py", title="Trader", icon="🤖")
backtest = st.Page("pages/4_Backtest.py", title="Backtest", icon="🧪")
paper = st.Page("pages/5_Paper_Trading.py", title="Paper Trading", icon="📝")
relatorios = st.Page("pages/6_Relatorios.py", title="Relatórios", icon="📑")

pg = st.navigation(
    {
        "Painel Principal": [dashboard, investimentos],
        "Trader Automático": [trader, backtest, paper, relatorios],
    }
)

pg.run()
