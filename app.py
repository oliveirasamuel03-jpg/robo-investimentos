import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# =====================
# CONFIG
# =====================
st.set_page_config(page_title="Invest Pro", layout="wide")

# =====================
# LOGIN BÁSICO
# =====================
USUARIO = "admin"
SENHA = "1234"

if "logado" not in st.session_state:
    st.session_state.logado = False

if not st.session_state.logado:

    st.title("🔐 Acesso ao Robô Investidor")

    usuario_input = st.text_input("Usuário")
    senha_input = st.text_input("Senha", type="password")

    if st.button("Entrar"):

        if usuario_input == USUARIO and senha_input == SENHA:
            st.session_state.logado = True
            st.success("Login realizado com sucesso!")
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos")

    st.stop()

# =====================
# RSI
# =====================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# =====================
# ATIVOS
# =====================
ativos = [
    "VALE3.SA","PETR4.SA","ITUB4.SA","BBDC4.SA",
    "BBAS3.SA","WEGE3.SA","MGLU3.SA","RENT3.SA"
]

# =====================
# APP PRINCIPAL
# =====================
st.title("📊 Invest Pro - Dashboard")

if st.button("🚀 Rodar Análise"):

    resultados = []
    dados_map = {}

    # =====================
    # ANÁLISE
    # =====================
    for ativo in ativos:

        dados = yf.download(ativo, start="2020-01-01")

        if dados.empty:
            continue

        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = dados.columns.droplevel(1)

        dados["media9"] = dados["Close"].rolling(9).mean()
        dados["media21"] = dados["Close"].rolling(21).mean()
        dados["rsi"] = rsi(dados["Close"])

        dados["sinal"] = 0
        dados.loc[dados["media9"] > dados["media21"], "sinal"] = 1
        dados.loc[dados["media9"] < dados["media21"], "sinal"] = -1

        dinheiro = 1000
        posicao = 0

        for i in range(len(dados)):
            preco = dados["Close"].iloc[i]
            sinal = dados["sinal"].iloc[i]
            r = dados["rsi"].iloc[i]

            if sinal == 1 and dinheiro > 0 and r < 70:
                posicao = dinheiro / preco
                dinheiro = 0

            elif posicao > 0 and (sinal == -1 or r > 75):
                dinheiro = posicao * preco
                posicao = 0

        valor_final = dinheiro + posicao * dados["Close"].iloc[-1]
        lucro = valor_final - 1000

        resultados.append((ativo, lucro))
        dados_map[ativo] = dados

    # =====================
    # RANKING
    # =====================
    resultados.sort(key=lambda x: x[1], reverse=True)

    st.subheader("🏆 Ranking")

    col1, col2, col3 = st.columns(3)

    for i, (ativo, lucro) in enumerate(resultados[:3]):
        col = [col1, col2, col3][i]

        col.metric(
            label=ativo,
            value=f"R$ {lucro:.2f}",
            delta=f"{lucro:.2f}"
        )

    st.divider()

    # =====================
    # SELEÇÃO
    # =====================
    ativo = st.selectbox("📌 Escolha o ativo", list(dados_map.keys()))
    df = dados_map[ativo]

    # =====================
    # GRÁFICO PROFISSIONAL (CANDLESTICK)
    # =====================
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Preço"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["media9"],
        name="Média 9",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["media21"],
        name="Média 21",
        line=dict(color="orange")
    ))

    fig.update_layout(
        template="plotly_dark",
        height=600,
        title=f"{ativo} - Análise Profissional"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================
    # RESULTADO
    # =====================
    st.subheader("📊 Resultado Final")

    for ativo, lucro in resultados:
        cor = "🟢" if lucro > 0 else "🔴"
        st.write(f"{cor} {ativo} → R$ {lucro:.2f}")
