import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# =====================
# CONFIGURAÇÃO VISUAL
# =====================
st.set_page_config(page_title="Robô Investidor", layout="wide")

st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            color: white;
        }
        .stButton>button {
            background-color: #00c46c;
            color: white;
            border-radius: 10px;
            height: 45px;
            width: 100%;
            font-weight: bold;
        }
        .card {
            background-color: #1c1f26;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Robô Investidor Pro")

# =====================
# RSI
# =====================
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# =====================
# ATIVOS
# =====================
ativos = [
    "VALE3.SA","PETR4.SA","ITUB4.SA","BBDC4.SA",
    "BBAS3.SA","WEGE3.SA","MGLU3.SA","RENT3.SA",
    "PRIO3.SA","EQTL3.SA"
]

# =====================
# BOTÃO PRINCIPAL
# =====================
if st.button("🚀 Rodar Análise"):

    resultados = []
    dados_ativos = {}

    # =====================
    # ANÁLISE
    # =====================
    for ativo in ativos:
        dados = yf.download(ativo, start="2018-01-01")

        if dados.empty:
            continue

        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = dados.columns.droplevel(1)

        close = dados["Close"]

        dados["media9"] = close.rolling(9).mean()
        dados["media21"] = close.rolling(21).mean()
        dados["media50"] = close.rolling(50).mean()
        dados["RSI"] = rsi(close)

        dados["sinal"] = 0
        dados.loc[dados["media9"] > dados["media21"], "sinal"] = 1
        dados.loc[dados["media9"] < dados["media21"], "sinal"] = -1

        dinheiro = 1000
        acoes = 0

        historico = []

        for i in range(len(dados)):
            preco = close.iloc[i]
            sinal = dados["sinal"].iloc[i]
            rsi_val = dados["RSI"].iloc[i]
            media50 = dados["media50"].iloc[i]

            if sinal == 1 and dinheiro > 0 and rsi_val < 70 and preco > media50:
                acoes = dinheiro / preco
                dinheiro = 0

            elif acoes > 0:
                if sinal == -1 or rsi_val > 70:
                    dinheiro = acoes * preco
                    acoes = 0

            historico.append(dinheiro + acoes * preco)

        lucro = historico[-1] - 1000

        resultados.append((ativo, lucro))
        dados_ativos[ativo] = dados

    # =====================
    # RANKING
    # =====================
    resultados.sort(key=lambda x: x[1], reverse=True)

    st.subheader("🏆 Ranking de Ativos")

    cols = st.columns(3)

    for i, (ativo, lucro) in enumerate(resultados[:3]):
        with cols[i]:
            st.markdown(f"""
            <div class="card">
                <h3>{ativo}</h3>
                <h2 style="color: {'#00ff88' if lucro > 0 else '#ff4d4d'}">
                    R$ {lucro:.2f}
                </h2>
            </div>
            """, unsafe_allow_html=True)

    # =====================
    # LISTA COMPLETA
    # =====================
    st.subheader("📋 Todos os Ativos")

    for ativo, lucro in resultados:
        cor = "🟢" if lucro > 0 else "🔴"
        st.write(f"{cor} {ativo} → R$ {lucro:.2f}")

    # =====================
    # GRÁFICO MELHORADO
    # =====================
    st.subheader("📈 Análise do Ativo")

    escolhido = st.selectbox("Escolha um ativo", list(dados_ativos.keys()))
    dados = dados_ativos[escolhido]

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(dados["Close"], label="Preço", linewidth=2)
    ax.plot(dados["media9"], label="Média 9")
    ax.plot(dados["media21"], label="Média 21")

    ax.set_title(f"{escolhido} - Análise Técnica")
    ax.legend()
    ax.grid()

    st.pyplot(fig)
