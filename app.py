import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Robô Investidor", layout="wide")

st.title("🤖 Robô de Investimentos com Gráficos")

# ===== RSI =====
def calcular_rsi(dados, periodo=14):
    delta = dados.diff()
    ganho = (delta.where(delta > 0, 0)).rolling(periodo).mean()
    perda = (-delta.where(delta < 0, 0)).rolling(periodo).mean()
    rs = ganho / perda
    return 100 - (100 / (1 + rs))

# ===== AÇÕES =====
ativos = [
    "VALE3.SA","PETR4.SA","ITUB4.SA","BBDC4.SA",
    "BBAS3.SA","WEGE3.SA","MGLU3.SA","RENT3.SA",
    "PRIO3.SA","EQTL3.SA"
]

if st.button("🚀 Rodar Análise"):

    resultados = []
    dados_ativos = {}

    # ===== TESTE DOS ATIVOS =====
    for ativo in ativos:
        dados = yf.download(ativo, start="2018-01-01")

        if dados.empty:
            continue

        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = dados.columns.droplevel(1)

        close = dados['Close']

        dados['media9'] = close.rolling(9).mean()
        dados['media21'] = close.rolling(21).mean()
        dados['media50'] = close.rolling(50).mean()
        dados['RSI'] = calcular_rsi(close)

        dados['sinal'] = 0
        dados.loc[dados['media9'] > dados['media21'], 'sinal'] = 1
        dados.loc[dados['media9'] < dados['media21'], 'sinal'] = -1

        dinheiro = 1000
        acoes = 0
        preco_compra = 0

        historico = []

        for i in range(len(dados)):
            preco = close.iloc[i]
            sinal = dados['sinal'].iloc[i]
            rsi = dados['RSI'].iloc[i]
            media50 = dados['media50'].iloc[i]

            if sinal == 1 and dinheiro > 0 and rsi < 70 and preco > media50:
                acoes = dinheiro / preco
                dinheiro = 0
                preco_compra = preco

            elif acoes > 0:
                if preco < preco_compra * 0.97:
                    dinheiro = acoes * preco
                    acoes = 0
                elif sinal == -1 or rsi > 70:
                    dinheiro = acoes * preco
                    acoes = 0

            valor_total = dinheiro + (acoes * preco)
            historico.append(valor_total)

        valor_final = historico[-1]
        lucro = valor_final - 1000

        # drawdown
        pico = historico[0]
        max_dd = 0

        for v in historico:
            if v > pico:
                pico = v
            dd = (pico - v) / pico
            if dd > max_dd:
                max_dd = dd

        resultados.append((ativo, lucro, max_dd))
        dados_ativos[ativo] = dados

    # ===== RANKING =====
    resultados = sorted(resultados, key=lambda x: x[1], reverse=True)

    st.subheader("📊 Ranking")
    for ativo, lucro, dd in resultados:
        cor = "🟢" if lucro >= 0 else "🔴"
        st.write(f"{cor} {ativo} → Lucro: R$ {lucro:.2f} | DD: {dd*100:.2f}%")

    # ===== FILTRO =====
    selecionados = [x for x in resultados if x[1] > 0 and x[2] < 0.30]

    carteira = []
    empresas = set()

    for ativo, lucro, dd in selecionados:
        emp = ativo[:4]
        if emp not in empresas:
            carteira.append((ativo, lucro, dd))
            empresas.add(emp)
        if len(carteira) == 3:
            break

    st.subheader("🤖 Portfólio Selecionado")
    for ativo, lucro, dd in carteira:
        st.write(f"{ativo} → Lucro: R$ {lucro:.2f}")

    # ===== SIMULAÇÃO =====
    capital = 1000
    por_ativo = capital / len(carteira)
    total = 0

    historico_portfolio = []

    for ativo, _, _ in carteira:
        dados = dados_ativos[ativo]
        close = dados['Close']

        dinheiro = por_ativo
        acoes = 0

        historico = []

        for i in range(len(dados)):
            preco = close.iloc[i]
            sinal = dados['sinal'].iloc[i]
            rsi = dados['RSI'].iloc[i]
            media50 = dados['media50'].iloc[i]

            if sinal == 1 and dinheiro > 0 and rsi < 70 and preco > media50:
                acoes = dinheiro / preco
                dinheiro = 0

            elif acoes > 0:
                if sinal == -1 or rsi > 70:
                    dinheiro = acoes * preco
                    acoes = 0

            valor = dinheiro + (acoes * preco)
            historico.append(valor)

        valor_final = historico[-1]
        total += valor_final

        # soma para gráfico geral
        if not historico_portfolio:
            historico_portfolio = historico
        else:
            historico_portfolio = [x + y for x, y in zip(historico_portfolio, historico)]

        st.write(f"{ativo} → Final: R$ {valor_final:.2f}")

    # ===== RESULTADO =====
    st.subheader("💰 Resultado Final")

    lucro_total = total - capital

    st.write(f"Valor final: R$ {total:.2f}")

    if lucro_total >= 0:
        st.success(f"Lucro: R$ {lucro_total:.2f}")
    else:
        st.error(f"Prejuízo: R$ {lucro_total:.2f}")

    # ===== GRÁFICO PORTFÓLIO =====
    st.subheader("📈 Evolução do Portfólio")

    fig, ax = plt.subplots()
    ax.plot(historico_portfolio)
    ax.set_title("Crescimento do capital")
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Valor (R$)")

    st.pyplot(fig)

    # ===== GRÁFICO INTERATIVO =====
    st.subheader("📊 Analisar Ação")

    acao = st.selectbox("Escolha uma ação", list(dados_ativos.keys()))
    dados = dados_ativos[acao]

    fig2, ax2 = plt.subplots()
    ax2.plot(dados['Close'], label="Preço")
    ax2.plot(dados['media9'], label="Média 9")
    ax2.plot(dados['media21'], label="Média 21")
    ax2.legend()

    ax2.set_title(f"{acao} - Gráfico")

    st.pyplot(fig2)