import yfinance as yf
import pandas as pd

# ===== FUNÇÃO RSI =====
def calcular_rsi(dados, periodo=14):
    delta = dados.diff()
    ganho = (delta.where(delta > 0, 0)).rolling(periodo).mean()
    perda = (-delta.where(delta < 0, 0)).rolling(periodo).mean()
    rs = ganho / perda
    return 100 - (100 / (1 + rs))

# ===== LISTA DE AÇÕES =====
ativos = [
    "VALE3.SA","PETR4.SA","PETR3.SA","ITUB4.SA","BBDC4.SA","BBDC3.SA",
    "BBAS3.SA","ABEV3.SA","WEGE3.SA","MGLU3.SA","LREN3.SA","RENT3.SA",
    "SUZB3.SA","JBSS3.SA","RADL3.SA","PRIO3.SA","RAIL3.SA","EQTL3.SA",
    "GGBR4.SA","CSNA3.SA","HAPV3.SA","BRFS3.SA","ELET3.SA","ELET6.SA"
]

dados_ativos = {}
resultados = []

# ===== TESTAR TODOS OS ATIVOS =====
for ativo in ativos:
    print(f"\n🔄 Testando {ativo}...")

    dados = yf.download(ativo, start="2018-01-01")

    if dados.empty:
        print("❌ Sem dados")
        continue

    if isinstance(dados.columns, pd.MultiIndex):
        dados.columns = dados.columns.droplevel(1)

    close = dados['Close']

    # INDICADORES
    dados['media9'] = close.rolling(9).mean()
    dados['media21'] = close.rolling(21).mean()
    dados['media50'] = close.rolling(50).mean()
    dados['RSI'] = calcular_rsi(close)

    dados['sinal'] = 0
    dados.loc[dados['media9'] > dados['media21'], 'sinal'] = 1
    dados.loc[dados['media9'] < dados['media21'], 'sinal'] = -1

    dinheiro = 1000
    acoes = 0
    taxa = 0.001
    stop_loss = 0.97
    preco_compra = 0

    historico = []

    for i in range(len(dados)):
        preco = close.iloc[i]
        sinal = dados['sinal'].iloc[i]
        rsi = dados['RSI'].iloc[i]
        media50 = dados['media50'].iloc[i]

        # COMPRA
        if sinal == 1 and dinheiro > 0 and rsi < 70 and preco > media50:
            acoes = (dinheiro * (1 - taxa)) / preco
            dinheiro = 0
            preco_compra = preco

        # VENDA
        elif acoes > 0:
            if preco < preco_compra * stop_loss:
                dinheiro = acoes * preco * (1 - taxa)
                acoes = 0

            elif sinal == -1 or rsi > 70:
                dinheiro = acoes * preco * (1 - taxa)
                acoes = 0

        valor_total = dinheiro + (acoes * preco)
        historico.append(valor_total)

    valor_final = historico[-1]
    lucro = valor_final - 1000

    # DRAWDOWN
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
print("\n📊 RANKING DOS ATIVOS:\n")
resultados = sorted(resultados, key=lambda x: x[1], reverse=True)

for ativo, lucro, dd in resultados[:10]:
    print(f"{ativo} → Lucro: R$ {lucro:.2f} | Drawdown: {dd*100:.2f}%")

# ===== FILTRO DE RISCO =====
selecionados = []

for ativo, lucro, dd in resultados:
    if lucro > 0 and dd < 0.30:
        selecionados.append((ativo, lucro, dd))

# ===== EVITAR EMPRESAS REPETIDAS =====
carteira_final = []
empresas_usadas = set()

for ativo, lucro, dd in selecionados:
    empresa = ativo[:4]

    if empresa not in empresas_usadas:
        carteira_final.append((ativo, lucro, dd))
        empresas_usadas.add(empresa)

    if len(carteira_final) == 3:
        break

print("\n🤖 PORTFÓLIO DIVERSIFICADO:\n")

for ativo, lucro, dd in carteira_final:
    print(f"{ativo} → Lucro: R$ {lucro:.2f} | Drawdown: {dd*100:.2f}%")

# ===== SIMULAR PORTFÓLIO =====
capital_total = 1000
capital_por_ativo = capital_total / len(carteira_final)

valor_total_final = 0

print("\n💰 SIMULAÇÃO DO PORTFÓLIO:\n")

for ativo, _, _ in carteira_final:
    dados = dados_ativos[ativo]
    close = dados['Close']

    dinheiro = capital_por_ativo
    acoes = 0
    taxa = 0.001
    stop_loss = 0.97
    preco_compra = 0

    for i in range(len(dados)):
        preco = close.iloc[i]
        sinal = dados['sinal'].iloc[i]
        rsi = dados['RSI'].iloc[i]
        media50 = dados['media50'].iloc[i]

        if sinal == 1 and dinheiro > 0 and rsi < 70 and preco > media50:
            acoes = (dinheiro * (1 - taxa)) / preco
            dinheiro = 0
            preco_compra = preco

        elif acoes > 0:
            if preco < preco_compra * stop_loss:
                dinheiro = acoes * preco * (1 - taxa)
                acoes = 0

            elif sinal == -1 or rsi > 70:
                dinheiro = acoes * preco * (1 - taxa)
                acoes = 0

    valor_final = dinheiro + (acoes * close.iloc[-1])
    valor_total_final += valor_final

    print(f"{ativo} → Final: R$ {valor_final:.2f}")

print("\n🏁 RESULTADO FINAL DO PORTFÓLIO:")
print(f"Valor total: R$ {valor_total_final:.2f}")
print(f"Lucro total: R$ {valor_total_final - capital_total:.2f}")