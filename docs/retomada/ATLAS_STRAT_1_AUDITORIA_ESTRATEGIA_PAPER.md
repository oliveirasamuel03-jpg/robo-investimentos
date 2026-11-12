# ATLAS — STRAT-1: auditoria estática de estratégia PAPER e fidelidade das métricas

Data: 12/11/2026. Repositório: `oliveirasamuel03-jpg/robo-investimentos`.
Branch: `atlas/audit-2-consolidacao-seguranca-estrategia`.
Código inspecionado em `atlas/sec-1-higiene-artifacts` (sem alterações operacionais desde o produto INFRA-1.2.4).
**Natureza: ANALYSIS_ONLY / STUDY_ONLY / PAPER_ONLY / NO_TRADING_AUTHORITY.**
Nenhum score, min_signal_score, threshold, perfil, ticket, capital, max_open_positions, provider, broker, BOS, Fibonacci, H1/H4, sinal externo ou regra operacional foi alterado.

## 1. Veredito

**ESTUDO ESTÁTICO CONCLUÍDO COM RISCOS A INVESTIGAR; NÃO É AUDITORIA DE RENTABILIDADE.**
O projeto possui definição de perfis, cálculo de score, filtros de entrada, saídas, dimensionamento de posição, cooldown, guardas de feed e perda diária e integração transacional PAPER. Existem testes sintéticos correspondentes. **Não houve execução de backtest, replay de mercado, Docker/PostgreSQL, cobertura, medição de rentabilidade ou validação em dataset externo nesta etapa.**

A INFRA-1.2.7 de integridade é pré-requisito para qualquer avaliação que dependa da confiança no ledger. Mesmo com ledger íntegro, ganho PAPER não é sinal automático de vantagem estatística nem autorização para dinheiro real.

## 2. Caminhos de decisão identificados

- `engines/trader_engine.py:48-109`: converte estado de configuração e risco em `PaperTradingConfig`; seleciona perfil, multiplica ticket e define `min_signal_score` e filtros. `allow_new_entries` depende de `RUNNING` e da trava de perda diária, mas **não substitui** o controle financeiro transacional e o STOP.
- `paper_trading_engine.py:380-399`: enriquece candles com MA20/MA50, RSI, ATR%, volatilidade, momentum, breakout_20 e pullback.
- `paper_trading_engine.py:434-464`: score composto por tendência, RSI, ATR, inclinação, posição frente à máxima, pullback e momentum.
- `paper_trading_engine.py:476-540`: entrada exige checks de tendência, RSI, ATR, pullback, momentum, breakout e score mínimo, com motivos de rejeição.
- `paper_trading_engine.py:849-909`: saída por stop duro, trailing, RSI, perda de médias e tempo de holding.
- `paper_trading_engine.py:912-937,1672-1736`: tamanho da entrada, cash, quantidade, estado de posições e equity.
- `core/daily_risk.py:42-115`: perda diária **realizada** a partir de SELL datados no dia UTC e comparada a limite configurado.
- `engines/trader_engine.py:253-326`: aquisição, validação do fingerprint, execução PAPER e reconciliação; sem tentativa financeira automática ao encontrar incerteza.
- `tests/test_strategy_rules.py` e `tests/test_trader_smoke.py`: há cenários sintéticos de compra, rejeição, stop, timeout, cooldown, volume, fallback, trava de perda e integração. **Existência desses testes não prova backtest realista nem que foram executados nesta sessão.**

## 3. Riscos por prioridade

### STRAT1-01 — ALTA: moeda de cotação versus rótulos em R$

O aplicativo usa ativos cripto tipicamente cotados com sufixo `-USD` e rotula capital/ticket/PnL em R$. Em `paper_trading_engine.py:1676-1689`, `quantity = round(notional / price, 6)`, com `cash -= notional`, sem conversão monetária explícita no caminho examinado. Na venda, `paper_trading_engine.py:1177-1190` soma `quantity * price` ao mesmo `cash`, e reconhece `(price - avg_price) * quantity` como resultado; também não foi localizada conversão nesse trecho.

**Risco:** caso o feed realmente retorne preço em USD enquanto o ledger/UI tratam o saldo como BRL, valores absolutos, tamanho da posição e PnL em reais ficam economicamente mal rotulados ou incorretos. Isso **não demonstra** que o feed efetivo está em moeda diferente em todos os caminhos, mas obriga contrato explícito de `quote_currency`, `accounting_currency`, FX, timestamp e fonte antes de afirmar retorno em R$.

**Teste futuro:** mesmo OHLC em USD com taxas FX sintéticas distintas; provar tamanho econômico, fechamento, equity e relatório com e sem conversão autorizada. **Não adicionar FX nem alterar provider nesta fase.** Se a simulação adotar unidade sintética sem conversão, não rotular como reais reais.

### STRAT1-02 — ALTA: custos de negociação e fricção de mercado não modelados no caminho PAPER

`paper_trading_engine.py:1205` e `:1727` gravam `cost: 0.0` em SELL e BUY. Os trechos inspecionados liquidam entrada/saída pelos valores de fechamento apresentados, sem ajuste observável para spread, corretagem, taxas, slippage ou execução parcial.

**Risco:** PnL PAPER tende a apresentar uma simulação idealizada e pode superestimar a rentabilidade; o tamanho do viés depende da frequência de operações, liquidez, taxas e spread observados. Não inferir ganho, percentual ou custo real sem dados.

**Teste futuro:** camada de relatório independente mostrando cenário bruto versus custo hipotético de sensibilidade, sem modificar o ledger ou regras de trade. Qualquer modelo operacional de custo/execução exige fase separada.

### STRAT1-03 — MÉDIA/ALTA: indicador `breakout_20` não significa rompimento positivo da máxima anterior

`paper_trading_engine.py:397` calcula `close / close.rolling(20).max() - 1.0`. Como a janela inclui o fechamento corrente, para preços finitos positivos, `breakout_20 <= 0`. Os perfis inspecionados em `core/trader_profiles.py:31,59,87` usam limites **negativos**, de forma que o indicador pode estar sendo usado intencionalmente como **distância da máxima recente**, não como confirmação de rompimento acima da máxima anterior.

**Risco:** nomenclatura e interpretação diagnóstica (“breakout”) podem induzir a ler proximidade da máxima como breakout. Não declarar bug de entrada apenas por essa identidade; primeiro comparar contratos/estudos/tests. Não mudar a janela nem thresholds sem fase específica.

**Teste futuro:** sequência com nova máxima, empate e preços abaixo da máxima; exigir especificação formal do conceito pretendido, sem transformar shadow em autoridade.

### STRAT1-04 — ALTA: somente fechamento no gatilho de stop e no preço de saída simulado

`paper_trading_engine.py:879-909` verifica stop, trailing e outros critérios pelo `latest["close"]`. O fechamento da posição em `:1177-1182` usa o mesmo fechamento, embora o dataframe de dados tenha `high` e `low`.

**Risco:** um candle que toca o stop intrabar e fecha acima não gera a mesma saída que um modelo de execução intrabar. Em situações de gap/slippage, o preço executável também pode diferir do fechamento. Isso é **limitação de realismo da simulação**, não prova de defeito de segurança do STOP do operador.

**Teste futuro:** candles sintéticos com mínima abaixo do stop e fechamento acima, e gaps entre candles; comparar política documentada e erro de estimativa, sem alterar saídas operacionais.

### STRAT1-05 — MÉDIA/ALTA: candle mais recente e fechamento definitivo não comprovados

`paper_trading_engine.py:1159` e em outras decisões a leitura usa `data.iloc[-1]` após enriquecimento. No código amostrado não se estabeleceu, em todos os caminhos de provedor/intervalo, que a última barra está **efetivamente fechada** e cronologicamente estável. Um candle parcial pode alterar indicadores e decisão dentro do mesmo período.

**Teste futuro:** simular candle aberto/atualizado, carimbo UTC, consolidação de timezone, duplicatas de timestamp, stale provider e limites dos intervalos; garantir que o pipeline seja explicitamente auditável. Não concluir look-ahead demonstrado sem reprodução.

### STRAT1-06 — MÉDIA: perda diária realizada não é limite de drawdown total

`core/daily_risk.py:42-76` soma `realized_pnl` de SELL do mesmo dia UTC. Posições abertas e seu PnL não realizado **não entram diretamente nesse cálculo**. Lucros realizados podem compensar perdas realizadas dentro do mesmo dia, e a trava é baseada no resultado **líquido diário**, conforme cálculo. Isso pode ser política legítima, mas deve ser apresentado ao usuário como **perda diária realizada líquida**, não como limite absoluto de exposição, perda por trade ou drawdown total.

**Teste futuro:** perdas e ganhos alternados, posição aberta profundamente negativa, fechamento perto da virada UTC, registros duplicados e cash/equity divergentes, com avaliação somente de diagnóstico. Não alterar `DAILY_LOSS_LIMIT_BRL` nem a regra sem fase.

### STRAT1-07 — MÉDIA: cobertura estatística insuficiente para inferência de vantagem

Testes sintéticos usam cenários construídos para aceitar/rejeitar sinais ou verificar integridade, e não fornecem evidência de generalização estatística. Não foram verificados backtests walk-forward, vazamento temporal do dataset, custos realistas, sobrevivência a regimes, bootstrap de resultados, amostras independentes, exposição acumulada ou intervalos de confiança.

**Critério para futura prova PAPER de desempenho:** dados com origem/licença/tempo documentados; períodos não utilizados na escolha de thresholds; custos e moedas definidos; normalização de timezone e candles completos; comparativo entre regime de mercado e período de feed ruim; distribuição de PnL líquido, drawdown máximo, lucro por trade, exposição, taxa de rejeição, turnover e sensibilidade a custo/latência.

Não promover a estratégia por resultado isolado de 10 dias ou 72 horas.

## 4. Proteções que não devem ser debilitadas

- Aguardam validação real: `assert_execution_allowed`, revalidação transacional, STOP, idempotência e `UNKNOWN_COMMIT`.
- Não transformar estudos H1/H4, BOS, pivô, Fibonacci, shadow, webhook ou sinal externo em autorização de BUY/SELL.
- Não flexibilizar `min_signal_score`, outros thresholds, perfis ou ticket para “fazer operar mais”.
- Não desativar guardas de feed FALLBACK, risco ou posições apenas para gerar amostras.
- Não corrigir custos/moeda misturando alterações em estratégia e infraestrutura. Métrica de qualidade do PAPER pode ser estudada em relatório separado, com código operacional congelado.

## 5. Matriz de testes futuros (todos não executados)

| ID | Teste | Resultado esperado |
| --- | --- | --- |
| ST-01 | USD/BRL e unidade contábil | Exposição e PnL coerentes com moeda declarada ou rótulo explicitamente sintético |
| ST-02 | Custos hipotéticos | Relatório bruto e sensibilidade a spread/taxas sem alterar ledger |
| ST-03 | Breakout de nova máxima | Valor <=0 quando máximo inclui candle; sem falsa interpretação operacional |
| ST-04 | Stop intrabar e gap | Limitação quantificada e comunicação conservadora |
| ST-05 | Candle aberto, fechado e duplicate timestamp | Sem tratar barra incompleta como fechada sem prova |
| ST-06 | PnL realizado líquido versus drawdown | Definições e rótulos não confundidos |
| ST-07 | Feed FALLBACK/DELAYED | Entradas conforme guardas, nenhum aumento de autoridade |
| ST-08 | Score e thresholds invariantes | Bytecode/configurações funcionais sem mudanças |
| ST-09 | Trade sem liquidez e posição fracionada | Quantidade, arredondamento e notional sob limites |
| ST-10 | Out-of-sample, walk-forward, custos e drawdown | Pesquisa independente em PAPER, sem otimização na amostra de teste |
| ST-11 | Idempotência do ledger | Eventos, recibos e resultados com commit incerto sem reenvio |
| ST-12 | Real broker e shadow | Ambos continuam sem permissão de operar dinheiro real |

## 6. Ordem de trabalho e decisão

1. Corrigir P0/P1 de retenção/quarentena na **INFRA-1.2.7** com PostgreSQL sintético; resolver desempenho STOP antes de 72 horas.
2. Em fase distinta, criar **estudo de fidelidade financeira PAPER** para moeda/custos/stop intrabar e origem de candles, com métricas de diagnóstico e sem aplicar ajustes operacionais.
3. Somente depois elaborar backtest reproduzível e monitoramento estatístico fora da amostra. Não usar performance ainda não medida para comprar equipamento ou habilitar dinheiro real.

**Status STRAT-1:** auditoria estática documentada; execução e validação das hipóteses pendentes. Não há autorização para alterar estratégia ou broker.
