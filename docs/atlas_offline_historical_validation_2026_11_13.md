# ATLAS — Avaliacao historica offline autorizada | 2026-11-13

**Natureza:** STUDY_ONLY / DIAGNOSTIC_ONLY / SHADOW_ONLY / PAPER_ONLY.  
**Fonte da autorizacao:** conversa do projeto, posterior ao estudo PR #78.  
**Base GitHub verificada:** `main@f0c932c21212064dcef37e1c928b0b188adf440f`.  
**Capital de hipotese:** R$ 10.000,00 (nao corresponde a capital alterado no app).  
**Status final nesta rodada:** **VALIDACAO DO AVALIADOR APROVADA COM LIMITACAO; BACKTEST HISTORICO REAL NAO EXECUTADO**.

## Escopo e resultados concretos

1. Inspecionadas no repositorio as funcoes de calculo de RSI, ATR, indicadores, score e avaliacao de compra de `paper_trading_engine.py`, assim como os parametros-base de `PaperTradingConfig` e os perfis.
2. Elaborado e executado em runtime isolado um **prototipo externo ao repositorio** chamado `atlas_historical_replay.py`: valida arquivos CSV 1h e manifesto de cinco criptoativos; extrai, por AST controlada, as funcoes existentes de indicadores e filtros de sinal a partir de um arquivo `paper_trading_engine.py` local e contabiliza avaliacao por regime ALTA / BAIXA / LATERAL.
3. Executados **14 testes Python/Pytest aprovados (14 passed in 0.38s)** com candles artificiais declarados `TEST_FIXTURE` e uma **replica local minima do engine para ensaiar o prototipo**, nunca o motor completo de producao. A testagem do prototipo com as funcoes da main verdadeira ainda depende da revisao e inclusao deste codigo em branch, sem modificar o motor.
4. Testes contemplaram verificacao de SHA-256, origem de dados, data/hora UTC, duplicatas, gaps, OHLC impossivel, volume negativo, ausencias, desalinhamento da janela entre ativos e ausencia de mutacao de entradas. Dados `VALIDATION_ARTIFACT` e fixtures no modo normal sao rejeitados.
5. O manifesto de exemplo deliberadamente possui `UNVERIFIED` e foi rejeitado com `DATA_QUALITY_BLOCKED` e `action_code=NO_CHANGE`, evidenciando fail-closed.

## Bloqueios e honestidade quantitativa

- **Nao ha dataset OHLCV 1h, integro e independentemente autenticado, disponivel nesta sessao para os cinco ativos.**
- Foram pesquisadas fontes publicas. A Binance publica arquivos spot 1h por par e CHECKSUM, por exemplo `BTCUSDT-1h-2026-09.zip`, mas a recuperacao/verificacao destes arquivos falhou neste ambiente. Nenhum numero de performance foi inferido.
- **BTCUSDT nao e BTC-USD:** fontes USDT seriam apenas proxies para a watchlist USD e nao comprovam resultado no provider Twelve Data.
- Retornos, drawdown, win rate, PnL, vantagem da IA, comparacao de performance entre regimes e sinais de entrada em tempo real **NAO FORAM CALCULADOS**.
- Avaliacoes descritivas dos filtros sem feed/context/guards/posicoes nao sao equivalentes a um backtest completo do app.
- Os 14 testes cobrem seguranca de entrada e o prototipo, **nao** aprovam o comportamento do motor real.

## Materiais produzidos fora do repositorio

- `research/atlas_historical_replay.py`: programa de pesquisa, somente leitura, sem chamadas de corretora, banco, Railway ou provider.
- `tests/test_atlas_historical_replay.py`: suite de 14 testes.
- `README.md`: instrucoes, escopo e limitacoes.
- `manifest_template.json`: exemplo intencionalmente invalido, para impedir uso sem verificacao.
- `validation_run.json` e `test_run.txt`: evidencia de ensaio offline.

Esses arquivos foram empacotados em arquivo ZIP na conversa para avaliacao humana; **este PR nao importa o prototipo nem altera codigo operacional**.

## Proximo passo de risco controlado

1. Obter cinco CSVs historicos de mesma janela, 1h, com proveniencia, checksum do fornecedor e conversao documentada para UTC e OHLCV.
2. Verificar diferencas de base USD versus USDT; nao equiparar providers.
3. Em **missao separada** com branch, revisar o codigo experimental, executar os testes contra o `paper_trading_engine.py` real, adicionar testes adversariais e usar apenas fontes verificadas.
4. Para realizar backtest real completo, reproduzir tambem contexto macro, feed guards, controle de risco, saldo, ordens, saidas, custos, slippage, limites e tempo; estimar desempenho fora da amostra com walk-forward.
5. Somente apos revisao de seguranca, decidir proximos estudos. **Sem merge automatico.**

## Salvaguardas

- Nenhuma ordem real ou PAPER oficial enviada.
- Nenhuma alteracao de parametros, capital real, ticket, score, min_signal_score, thresholds, provider, broker, estrategia, banco, volume, env, Railway ou runtime.
- Nenhuma mudanca de `main`.
- FASE 2.6C permanece BLOQUEADA.
