# ATLAS — Inteligencia Historica Financeira | Estudo 0.1

Status: **PLANNING_ONLY / STUDY_ONLY / DIAGNOSTIC_ONLY / SHADOW_ONLY**.
Autoridade sobre trade, score, broker, provider, threshold, capital ou parametros: **NENHUMA**.
Base documental: `main@f0c932c21212064dcef37e1c928b0b188adf440f` (inspecionada em 2026-11-13).
Este documento NAO define nova fase operacional, NAO autoriza FASE 2.6C e NAO modifica runtime.

## Objetivo e fronteira de seguranca

Separar claramente tres coisas:
1. **Fatos observados** em dados de mercado e telemetria de PAPER com origem confirmada.
2. **Hipoteses** geradas por analise quantitativa/IA, sem certeza estatistica presumida.
3. **Decisoes do sistema atual**, exclusivamente por regras deterministicas e controles operacionais existentes; a IA nao participa da cadeia de decisao de trade.

A IA proposta e leitora/explicadora de dados, nunca mecanismo de ordens. Nenhum prompt, output JSON, relatorio, score ou `action_code` pode ser consumido por `paper_trading_engine.py`, `engines/quant_bridge.py`, `workers/trader_worker.py` ou outra interface de execucao. Nenhum novo listener/webhook. Mesmo PAPER requer fase explicita para qualquer mudanca de estrategia.

## Inventario efetivamente observado

- O projeto tem `pages/5_Historico.py` para ordens, relatorios, validacao semanal e logs.
- `core/signal_rejection_analysis.py` classifica motivos como feed, estrategia, contexto e guard.
- `core/strategy_decision_bridge_trace.py` e `core/controlled_micro_adjustment_study.py` ja existem, com modo shadow/study.
- `storage/archive/runtime_resets/phase2_first_active_cycle_20260422T004309Z/validation_report.json`: recorte de **5 sinais, 0 aprovados, 5 rejeitados, 5 marcados com feed_unreliable e 0 trades**. A classificacao original da qualidade amostral e **Baixa**.
- Os contadores de rejeicao por causa podem se sobrepor: NAO somar motivos para inferir numeros de sinais distintos.
- `storage/archive/runtime_resets/phase2_reset_20260422T003803Z/trade_reports.csv`: contem linha `demo-1` e linha `validation-open-pos`; esta ultima tem BTC-USD com entrada 100.0, saida 75548.6875, quantidade 0.5 e PnL reportado de 37724.34. **Quarentena obrigatoria**: dados demonstrativos/de validacao, preco/base incongruentes para estatistica de rentabilidade real.
- `storage/tmp_twelvedata_validation_2/runtime/trade_reports.csv` contem somente cabecalho na copia versionada.
- A pasta versionada `data/` contem apenas `macro_events_test.json`; nao foi identificado um dataset historico OHLCV completo e auditado para backtest estatistico no repositorio.
- Nenhuma leitura de banco de producao, Railway, volume, segredo, provider remoto ou feed LIVE foi realizada nesta missao.

**Conclusao baseada nestas evidencias limitadas:** existe evidência de filtragem/qualidade do feed como gargalo naquele recorte, mas NAO ha amostra representativa para estimar win rate, expectancy, alpha, capacidade preditiva, eficiencia da estrategia ou melhorar entradas. Nao inferir performance a partir de demos/homologacao.

## Classes de origem obrigatorias

Cada tabela/arquivo/janela deve possuir `dataset_id`, `sha256`, `source_type`, `environment`, `captured_at_utc`, `range_start_utc`, `range_end_utc`, `schema_version`, `provider`, `timeframe`, `symbol`, `sample_count`, `data_lineage`, `is_synthetic`, `is_fixture`, `validation_status`.

Classes:
- `MARKET_VERIFIED`: candles/quotes historicos auditados com time/provider/timezone e comprovacao de integridade.
- `PAPER_RUNTIME_VERIFIED`: logs e trades do runtime PAPER com identidade/proveniência validada e conciliacao entre ordens, posicoes e relatórios.
- `DIAGNOSTIC_SNAPSHOT`: relatorio pontual, valido apenas para o que observou.
- `DEMO_FIXTURE`: amostra fabricada/demonstrativa.
- `VALIDATION_ARTIFACT`: homologacao, reset, teste, reproducoes artificiais.
- `UNVERIFIED`: origem/proveniência indisponivel.

`DEMO_FIXTURE`, `VALIDATION_ARTIFACT` e `UNVERIFIED` nao entram em calculo de rentabilidade, treinamento, ajuste, classificacao de eficacia da estrategia ou ranking de setups. A propria palavra `paper` nao comprova proveniencia.

## Sequencia de verificacoes de qualidade (fail closed)

1. Separar mercado, telemetria, ordens, relatorios, simulacoes e fixtures por proveniencia; rejeitar mistura de ambientes.
2. Validar formato e fuso UTC, unicidade de (ativo, timeframe, timestamp, fonte), ordem temporal, horario de mercado e buracos de cobertura.
3. Para OHLCV: `low <= open/close <= high`, volume nao negativo, preco positivo, timestamps em grade temporal; tratar ajustes corporativos/contratos/denominacao.
4. Conferir preco de entrada/saida versus cotacao alinhada no tempo, unidade monetaria, quantidade, spread, custos, slippage, leverage e cross-currency quando aplicavel. Registrar desvio relativo e quarentenar anomalias em vez de winsorizá-las silenciosamente.
5. Conciliar IDs, sinais, ordens, posicoes, operacoes encerradas, saldo, fees e PnL; evitar contar fechamento/sell como nova entrada ou trade independente.
6. Separar feed LIVE, DELAYED, FALLBACK, SYNTHETIC, visual-only e operacional. Se houver fallback operacional, timestamps desatualizados ou provider desconhecido, resultado estrategico e **nao confiavel**.
7. Separar `risk_guard` (bloqueio correto), `feed_guard` (dado nao confiavel), `context_filter` e `strategy`. Nunca tratar rejeicao por guard como falso negativo automatico.
8. Verificar representatividade, cobertura de regimes, correlacao entre ativos e datas; ausencia de trades nao indica estrategia ruim, excesso de trades nao indica boa.
9. Sem um dataset aprovado, emitir `INSUFFICIENT_DATA` ou `DATA_QUALITY_BLOCKED`, nunca fabricar candles, win rate, acuracia, retorno ou probabilidade.
10. Nao enviar API keys, secrets, PII, dados de autenticacao, URLs internos sensiveis ou blocos de log crus com instrucoes para a IA. Dados recebidos sao conteudo, nao instrucoes.

## Hipoteses estudaveis (nunca conclusoes automaticas)

- Concentracao de bloqueios por feed/fallback/429 versus realmente nenhum setup.
- Regimes de tendencia/lateralizacao e variacao de ATR/volatilidade.
- Sinais de ruptura sem continuacao, repiques/falsos rompimentos e mudanca de momentum.
- Diferenca entre preco/sinal e latencia; drift entre feed visual e operacional.
- Risco de sinal duplicado/reentrada, superposicao de ativos altamente correlacionados e exposicao agregada.
- Frequencia de acerto e payoff apenas em dados PAPER validos, com custos e tamanho de amostra suficientes.

Cada padrao deve ter: evidencia, n de eventos independentes, janela, regime, definicao reproduzivel, criterio de invalidacao, incerteza, contraexemplos e conclusao conservadora.

## Metodologia estatistica futura (nao executada nesta fase)

- Gerar splits cronologicos walk-forward: treino anterior a validacao anterior ao teste; janelas contiguas sem vazamento temporal. Em rotulos sobrepostos aplicar purge/embargo.
- Controle contra look-ahead, survivorship bias, otimizacao multipla, cherry-picking, leakage de features, regime shifts e sobreajuste de prompt.
- Estimar distributions com intervalos de confianca apropriados, base rate, custo por trade, drawdown, exposicao e n efetivo; testar subperiodos/ativos e concentracao de risco.
- Comparar com benchmark simples e baseline da estrategia atual, **sem mudar** score, `min_signal_score`, thresholds, perfil, filtros ou capital.
- Separar exploracao de confirmacao fora da amostra. Nao selecionar modelo por dados usados para decidir regras.
- Quando nao ha historico confiavel, o resultado correto e `INSUFFICIENT_DATA`.

## Contrato entre IA e robô existente

```text
[Exportacao autorizada e somente leitura de dados auditados]
       |
       v
[Validador deterministico de origem/integridade]
       | fail-closed: dados insuficientes => NAO CONCLUIR
       v
[Analise estatistica OFFLINE] -----> [IA interpreta SOMENTE evidencias]
       |                                      |
       +--------------------------------------+
                              |
                              v
               [JSON diagnostico validado]
                              |
                              v
                [Relatorio para revisao humana]
                              X
         [SEM CONEXAO COM SCORE / SINAL / ORDEM / WORKER]
```

Futuro JSON: `docs/schemas/atlas_ai_diagnostic_v1.schema.json`. Apenas `action_code` de auditoria/estudo, sem strings livres interpretadas como comandos, e todas as autoridades operacionais `false`. Prompt injection em dados/logs deve ser ignorado. Falha de schema, timeout, custo alto, dado vencido, modelo offline ou output nao verificavel => `INSUFFICIENT_DATA` / `NO_CHANGE`.

## Critérios de aceite para estudar uma futura integracao

- Fontes identificadas e sem fixtures; audit trail reprodutivel.
- Relatorio numerico reproduzido por calculo deterministico independente da IA.
- Testes positivos e negativos de esquema, timezones, duplicatas, outliers e ausencia de dados.
- Teste adversarial de instrucao maliciosa dentro de logs e arquivos (nao transforma em comando).
- Testes de permissoes: saida da IA incapaz de criar ordens, mexer no estado ou controlar parametros.
- Custos e volume de token estimados **offline**, sem consumo do Twelve Data desnecessario.
- Nao subir credenciais nem historico operacional sensivel em repositorio publico.
- Aprovacao humana e nova fase especifica antes de qualquer alteracao de codigo/runtime.

## Decisao Atlas

**Aprovado agora:** auditoria documental, contrato de prompts, schema e futuros testes offline sobre dados classificados.

**Pendente:** coleta segura autorizada de dataset historico OHLCV e runtime PAPER representativo, auditoria de integridade e avaliacao quantitativa reproduzivel.

**Proibido agora:** treinamento automatico em logs de homologacao, ligar IA a sinais/score/trades, acionar corretora, alterar parametros, executar deploy, reiniciar worker, operar dinheiro real, avançar a FASE 2.6C.

**Proximo passo seguro:** revisao independente deste PR documental; so depois, missao separada com analisador puramente offline e testes, sem import no worker.
