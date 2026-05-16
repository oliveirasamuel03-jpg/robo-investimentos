# Trade Ops Desk

Plataforma Streamlit focada exclusivamente em trade operacional, com worker continuo, paper trading, controle de risco, autenticacao e persistencia local ou em PostgreSQL.

## Visao geral

O produto agora e orientado apenas ao fluxo de trading:

- painel do trader com sinais, posicoes, PnL e monitoramento do worker
- worker continuo para execucao automatizada
- historico operacional com ordens, relatorios fechados e logs
- controle administrativo do bot
- autenticacao com perfis de usuario
- persistencia local em `storage/runtime/`
- persistencia compartilhada em PostgreSQL quando `DATABASE_URL` existir

## Arquitetura

### Interface

- `app.py`: entrada principal da plataforma
- `pages/0_Login.py`: login e bootstrap do admin inicial
- `pages/1_Trader.py`: painel do trader
- `pages/4_Controle_do_Bot.py`: controle administrativo
- `pages/5_Historico.py`: ordens, relatorios operacionais e logs

### Core

- `core/config.py`: configuracao central e caminhos de runtime
- `core/persistence.py`: leitura e escrita local ou em PostgreSQL
- `core/state_store.py`: estado operacional do bot e do trader
- `core/production_monitor.py`: saude operacional, heartbeat, feed e broker
- `core/alerts.py`: envio de alertas por email com cooldown
- `core/retention.py`: retencao automatica, arquivamento e resumos semanais
- `core/trader_profiles.py`: perfis Reservado, Equilibrado e Agressivo
- `core/trader_reports.py`: relatorios fechados e analytics
- `core/auth/`: autenticacao, sessao e autorizacao

### Engines e workers

- `engines/trader_engine.py`: orquestracao do ciclo do trader
- `engines/quant_bridge.py`: ponte oficial para o fluxo de paper trading
- `paper_trading_engine.py`: execucao, sinais, relatorios e paper state
- `workers/trader_worker.py`: loop continuo do worker

## Estrutura de pastas

```text
.
|-- app.py
|-- core/
|   |-- auth/
|   |-- config.py
|   |-- persistence.py
|   |-- state_store.py
|   |-- trader_profiles.py
|   `-- trader_reports.py
|-- engines/
|-- pages/
|-- storage/
|   |-- archive/
|   |   |-- logs/
|   |   |-- reports/
|   |   `-- weekly_reports/
|   |-- cache/
|   |-- reports/
|   `-- runtime/
|       `-- weekly_reports/
|-- tests/
|-- workers/
|-- requirements.txt
|-- requirements-dev.txt
`-- DEPLOY_RAILWAY.md
```

## Runtime e persistencia

Sem `DATABASE_URL`, o app usa arquivos locais em `storage/runtime/`.
Com `DATABASE_URL`, o estado principal vai para PostgreSQL e o filesystem segue como fallback seguro.

Arquivos de runtime principais:

- `storage/runtime/bot_state.json`
- `storage/runtime/trader_orders.csv`
- `storage/runtime/trade_reports.csv`
- `storage/runtime/bot_log.csv`
- `storage/runtime/auth_users.json`

Arquivamento automatico:

- `storage/archive/reports/trade_reports_YYYY_MM.csv`
- `storage/archive/logs/bot_log_YYYY_MM.csv`
- `storage/runtime/weekly_reports/`
- `storage/archive/weekly_reports/`

## Variaveis de ambiente

Copie o arquivo de exemplo:

```bash
cp .env.example .env
```

Variaveis suportadas:

- `APP_TITLE`
- `APP_ENV`
- `ROBO_STORAGE_DIR`
- `DATABASE_URL`
- `AUTH_REQUIRED`
- `AUTH_SESSION_TIMEOUT_MINUTES`
- `ADMIN_USERNAME`
- `ADMIN_PASSWORD`
- `PRODUCTION_MODE`
- `MARKET_DATA_PROVIDER`
- `MARKET_DATA_FALLBACK_PROVIDER`
- `TWELVEDATA_API_KEY`
- `TWELVEDATA_API_BASE`
- `MACRO_ALERTS_ENABLED`
- `MACRO_ALERT_PRE_EVENT_MINUTES`
- `MACRO_ALERT_EVENT_WINDOW_MINUTES`
- `MACRO_ALERT_POST_EVENT_MINUTES`
- `MACRO_ALERTS_FILE`
- `EXTERNAL_SIGNAL_WEBHOOK_ENABLED`
- `EXTERNAL_SIGNAL_SECRET`
- `EXTERNAL_SIGNAL_MAX_AGE_SECONDS`
- `EXTERNAL_SIGNAL_ALLOWED_SOURCES`
- `EXTERNAL_SIGNAL_ALLOWED_TIMEFRAMES`
- `EXTERNAL_SIGNAL_DEDUPE_SECONDS`
- `EXTERNAL_SIGNAL_TEST_PANEL_ENABLED`
- `CALIBRATION_PREVIEW_ENABLED`
- `CALIBRATION_PREVIEW_MARGIN`
- `CALIBRATION_PREVIEW_MAX_EXAMPLES`
- `MULTITF_SWING_AUDIT_ENABLED`
- `MULTITF_SWING_TIMEFRAMES`
- `MULTITF_SWING_CACHE_TTL_SECONDS`
- `MULTITF_SWING_MAX_SYMBOLS`
- `MULTITF_SWING_REQUIRE_LIVE_FEED`
- `MULTITF_SWING_SHADOW_ONLY`
- `MULTITF_INTRADAY_FETCH_ENABLED`
- `MULTITF_INTRADAY_TIMEFRAMES`
- `MULTITF_INTRADAY_CACHE_TTL_SECONDS`
- `MULTITF_INTRADAY_MAX_SYMBOLS`
- `MULTITF_INTRADAY_MAX_CALLS_PER_CYCLE`
- `MULTITF_INTRADAY_REQUIRE_LIVE_FEED`
- `MULTITF_INTRADAY_SHADOW_ONLY`
- `MULTITF_INTRADAY_PROVIDER_BUDGET_MODE`
- `BOS_PIVOT_TRACE_AUDIT_ENABLED`
- `BOS_PIVOT_TRACE_SHADOW_ONLY`
- `STRATEGY_DECISION_BRIDGE_TRACE_ENABLED`
- `STRATEGY_DECISION_BRIDGE_TRACE_SHADOW_ONLY`
- `DAILY_LOSS_LIMIT_BRL`
- `ALERT_EMAIL_ENABLED`
- `ALERT_EMAIL_PROVIDER`
- `ALERT_EMAIL_TO`
- `ALERT_EMAIL_FROM`
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_USE_TLS`
- `RESEND_API_KEY`
- `RESEND_API_BASE`
- `ALERT_HEARTBEAT_MAX_DELAY_SECONDS`
- `ALERT_MAX_CONSECUTIVE_ERRORS`
- `ALERT_FEED_FALLBACK_MAX_MINUTES`
- `ALERT_COOLDOWN_MINUTES`
- `ALERT_SEND_RECOVERY_EMAIL`
- `REPORT_EMAIL_ENABLED`
- `REPORT_EMAIL_PROVIDER`
- `REPORT_EMAIL_TO`
- `REPORT_EMAIL_FROM`
- `REPORT_EMAIL_SMTP_HOST`
- `REPORT_EMAIL_SMTP_PORT`
- `REPORT_EMAIL_SMTP_USERNAME`
- `REPORT_EMAIL_SMTP_PASSWORD`
- `REPORT_EMAIL_USE_TLS`
- `REPORT_EMAIL_TIMEOUT_SECONDS`
- `REPORT_EMAIL_DAILY_ENABLED`
- `REPORT_EMAIL_WEEKLY_ENABLED`
- `REPORT_EMAIL_10DAY_ENABLED`
- `REPORT_EMAIL_FINAL_ENABLED`
- `RETENTION_ENABLED`
- `RETENTION_DAYS`
- `RETENTION_RUN_INTERVAL_HOURS`
- `RETENTION_ARCHIVE_TRADER_ORDERS`
- `WEEKLY_REPORT_RUNTIME_WEEKS`

## Dados de mercado

Na FASE 2, a camada de dados pode operar com cadeia de providers para manter o paper trading resiliente sem mascarar degradacao:

- provider principal: `Twelve Data`
- fallback secundario: `Yahoo`
- ultimo recurso: fallback sintetico ja existente

Configuracao recomendada para o modo gratis do Twelve Data:

```env
MARKET_DATA_PROVIDER=twelvedata
MARKET_DATA_FALLBACK_PROVIDER=yahoo
TWELVEDATA_API_KEY=sua_chave_do_twelve_data
TWELVEDATA_API_BASE=https://api.twelvedata.com
TWELVEDATA_MIN_CACHE_TTL_SECONDS=900
```

Observacoes importantes:

- a watchlist cripto da fase continua `BTC-USD`, `ETH-USD`, `BNB-USD`, `SOL-USD`, `LINK-USD`
- o runtime limpo da FASE 2 continua com capital-base de `R$ 1.000,00`
- o app mostra separadamente o provider efetivo e a natureza da fonte usada no ciclo
- `LIVE` indica dado fresco de provider real
- `DELAYED` indica reaproveitamento de cache ou mistura de fontes nao totalmente frescas
- `FALLBACK` indica uso do fallback sintetico e nao deve ser tratado como dado ao vivo
- `TWELVEDATA_MIN_CACHE_TTL_SECONDS=900` reduz a cadencia do provider principal para caber melhor no plano gratis com a watchlist atual de 5 ativos

Pelos docs oficiais do Twelve Data, o plano gratis inclui cripto em tempo real e o endpoint `time_series` consome `1` credito por simbolo. Isso torna a camada mais resiliente, mas ainda exige monitorar a franquia diaria e o comportamento do worker.

### FASE 2.5B - BOS/Pivot Trace Audit

A FASE 2.5B adiciona uma auditoria SHADOW_ONLY para explicar por que BOS e pivos 4H/1H ainda nao confirmam. Ela reutiliza os candles reais ja obtidos pelo fetcher intraday da FASE 2.5A, portanto nao cria novas chamadas ao provider nem reduz TTL/cache.

- BOS confirmado exige fechamento alem do topo/fundo estrutural com buffer.
- Pavio isolado vira `BOS_BY_WICK_ONLY`, nunca confirmacao.
- Fechamento pequeno vira `BOS_BY_CLOSE_WEAK`.
- Pivo exige estrutura minima de swings e fechamento/ativacao objetiva.
- A leitura 1H/4H distingue `H1_LEADS_H4`, `H4_CONFIRMS_H1`, conflito, ruido ou dados insuficientes.
- Tudo permanece `SHADOW_ONLY`: nao aprova trade, nao altera score real, nao muda broker, nao altera PnL e preserva PAPER TRADING.

### FASE 2.5B.1 - Strategy Decision Bridge Trace

A FASE 2.5B.1 adiciona uma ponte SHADOW_ONLY entre a estrutura confirmada em diagnostico e os bloqueios reais da estrategia. Ela responde por simbolo se o bloqueio veio de score, RSI, setup inelegivel, fallback atual/historico, guards ou divergencia entre Multi-TF e BOS/Pivo.

- Se BOS/Pivo 4H/1H confirmar estrutura, isso continua sem autoridade de entrada.
- A ponte separa fallback do ciclo atual de fallback acumulado/historico.
- A ponte mostra quando `h4_bos_missing` vem do top ativo ou de leitura agregada da watchlist.
- Recomendacoes sao apenas diagnosticas, como `study_real_strategy_blocker` ou `accumulated_fallback_only`.
- Nenhum score, threshold, broker, posicao, PnL ou ordem paper/real e alterado.

### FASE 2.5B.1A - Feed/Fallback Scope Reconciliation

A FASE 2.5B.1A adiciona uma reconciliacao DIAGNOSTIC_ONLY para separar fallback do ciclo atual, fallback acumulado/historico, fallback de candidato antigo e fallback apenas visual do grafico.

- Se o worker esta `LIVE`, com fallback atual `0` e ativos live no ciclo, o escopo nao pode ser `CURRENT_CYCLE`.
- Fallback antigo continua visivel como acumulado/historico, mas deixa de parecer gargalo atual.
- O Shadow Decision Simulator passa a mostrar dominante atual e dominante acumulado separadamente.
- A Strategy Decision Bridge usa essa leitura para explicar que score/setup/RSI seguem como bloqueios reais quando o feed atual esta limpo.
- Nada muda em score real, thresholds, provider, cache/TTL, budget da Twelve Data, broker, posicoes, PnL, historico ou ordens.

### FASE 2.5B.2 - NO_SETUP_ELIGIBLE Decomposition

A FASE 2.5B.2 adiciona uma camada DIAGNOSTIC_ONLY / SHADOW_ONLY para explicar por que o setup real ainda retorna `NO_SETUP_ELIGIBLE` mesmo quando estrutura, BOS/Pivo, Multi-TF ou Fibonacci aparecem favoraveis.

- A decomposicao cruza candidatos reais rejeitados, Strategy Decision Bridge, BOS/Pivo, Multi-TF, Fibonacci/estrutura e Feed/Fallback Scope.
- O bloco separa casos como estrutura confirmada mas setup real inelegivel, score quase no minimo com breakout ausente e pullback/reacao ainda nao mapeados pela regra real.
- O feed atual limpo continua explicitamente marcado como nao bloqueador quando `fallback_current=0`.
- Recomendacoes sao apenas diagnosticas, como `study_real_rule_mapping`, `study_breakout_confirmation` ou `keep_blocked_until_real_setup_maps_structure`.
- Nada muda em score real, thresholds, provider, cache/TTL, budget da Twelve Data, broker, posicoes, PnL, historico ou ordens.

### FASE 2.5B.2A - Reversal Blocker Routing Audit

A FASE 2.5B.2A adiciona uma camada DIAGNOSTIC_ONLY / SHADOW_ONLY para explicar quando `REVERSAL_NOT_ELIGIBLE` aparece no roteamento do `trend_pullback_breakout`.

- A auditoria cruza sinais rejeitados, Strategy Decision Bridge, NO_SETUP_ELIGIBLE Decomposition, Multi-TF, BOS/Pivo, Fibonacci/estrutura e Feed/Fallback Scope.
- O bloco separa risco legitimo de reversao, conflito Multi-TF, blocker de reversao em setup de tendencia e possivel roteamento misto com `reversal_v_pattern`.
- Recomendacoes permanecem em whitelist diagnostica, como `study_routing_map`, `study_multitf_conflict` e `keep_blocked_until_structure_confirms`.
- `should_keep_blocked` permanece verdadeiro e `safe_to_change_strategy_now` permanece falso.
- Nada muda em score real, thresholds, provider, cache/TTL, budget da Twelve Data, broker, posicoes, PnL, historico ou ordens.

### FASE 2.5B.2B - Setup/Blocker Taxonomy Clarification

A FASE 2.5B.2B adiciona uma camada DIAGNOSTIC_ONLY / SHADOW_ONLY para clarificar a taxonomia dos bloqueios entre setup de tendencia, reversao, score, BOS, pivo, pullback e Multi-TF.

- A auditoria cruza NO_SETUP_ELIGIBLE Decomposition, Reversal Blocker Routing Audit, Strategy Decision Bridge, BOS/Pivo, Multi-TF, Fibonacci/estrutura e Feed/Fallback Scope.
- A camada separa blocker oficial real de motivo explicativo normalizado, sem alterar a rejeicao oficial.
- Casos como pivo acionado sem BOS viram leitura explicativa `NO_SETUP_WITH_PIVOT_BUT_NO_BOS`, mantendo o candidato bloqueado.
- `REVERSAL_NOT_ELIGIBLE` pode aparecer como contexto de risco sem virar automaticamente motivo primario normalizado para `trend_pullback_breakout`.
- Mensagens sugeridas para UI/email/log sao apenas explicativas e evitam linguagem operacional de compra/entrada.
- `should_keep_blocked` permanece verdadeiro, `safe_to_change_strategy_now` permanece falso e `safe_to_change_threshold_now` permanece falso.
- Nada muda em score real, thresholds, provider, cache/TTL, budget da Twelve Data, broker, posicoes, PnL, historico ou ordens.

### FASE 2.5B.2C - BOS Confirmation Quality Audit

A FASE 2.5B.2C adiciona uma camada DIAGNOSTIC_ONLY / SHADOW_ONLY para explicar por que BOS nao confirmou, separando pavio, fechamento fraco, fechamento insuficiente, BOS falhado, reteste pendente e conflito multi-timeframe.

- A auditoria cruza BOS/Pivo, Multi-TF, Setup/Blocker Taxonomy, NO_SETUP_ELIGIBLE Decomposition, Strategy Decision Bridge, Fibonacci/estrutura e Feed/Fallback Scope.
- Pavio isolado vira `BOS_BY_WICK_ONLY`; fechamento pequeno vira `BOS_BY_CLOSE_WEAK`; rompimento que volta para dentro vira `BOS_FAILED`.
- Casos de 1H sem 4H ou 4H sem 1H aparecem como confirmacao multi-timeframe ausente, sem autoridade operacional.
- Mensagens de UI/email/log explicam a qualidade do fechamento, reteste e relacao 1H/4H sem sugerir compra ou ajuste de score.
- `should_keep_blocked` permanece verdadeiro, `safe_to_change_strategy_now` permanece falso e `safe_to_change_threshold_now` permanece falso.
- Nada muda em score real, thresholds, provider, cache/TTL, budget da Twelve Data, broker, posicoes, PnL, historico ou ordens.

### FASE 2.5B.2D - H1 Confirmation After H4 BOS Audit

A FASE 2.5B.2D adiciona uma camada DIAGNOSTIC_ONLY / SHADOW_ONLY para explicar por que uma estrutura confirmada no 4H ainda nao recebeu confirmacao suficiente no 1H.

- A auditoria cruza BOS/Pivot trace, BOS Confirmation Quality, Multi-TF, Setup/Blocker Taxonomy, NO_SETUP_ELIGIBLE Decomposition, Strategy Decision Bridge, Fibonacci/estrutura e Feed/Fallback Scope.
- Diferencia 1H sem dados suficientes, 1H sem BOS, 1H contra o 4H, 1H lateral, pivo 1H formando, reteste 1H pendente e 1H confirmado apenas em shadow.
- Casos como 4H com `BOS_RETEST_CONFIRMED` e 1H `INSUFFICIENT_DATA` viram `H1_INSUFFICIENT_DATA_AFTER_H4_BOS`, mantendo o bloqueio.
- Mensagens de UI/email/log explicam timing estrutural sem sugerir compra, ignorar 1H ou usar 4H como autoridade operacional.
- `should_keep_blocked` permanece verdadeiro, `safe_to_change_strategy_now` permanece falso e `safe_to_change_threshold_now` permanece falso.
- Nada muda em score real, thresholds, provider, cache/TTL, budget da Twelve Data, broker, posicoes, PnL, historico ou ordens.

### FASE 2.6A - Post-10D Calibration Plan

A FASE 2.6A adiciona uma camada PLANNING_ONLY / DIAGNOSTIC_ONLY / SHADOW_ONLY para consolidar a avaliacao final de 10 dias e propor estudos de microajuste controlado em PAPER.

- O plano consolida classificacao final, saude operacional, feed, provider, confiabilidade do worker, coerencia UI/estado, gargalo dominante e setup dominante.
- Quando o ciclo fecha como `APROVADO COM RESSALVAS`, a proxima etapa recomendada e `FASE 2.6B - Controlled Micro-Adjustment Study`.
- Microajustes sao apenas estudos futuros, sempre com `allowed_now=false`, `requires_next_phase=true` e janela de validacao PAPER nova.
- Acoes como dinheiro real, reducao global de `min_signal_score`, remocao de guards, uso direto de BOS/Fibonacci e aumento de risco entram em `blocked_actions`.
- `should_continue_paper` permanece verdadeiro, `should_start_real_money` permanece falso, `should_change_threshold_now` permanece falso e `should_change_profile_now` permanece falso.
- Nada muda em estrategia real, score real, thresholds, broker, provider, cache/TTL, budget da Twelve Data, ordens, posicoes, wallet, PnL ou historico.

### FASE 2.6B - Controlled Micro-Adjustment Study

A FASE 2.6B adiciona uma camada STUDY_ONLY / DIAGNOSTIC_ONLY / SHADOW_ONLY para comparar microajustes candidatos depois do ciclo 10D aprovado com ressalvas.

- O estudo ranqueia candidatos como `real_rule_mapping_study`, `h1_after_h4_bos_mapping_study`, `secondary_confirmation_micro_adjustment_study`, `breakout_confirmation_quality_study`, `pullback_quality_study` e `clean_cycle_reset_study`.
- Todos os candidatos permanecem com `allowed_now=false`, `requires_next_phase=true`, `can_change_threshold=false`, `can_change_profile=false` e `can_affect_real_trade=false`.
- Contexto desfavoravel ou critico bloqueia qualquer aplicacao imediata e mantem a recomendacao em observacao/estudo.
- `blocked_actions` inclui dinheiro real, reducao global de score minimo, mudanca de perfil agressivo, aplicacao imediata de microajuste, bypass de guards e uso direto de BOS/Fibonacci/H1/H4 como gatilho.
- `should_continue_paper` permanece verdadeiro, `should_start_real_money` permanece falso, `should_change_threshold_now` permanece falso e `should_apply_micro_adjustment_now` permanece falso.
- Nada muda em estrategia real, score real, thresholds, broker, provider, cache/TTL, budget da Twelve Data, ordens, posicoes, wallet, PnL ou historico.

## Instalacao local

### 1. Criar ambiente virtual

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux ou macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Para testes:

```bash
pip install -r requirements-dev.txt
```

### 3. Configurar ambiente

```bash
cp .env.example .env
```

Em desenvolvimento:

```env
AUTH_REQUIRED=false
```

### 4. Rodar a plataforma

```bash
streamlit run app.py
```

### 5. Rodar o worker

```bash
python -m workers.trader_worker
```

## Validacao rapida

Compilacao:

```bash
python -m py_compile app.py paper_trading_engine.py workers/trader_worker.py
```

Testes:

```bash
python -m pytest -q
```

## Deploy

O deploy recomendado usa:

- `web`: Streamlit
- `worker`: loop continuo do trader
- `Postgres`: persistencia compartilhada

Veja o passo a passo em [`DEPLOY_RAILWAY.md`](DEPLOY_RAILWAY.md).

## Modo producao

O app continua em `paper trading` por padrao. O modo producao desta etapa adiciona observabilidade e alertas sem liberar ordens reais.

Para ativar:

```env
PRODUCTION_MODE=true
ALERT_EMAIL_ENABLED=true
ALERT_EMAIL_PROVIDER=smtp
ALERT_EMAIL_TO=oliveirasamuel03@gmail.com
ALERT_EMAIL_FROM=
SMTP_HOST=seu_smtp
SMTP_PORT=587
SMTP_USERNAME=seu_usuario
SMTP_PASSWORD=sua_senha
SMTP_USE_TLS=true
RESEND_API_KEY=
ALERT_HEARTBEAT_MAX_DELAY_SECONDS=180
ALERT_MAX_CONSECUTIVE_ERRORS=3
ALERT_FEED_FALLBACK_MAX_MINUTES=15
ALERT_COOLDOWN_MINUTES=30
ALERT_SEND_RECOVERY_EMAIL=true
BROKER_PROVIDER=paper
BROKER_MODE=paper
```

Para cloud, o provider recomendado agora e `resend`, mantendo `smtp` como fallback:

```env
ALERT_EMAIL_PROVIDER=resend
ALERT_EMAIL_FROM=Trade Ops Desk <alerts@seudominio.com>
ALERT_EMAIL_TO=oliveirasamuel03@gmail.com
RESEND_API_KEY=re_xxxxxxxxx
RESEND_API_BASE=https://api.resend.com/emails
BROKER_PROVIDER=paper
BROKER_MODE=paper
```

## Email reporting dos relatorios

O worker pode enviar relatorios operacionais por email sem sair do PAPER mode:

- diario: uma vez por dia
- semanal: uma vez por semana
- bloco de avaliacao: dias 10, 20 e 30
- final de 30 dias: uma unica vez

Configuracao minima:

```env
REPORT_EMAIL_ENABLED=true
REPORT_EMAIL_PROVIDER=smtp
REPORT_EMAIL_TO=seu@email.com
REPORT_EMAIL_FROM=desk@email.com
REPORT_EMAIL_SMTP_HOST=seu_smtp
REPORT_EMAIL_SMTP_PORT=587
REPORT_EMAIL_SMTP_USERNAME=seu_usuario
REPORT_EMAIL_SMTP_PASSWORD=sua_senha
REPORT_EMAIL_USE_TLS=true
REPORT_EMAIL_DAILY_ENABLED=true
REPORT_EMAIL_WEEKLY_ENABLED=true
REPORT_EMAIL_10DAY_ENABLED=true
REPORT_EMAIL_FINAL_ENABLED=true
```

Como testar localmente:

1. configure as variaveis acima no `.env`
2. rode `streamlit run app.py`
3. rode `python -m workers.trader_worker`
4. confira em `Controle do Bot` a secao `Email reporting`
5. verifique `storage/runtime/bot_log.csv` para eventos `[report_email_delivery]`

Comportamento em falha e garantias de seguranca:

- o envio e sempre `best-effort`
- se SMTP ou Resend falhar, o worker continua o ciclo normalmente
- os relatorios locais continuam sendo gerados mesmo sem email
- a entrega por email nao dispara trades, nao muda score, nao muda broker e nao toca em ordem real
- PAPER mode continua obrigatorio em todos os emails e no fluxo operacional

Observacao importante:

- o `smtp` continua suportado para ambiente local ou legado
- no `resend`, o campo `from` precisa usar um dominio verificado para entrega fora do modo de teste
- o app continua em `paper` e nao libera ordens reais nesta etapa

Como validar:

- abrir `Controle do Bot`
- conferir o bloco `Modo producao`
- usar `Recalcular saude`
- usar `Testar email`
- confirmar que o broker permanece `Simulado`

Mesmo com `PRODUCTION_MODE=true`, a etapa atual nao habilita envio de ordem real.

## Trava diaria de perda (paper trading)

A plataforma agora possui uma trava explicita de limite de perda diaria no modo paper:

- metrica usada: `PnL realizado do dia UTC` (somente trades `SELL` fechados)
- quando a perda consumida atinge ou ultrapassa `DAILY_LOSS_LIMIT_BRL`, novas entradas ficam bloqueadas
- posicoes ja abertas continuam com a gestao normal do robo
- estado auditavel e persistente em `state.risk` com motivo, timestamp de bloqueio e valor observado
- desbloqueio automatico na virada do dia UTC

Variavel de ambiente:

```env
DAILY_LOSS_LIMIT_BRL=100
```

## Contexto de mercado cripto

O app agora possui uma camada leve de contexto de mercado para cripto, pensada apenas como filtro auxiliar em `paper trading`.

O contexto considera:

- movimento recente do `BTC`
- volatilidade do ambiente cripto
- regime predominante (`tendencia`, `lateral`, `baixa`)
- consistencia da watchlist cripto observada

Status possiveis:

- `FAVORAVEL`
- `NEUTRO`
- `DESFAVORAVEL`
- `CRITICO`

Como influencia os sinais:

- o contexto **nao gera ordem**
- o contexto **nao usa noticia como gatilho**
- quando o ambiente fica `DESFAVORAVEL`, o robo endurece o score minimo para entradas de cripto
- quando o ambiente fica `CRITICO`, novas entradas fracas de cripto podem ser bloqueadas
- se o calculo do contexto falhar, o sistema volta para `NEUTRO` sem travar o worker

Onde acompanhar:

- `Trader`: painel simples com contexto atual, motivo, impacto e sinais barrados
- `Controle do Bot`: painel administrativo com status, regime, contexto por periodo e impacto estimado

Observacao importante:

- esta camada serve apenas para filtrar ou endurecer sinais
- ordens reais continuam desabilitadas
- o broker permanece em `paper`

## Alertas macro de risco

A plataforma pode carregar eventos macro configurados manualmente para atuar como filtro operacional de risco em `paper trading`.

Esta camada:

- nao gera compra ou venda
- nao aprova trades sozinha
- apenas reduz confianca, endurece setups frageis ou bloqueia novas entradas quando o risco macro esta alto
- preserva a autoridade final da estrategia interna e dos guards fixos do codigo
- ignora eventos invalidos com seguranca e nao interrompe o worker

Configuracao minima:

```env
MACRO_ALERTS_ENABLED=false
MACRO_ALERT_PRE_EVENT_MINUTES=60
MACRO_ALERT_EVENT_WINDOW_MINUTES=30
MACRO_ALERT_POST_EVENT_MINUTES=60
MACRO_ALERTS_FILE=storage/runtime/macro_events.json
```

Exemplo de arquivo local:

```json
[
  {
    "title": "FOMC",
    "currency": "USD",
    "datetime": "2026-05-01T18:00:00Z",
    "impact_level": "HIGH",
    "enabled": true
  }
]
```

Status de janela:

- `INACTIVE`: sem evento dentro da janela de risco
- `PRE_EVENT`: evento se aproximando
- `IN_EVENT`: evento em andamento
- `POST_EVENT`: estabilizacao apos o evento

Como influencia os sinais:

- `TREND_FOLLOWING_SWING` pode seguir elegivel, mas precisa de margem de score maior
- `BREAKOUT_CONFIRMED` fica mais seletivo durante janela macro
- `REVERSAL_V_PATTERN` e o setup mais restrito
- se o feed estiver em fallback e houver alerta macro alto, novas entradas sao bloqueadas

Onde acompanhar:

- `Trader`: bloco `Alerta macro de risco`
- `Controle do Bot`: painel administrativo com impacto, janela, evento, penalidade e efeito operacional

## FASE 3A - External signal audit

A FASE 3A adiciona uma fundacao leve para providers e um intake de sinais externos em modo `audit-only`.

Esta camada:

- recebe, valida, registra e persiste sinais externos apenas para auditoria
- nao aprova trades
- nao abre nem fecha posicoes
- nao altera score, thresholds, estrategia, guards, broker ou feed operacional
- nao substitui o Twelve Data nem a cadeia atual de dados de mercado
- preserva `PAPER TRADING` obrigatorio
- rejeita com seguranca payloads invalidos, expirados, duplicados, fora da watchlist ou de fonte nao permitida

Configuracao segura:

```env
EXTERNAL_SIGNAL_WEBHOOK_ENABLED=false
EXTERNAL_SIGNAL_SECRET=
EXTERNAL_SIGNAL_MAX_AGE_SECONDS=300
EXTERNAL_SIGNAL_ALLOWED_SOURCES=
EXTERNAL_SIGNAL_ALLOWED_TIMEFRAMES=30s,1m,5m,15m,1h,1d
EXTERNAL_SIGNAL_DEDUPE_SECONDS=300
EXTERNAL_SIGNAL_TEST_PANEL_ENABLED=false
```

Com os defaults acima, nenhum sinal externo e aceito. Para validacao controlada, habilite o webhook, configure um segredo e liste fontes permitidas em `EXTERNAL_SIGNAL_ALLOWED_SOURCES`.

Payload esperado pela funcao de intake auditavel:

```json
{
  "token": "...",
  "source": "tradingview",
  "strategy": "alerta_externo",
  "symbol": "BTC-USD",
  "timeframe": "15m",
  "side": "BUY",
  "alert_price": 65000.0,
  "score": 0.72,
  "ts": "2026-04-24T12:00:00+00:00",
  "extra": {}
}
```

Na estrutura Streamlit atual, esta etapa nao adiciona uma rota HTTP arriscada. A validacao fica disponivel em `core.external_signals.process_external_signal_payload(...)` para uso futuro por uma rota segura ou servico separado. Os paineis `Trader` e `Controle do Bot` mostram o ultimo status de auditoria e deixam claro que sinal externo nao e gatilho de execucao.

### FASE 3B - Test harness interno

A FASE 3B adiciona um painel admin-only em `Controle do Bot` para testar payloads manualmente, sem rota publica.

Regras do painel:

- fica desativado por padrao com `EXTERNAL_SIGNAL_TEST_PANEL_ENABLED=false`
- quando desativado, mostra apenas leitura e nao renderiza controles de envio
- quando ativado para validacao controlada, exige token digitado no painel
- o token nao e exibido, logado ou persistido
- o resultado `ACCEPTED_FOR_AUDIT` significa somente aceite para auditoria, nunca aceite de trade
- os eventos recentes aparecem em `Trader` e `Controle do Bot`, limitados a exibicao dos 10 mais recentes
- `state.external_signal.recent_events` continua limitado a 20 eventos e nao altera historico CSV

Para um teste controlado, configure temporariamente:

```env
EXTERNAL_SIGNAL_WEBHOOK_ENABLED=true
EXTERNAL_SIGNAL_TEST_PANEL_ENABLED=true
EXTERNAL_SIGNAL_SECRET=um_token_de_teste
EXTERNAL_SIGNAL_ALLOWED_SOURCES=tradingview
```

Depois da validacao, volte `EXTERNAL_SIGNAL_TEST_PANEL_ENABLED=false`.

## Calibration Preview - FASE 2.6

A camada `Calibration Preview` mostra sinais rejeitados que ficaram perto do score minimo sob condicoes seguras. Ela e apenas diagnostica:

- nao aprova trades
- nao reduz thresholds
- nao altera estrategia, score, guards, broker, macro alert ou sinais externos
- nao abre nem fecha posicoes
- preserva `PAPER TRADING`

Configuracao:

```env
CALIBRATION_PREVIEW_ENABLED=true
CALIBRATION_PREVIEW_MARGIN=0.04
CALIBRATION_PREVIEW_MAX_EXAMPLES=10
```

Com `min_signal_score=0.74` e margem `0.04`, o preview observa sinais rejeitados entre `0.70` e `0.74`, desde que o feed esteja `LIVE`, sem fallback no ciclo, macro alert inativo, contexto `FAVORAVEL`/`NEUTRO`, broker `PAPER`, ativo na watchlist e sem trava diaria ativa.

Use esta leitura apenas para decidir se ha base para estudar calibracao futura. Nenhuma calibracao e aplicada automaticamente.

## Multi-Timeframe Swing Diagnostic - FASE 2.5

A camada `FASE 2.5 - Diagnostico Multi-Timeframe Swing` compara 1D, 4H e 1H em modo `SHADOW_ONLY` para explicar estrutura swing sem virar gatilho de entrada.

- `1D` representa direcao principal e filtro macro da estrategia.
- `4H` representa estrutura intermediaria, pivo, BOS, pullback e continuacao.
- `1H` representa confirmacao de retomada/timing estrutural.
- `15m` permanece visual/refino de UI, sem autoridade operacional.

Seguranca:

- nao aprova trades
- nao altera score real nem `min_signal_score`
- nao muda thresholds, broker, PnL, historico ou posicoes oficiais
- nao transforma Fibonacci, BOS, 4H ou 1H em gatilho
- preserva `PAPER TRADING`

Configuracao:

```env
MULTITF_SWING_AUDIT_ENABLED=true
MULTITF_SWING_TIMEFRAMES=1d,4h,1h
MULTITF_SWING_CACHE_TTL_SECONDS=900
MULTITF_SWING_MAX_SYMBOLS=5
MULTITF_SWING_REQUIRE_LIVE_FEED=true
MULTITF_SWING_SHADOW_ONLY=true
```

Para proteger os limites da Twelve Data, a implementacao inicial reaproveita os candles ja carregados no ciclo operacional e deriva 4H/1D por resample local. O diagnostico registra `estimated_provider_calls=0`, `cache_status=cycle_data_resample_only` e degrada se o feed nao estiver `LIVE`.

`MULTITF_SWING_SHADOW_ONLY` e documentado como reforco operacional, mas a seguranca no codigo forca esta fase a permanecer `SHADOW_ONLY`.

Procure nos logs por `[multi_tf_swing_audit_summary]`, `[multi_tf_swing_audit_candidate]`, `[multi_tf_swing_alignment]`, `[multi_tf_swing_missing_confirmation]` e `[multi_tf_swing_provider_guard]`.

### FASE 2.5A - Intraday Multi-Timeframe Fetcher

A camada `multi_timeframe_intraday_fetcher` busca candles reais 4H/1H para alimentar a auditoria 2.5. Ela tambem e `SHADOW_ONLY`:

- nao aprova entrada
- nao altera score real
- nao altera broker, PnL, historico ou posicoes oficiais
- nao transforma 4H, 1H, BOS, pivo ou Fibonacci em gatilho
- preserva `PAPER TRADING`

Configuracao:

```env
MULTITF_INTRADAY_FETCH_ENABLED=true
MULTITF_INTRADAY_TIMEFRAMES=4h,1h
MULTITF_INTRADAY_CACHE_TTL_SECONDS=1800
MULTITF_INTRADAY_MAX_SYMBOLS=5
MULTITF_INTRADAY_MAX_CALLS_PER_CYCLE=3
MULTITF_INTRADAY_REQUIRE_LIVE_FEED=true
MULTITF_INTRADAY_SHADOW_ONLY=true
MULTITF_INTRADAY_PROVIDER_BUDGET_MODE=conservative
```

Para proteger a Twelve Data, a busca usa cache por `symbol+timeframe`, TTL minimo conservador, limite de chamadas por ciclo e prioridade por simbolos mais promissores: top da auditoria estrutural, top do preview/calibracao, top da auditoria Fibonacci, `BTC-USD` e depois a watchlist.

Se o budget acabar ou o feed nao estiver `LIVE`, a camada registra `provider_budget_guard_active`, `provider_guard_reason` e mantem a auditoria como `INSUFFICIENT_DATA`/`PROVIDER_BLOCKED`. Dados fallback/sinteticos nao sao tratados como intraday real.

Procure nos logs por `[multi_tf_intraday_fetch_summary]`, `[multi_tf_intraday_fetch_cache]`, `[multi_tf_intraday_fetch_provider_guard]`, `[multi_tf_intraday_fetch_symbol]`, `[multi_tf_intraday_fetch_error]` e `[multi_tf_swing_audit_with_intraday]`.

## Strategy Bottleneck - FASE 2.7

A camada `Strategy Bottleneck` detalha quais filtros internos mais bloqueiam os sinais rejeitados. Ela e apenas diagnostica:

- nao aprova trades
- nao reduz thresholds
- nao altera `min_signal_score`
- nao altera calculo de score, estrategia, guards, broker, macro alert ou sinais externos
- nao abre nem fecha posicoes
- preserva `PAPER TRADING`

A leitura separa gargalos como `SCORE_BELOW_MIN`, `RSI_OUT_OF_RANGE`, `TREND_NOT_CONFIRMED`, `MOMENTUM_WEAK`, `SECONDARY_CONFIRMATION_WEAK`, `VOLATILITY_FILTER`, `CONTEXT_FILTER`, `FEED_BLOCK` e `GUARD_BLOCK`.

Use esta camada para entender por que `trend_pullback_breakout` esta rejeitando sinais antes de qualquer calibracao futura. Nenhuma calibracao e aplicada automaticamente.

## Ajuste Fino Conservador - FASE 2

A camada `Ajuste Fino FASE 2` aplica um unico relaxamento pequeno e reversivel no setup dominante `trend_pullback_breakout`: quando a unica falha e a confirmacao secundaria `breakout_not_confirmed`, o setup pode continuar elegivel se `breakout_20` estiver no maximo `0.005` abaixo de `breakout_min`.

Este ajuste nao reduz `min_signal_score` global, nao altera ticket, nao aumenta posicoes, nao muda broker e nao ativa ordem real. Ele so pode aplicar quando todos os guards abaixo estao limpos:

- broker e validacao em `PAPER`
- feed operacional `LIVE` com provider conhecido
- contexto diferente de `CRITICO`
- macro alert inativo
- trava diaria de perda liberada
- limite de posicoes com espaco disponivel
- score ajustado ja acima do minimo efetivo
- tendencia, RSI, ATR, pullback e momentum primario/medio aprovados
- unica rejeicao original igual a `breakout_not_confirmed`

A auditoria aparece em `Trader`, `Controle do Bot`, estado compartilhado e logs `[phase2_fine_tune_summary]` com contadores de aplicado/bloqueado por guard.

### Ajuste Fino FASE 2.1

A camada `phase2_multi_minor_confirmation_fine_tune` cobre o caso observado no ciclo limpo: sinais quase elegiveis com pequenas falhas combinadas de confirmacao secundaria, momentum e score muito proximo do minimo.

Ela so atua no setup `trend_pullback_breakout` e apenas quando os motivos de rejeicao estao restritos a:

- `breakout_not_confirmed`
- `confidence_too_low`
- `score_below_minimum`

Guards obrigatorios:

- `PAPER` confirmado e broker real sem autoridade
- feed operacional `LIVE`
- provider efetivo `twelvedata` ou `mixed`
- fallback atual igual a `0`
- contexto diferente de `CRITICO`
- macro alert inativo
- trava diaria de perda liberada
- posicoes abertas abaixo do limite
- score gap no maximo `0.015`
- tendencia, RSI, ATR e pullback limpos
- breakout e momentum apenas marginalmente abaixo dos limites

Se qualquer falha primaria aparecer, como `trend_not_confirmed`, `reversal_not_eligible`, `volatility_out_of_range`, feed fallback, contexto critico, daily risk ativo ou limite de posicao atingido, o ajuste fica bloqueado. A auditoria aparece como `Ajuste Fino FASE 2.1`, nos logs `[phase2_1_fine_tune_summary]`, `[phase2_1_fine_tune_applied]` e `[phase2_1_fine_tune_blocked]`.

## Watchlist padrao da validacao swing

Para a fase atual de validacao swing em `paper trading`, a watchlist padrao foi consolidada para um escopo `CRYPTO ONLY` com cinco ativos:

- `BTC-USD`
- `ETH-USD`
- `BNB-USD`
- `SOL-USD`
- `LINK-USD`

Motivo da selecao:

- reduzir ruido e excesso de comparacoes irrelevantes
- melhorar a coerencia dos testes e dos relatorios
- alinhar a watchlist com o modulo de contexto de mercado cripto
- priorizar liquidez, leitura tecnica e estabilidade operacional

Justificativa objetiva por ativo:

- `BTC-USD`: referencia principal do mercado, maior liquidez e melhor leitura estrutural. Risco principal: pode ficar mais lento em algumas janelas. Perfil: consistencia / tendencia.
- `ETH-USD`: segundo ativo-base da watchlist, muito liquido e com excelente relevancia. Risco principal: costuma ampliar movimentos do BTC em periodos de stress. Perfil: consistencia / tendencia com mais amplitude.
- `BNB-USD`: ativo intermediario entre estabilidade e oportunidade, com boa liquidez e leitura relativamente organizada. Risco principal: impacto especifico do ecossistema relacionado. Perfil: consistencia moderada / tendencia.
- `SOL-USD`: ativo de maior oportunidade, com swings mais fortes e boa chance de tendencia. Risco principal: volatilidade mais alta e ruido maior. Perfil: volatilidade / tendencia.
- `LINK-USD`: diversificacao tatica dentro da watchlist, com leitura tecnica util em varios ciclos. Risco principal: pode perder tracao em alguns periodos. Perfil: tendencia / oportunidade moderada.

Ativos desaconselhados nesta fase:

- memecoins
- altcoins de baixa liquidez
- ativos com feed instavel
- ativos excessivamente erraticos para uma validacao inicial de 10 dias

Como alterar no app:

- abrir `Trader`
- entrar em `Modo avancado`
- editar `Lista de ativos`
- ou usar `Aplicar watchlist recomendada`

Compatibilidade da etapa:

- a ETAPA 3 continua intacta
- o modo segue 100% `paper`
- o contexto de mercado cripto reaproveita esta watchlist sem duplicar logica
- nao ha habilitacao de ordens reais nesta etapa

## Configuracao inicial do ciclo swing

Para iniciar a validacao swing de `10 dias` com mais disciplina, a base operacional do ciclo atual foi alinhada para:

- capital inicial de validacao: `R$ 1.000,00`
- entrada padrao por operacao: `R$ 100,00`
- maximo de posicoes abertas: `2`
- modo exibido da fase: `swing_10d_crypto`
- execucao: `paper trading` obrigatoria

Mapeamento na estrutura atual:

- `wallet_value` e `paper_state.initial_capital`: capital inicial do ciclo
- `trader.ticket_value`: valor base por entrada
- `trader.max_open_positions`: limite de exposicao simultanea
- `validation.validation_mode`: continua interno como `swing_10d` para preservar compatibilidade
- `validation.validation_mode_label`: exposto como `swing_10d_crypto`
- `security.real_mode_enabled`: continua `false` por padrao nesta fase

Observacoes importantes:

- o robo ja respeita saldo simulado insuficiente e o limite maximo de posicoes antes de abrir novas entradas
- os defaults novos valem para novos ciclos e para estados legados ainda nao iniciados
- para recomecar um ciclo limpo com esses parametros, use `Salvar configuracao avancada` e depois `Resetar modulo trader`

Novos indicadores de PnL no Trader:

- `PnL Acumulado`: resultado total realizado
- `Ultima Operacao`: resultado do trade fechado mais recente
- `PnL em Aberto`: resultado das posicoes ainda abertas
- `Retorno do ciclo`: soma do realizado com o aberto sobre o capital inicial atual

## Retencao e validacao semanal

O historico operacional agora aplica uma politica segura de retencao:

- mantem no runtime os ultimos `60 dias` por padrao
- arquiva automaticamente relatorios e logs mais antigos por mes
- nao apaga diretamente nada antes de arquivar
- executa a rotina uma vez por dia dentro do worker
- continua compativel com local e com `DATABASE_URL`

Arquivos que entram na retencao:

- `storage/runtime/trade_reports.csv`
- `storage/runtime/bot_log.csv`

`trader_orders.csv` permanece no runtime por padrao nesta etapa para preservar a aba `Ordens` sem alterar seu comportamento. Se quiser avaliar essa expansao depois, existe a flag `RETENTION_ARCHIVE_TRADER_ORDERS`.

Resumos semanais:

- quantidade de trades
- win rate semanal
- payoff semanal
- pnl semanal
- melhores e piores ativos
- trades fechados cedo demais
- trades mantidos tempo demais
- erros operacionais
- percentual de ciclos em fallback
- sugestoes analiticas
- observacao final de estabilidade

Como acessar:

- abrir `Historico`
- entrar na aba `Validacao Semanal`
- selecionar a semana
- exportar o consolidado semanal em CSV quando necessario

Como validar que o historico nao foi quebrado:

- a aba `Relatorios Trader` continua lendo o runtime recente normalmente
- a aba `Logs` continua mostrando o runtime atual
- o resumo semanal passa a mostrar os dados arquivados de forma consolidada

## Seguranca

- senhas com hash `bcrypt`
- autenticacao obrigatoria em producao
- `real mode` restrito a admin com confirmacao de senha
- segredos fora do codigo
- arquivos de runtime e logs fora do versionamento

## Compatibilidade

Wrappers de compatibilidade na raiz foram mantidos apenas para o fluxo de trade legado. O caminho oficial da interface continua sendo `app.py` e `pages/`.
