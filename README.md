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
- `RETENTION_ENABLED`
- `RETENTION_DAYS`
- `RETENTION_RUN_INTERVAL_HOURS`
- `RETENTION_ARCHIVE_TRADER_ORDERS`
- `WEEKLY_REPORT_RUNTIME_WEEKS`

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
