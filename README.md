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
|   |-- cache/
|   |-- reports/
|   `-- runtime/
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
- `ALERT_EMAIL_TO`
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_USE_TLS`
- `ALERT_HEARTBEAT_MAX_DELAY_SECONDS`
- `ALERT_MAX_CONSECUTIVE_ERRORS`
- `ALERT_FEED_FALLBACK_MAX_MINUTES`
- `ALERT_COOLDOWN_MINUTES`
- `ALERT_SEND_RECOVERY_EMAIL`

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
ALERT_EMAIL_TO=oliveirasamuel03@gmail.com
SMTP_HOST=seu_smtp
SMTP_PORT=587
SMTP_USERNAME=seu_usuario
SMTP_PASSWORD=sua_senha
SMTP_USE_TLS=true
ALERT_HEARTBEAT_MAX_DELAY_SECONDS=180
ALERT_MAX_CONSECUTIVE_ERRORS=3
ALERT_FEED_FALLBACK_MAX_MINUTES=15
ALERT_COOLDOWN_MINUTES=30
ALERT_SEND_RECOVERY_EMAIL=true
BROKER_PROVIDER=paper
BROKER_MODE=paper
```

Como validar:

- abrir `Controle do Bot`
- conferir o bloco `Modo producao`
- usar `Recalcular saude`
- usar `Testar email`
- confirmar que o broker permanece `Simulado`

Mesmo com `PRODUCTION_MODE=true`, a etapa atual nao habilita envio de ordem real.

## Seguranca

- senhas com hash `bcrypt`
- autenticacao obrigatoria em producao
- `real mode` restrito a admin com confirmacao de senha
- segredos fora do codigo
- arquivos de runtime e logs fora do versionamento

## Compatibilidade

Wrappers de compatibilidade na raiz foram mantidos apenas para o fluxo de trade legado. O caminho oficial da interface continua sendo `app.py` e `pages/`.
