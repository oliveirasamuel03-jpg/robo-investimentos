# Robo de Investimentos

Plataforma de trading e pesquisa em Streamlit com dois fluxos principais:

- `Trader`: operacao automatizada em paper trading com worker continuo.
- `Investimento`: pesquisa institucional, backtest, walk-forward e analise de carteira.

O projeto foi endurecido para uso local e deploy com separacao clara entre codigo-fonte, estado de execucao, autenticacao e persistencia compartilhada.

## Visao geral

Esta base combina:

- interface Streamlit multipagina
- worker continuo para o trader
- paper trading com sincronizacao de estado
- persistencia local em arquivos JSON/CSV
- persistencia compartilhada em PostgreSQL quando `DATABASE_URL` estiver configurado
- autenticacao com hash de senha e controle por perfil

O objetivo e manter o comportamento atual do app sem simplificar a plataforma, mas deixando o repositório mais previsivel para desenvolvimento e deploy.

## Arquitetura atual

### Interface

- `app.py`: ponto de entrada principal do app
- `pages/0_Login.py`: login e bootstrap do primeiro admin
- `pages/1_Trader.py`: painel do trader
- `pages/2_Investimento.py`: pesquisa institucional
- `pages/3_Carteira.py`: carteira e capital
- `pages/4_Controle_do_Bot.py`: acoes administrativas
- `pages/5_Historico.py`: auditoria de ordens e logs

### Nucleo

- `core/config.py`: configuracao central, env vars e diretorios
- `core/persistence.py`: persistencia local e PostgreSQL
- `core/state_store.py`: estado do app trader/investimento
- `core/auth/`: autenticacao, sessao e autorizacao
- `core/ui.py`: utilitarios visuais compartilhados

### Engines

- `engines/trader_engine.py`: orquestracao do ciclo do trader
- `engines/quant_bridge.py`: ponte de compatibilidade entre engines
- `engines/investment_research_engine.py`: pipeline de pesquisa
- `paper_trading_engine.py`: simulacao e estado do paper trading

### Execucao

- `workers/trader_worker.py`: loop continuo do trader
- `bot_engine.py` e `bot_runner.py`: compatibilidade com fluxo legado

## Estrutura de pastas

```text
.
|-- app.py
|-- core/
|   |-- auth/
|   |-- config.py
|   |-- persistence.py
|   |-- state_store.py
|   `-- ui.py
|-- engines/
|-- institutional/
|-- pages/
|-- portfolio/
|-- storage/
|   |-- cache/
|   |-- reports/
|   `-- runtime/
|-- workers/
|-- requirements.txt
|-- requirements-dev.txt
`-- DEPLOY_RAILWAY.md
```

## Persistencia e runtime

O projeto opera em dois modos:

- sem `DATABASE_URL`: usa arquivos locais em `storage/runtime/`
- com `DATABASE_URL`: usa PostgreSQL para estado compartilhado e mantem o filesystem como fallback local

Arquivos de runtime principais:

- `storage/runtime/bot_state.json`
- `storage/runtime/paper_state.json`
- `storage/runtime/paper_trades.json`
- `storage/runtime/trader_orders.csv`
- `storage/runtime/investor_orders.csv`
- `storage/runtime/bot_log.csv`
- `storage/runtime/auth_users.json`

Arquivos de runtime nao devem ser versionados.

## Variaveis de ambiente

Copie o arquivo de exemplo:

```bash
cp .env.example .env
```

Variaveis suportadas:

- `APP_TITLE`: titulo exibido no app
- `APP_ENV`: `development` ou `production`
- `ROBO_STORAGE_DIR`: diretorio de armazenamento local
- `DATABASE_URL`: ativa PostgreSQL quando preenchida
- `AUTH_REQUIRED`: força autenticacao mesmo em ambiente local
- `AUTH_SESSION_TIMEOUT_MINUTES`: timeout de sessao
- `ADMIN_USERNAME`: bootstrap opcional do primeiro admin
- `ADMIN_PASSWORD`: bootstrap opcional do primeiro admin

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

Em desenvolvimento, a autenticacao pode permanecer opcional com:

```env
AUTH_REQUIRED=false
```

### 4. Rodar o app

```bash
streamlit run app.py
```

### 5. Rodar o worker localmente

```bash
python -m workers.trader_worker
```

## Testes basicos

Compilacao:

```bash
python -m py_compile app.py paper_trading_engine.py
```

Suite basica:

```bash
python -m pytest -q
```

## Deploy

O deploy recomendado usa:

- `web`: Streamlit
- `worker`: loop continuo do trader
- `Postgres`: persistencia compartilhada

Guia passo a passo:

- veja [`DEPLOY_RAILWAY.md`](DEPLOY_RAILWAY.md)

## Seguranca

- senhas sao armazenadas com hash `bcrypt`
- autenticacao e opcional em dev, mas obrigatoria em ambiente de producao
- `real mode` exige conta admin e confirmacao de senha
- segredos nao devem ser hardcoded no codigo
- `.env`, `storage/runtime/` e logs locais ficam fora do versionamento

## Compatibilidade e legado

Alguns arquivos na raiz foram mantidos como wrappers de compatibilidade para fluxos antigos. O caminho oficial para a interface continua sendo `app.py` e `pages/`.

## Comandos uteis

Subir app:

```bash
streamlit run app.py
```

Rodar worker:

```bash
python -m workers.trader_worker
```

Executar testes:

```bash
python -m pytest -q
```
