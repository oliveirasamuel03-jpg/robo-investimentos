# Deploy no Railway

Este projeto deve ser publicado com tres componentes no mesmo projeto Railway:

- `web`: interface Streamlit
- `worker`: loop continuo do trader
- `Postgres`: persistencia compartilhada

## Antes de comecar

Confirme que a branch com as alteracoes ja foi mergeada na `main`.

## Passo 1. Criar ou abrir o projeto no Railway

1. Entre no Railway.
2. Crie um projeto novo ou abra o projeto existente.
3. Conecte o repositorio `oliveirasamuel03-jpg/robo-investimentos`.

## Passo 2. Criar o service web

1. Clique em `Add Service`.
2. Escolha `GitHub Repo`.
3. Selecione o repositorio.
4. No service `web`, abra `Settings`.
5. Em `Start Command`, configure:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

6. Gere um dominio publico em `Networking` ou `Domains`.

## Passo 3. Criar o service worker

1. No mesmo projeto, clique em `Add Service`.
2. Escolha o mesmo repositorio.
3. Renomeie o service para algo como `worker`.
4. Em `Settings`, configure:

```bash
python -m workers.trader_worker
```

O worker nao precisa de dominio publico.

## Passo 4. Adicionar Postgres

1. Clique em `Add Service`.
2. Escolha `Database` e depois `Postgres`.
3. Aguarde o Railway provisionar o banco.

Quando o banco ficar pronto, o Railway cria a variavel `DATABASE_URL`.

## Passo 5. Variaveis de ambiente

No service `web` e no service `worker`, configure:

```env
APP_ENV=production
AUTH_REQUIRED=true
PRODUCTION_MODE=true
BROKER_PROVIDER=paper
BROKER_MODE=paper
```

Se quiser alertas por email no Railway, configure tambem no `web` e no `worker`:

```env
ALERT_EMAIL_ENABLED=true
ALERT_EMAIL_PROVIDER=resend
ALERT_EMAIL_TO=oliveirasamuel03@gmail.com
ALERT_EMAIL_FROM=Trade Ops Desk <alerts@seudominio.com>
RESEND_API_KEY=re_xxxxxxxxx
RESEND_API_BASE=https://api.resend.com/emails
ALERT_HEARTBEAT_MAX_DELAY_SECONDS=180
ALERT_MAX_CONSECUTIVE_ERRORS=3
ALERT_FEED_FALLBACK_MAX_MINUTES=15
ALERT_COOLDOWN_MINUTES=30
ALERT_SEND_RECOVERY_EMAIL=true
```

Opcional para bootstrap do primeiro admin:

```env
ADMIN_USERNAME=seu_admin
ADMIN_PASSWORD=sua_senha_forte
```

Observacoes:

- se `ADMIN_USERNAME` e `ADMIN_PASSWORD` nao forem definidos, o app permite criar o primeiro admin na tela de login
- `DATABASE_URL` deve estar presente no `web` e no `worker`
- `ROBO_STORAGE_DIR` e opcional; em producao com Postgres o estado principal vem do banco
- `BROKER_PROVIDER=paper` e `BROKER_MODE=paper` mantem a etapa em simulacao
- `PRODUCTION_MODE=true` ativa monitoramento e alertas, mas nao libera ordem real
- em cloud, prefira `ALERT_EMAIL_PROVIDER=resend`; `smtp` continua disponivel como fallback

## Passo 6. Fazer deploy

1. Rode o deploy do `web`.
2. Rode o deploy do `worker`.
3. Confirme que os dois services ficaram saudaveis.

## Passo 7. Validacao pos-deploy

### Web

1. Abra o dominio do `web`.
2. Confirme que a tela de login abre.
3. Faça login com admin.
4. Entre em `Trader`.

### Worker

1. Coloque o bot em `RUNNING`.
2. Aguarde 1 ou 2 minutos.
3. Valide:
   - `worker_status` como `online`
   - `worker_heartbeat` atualizado
   - `cash` e `equity` carregando
   - ordens e historico sendo atualizados
   - `Controle do Bot` mostrando `Modo producao`
   - broker em `Simulado`
   - email de teste funcionando, se SMTP estiver configurado

## Troubleshooting

### O web nao sobe

Cheque:

- `Start Command` do Streamlit
- dominio gerado no service correto
- logs do deploy

### O worker fica offline

Cheque:

- `Start Command` do worker
- `DATABASE_URL` disponivel
- logs do worker
- bot em `RUNNING`

### O app abre mas o estado parece inconsistente

Quase sempre significa que:

- o `web` e o `worker` nao estao lendo o mesmo `DATABASE_URL`
- ou um dos services ainda esta em commit antigo

Nesses casos:

1. confirme a `main` atual no GitHub
2. use `Redeploy` no `web`
3. use `Redeploy` no `worker`

## Comandos de referencia

Web:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

Worker:

```bash
python -m workers.trader_worker
```
