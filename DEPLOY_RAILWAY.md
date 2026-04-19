# Deploy no Railway

Este projeto deve ser publicado no Railway com dois services separados apontando para o mesmo repositório:

- `web`: interface Streamlit
- `worker`: processo 24h do trader

## Por que separar

O Railway recomenda worker como um service separado e always-on. Isso evita misturar a interface web com o loop contínuo do robô.

## Service 1: Web

Crie um service a partir deste repositório e configure:

- Start Command:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

- Healthcheck:
  use `/` ou deixe sem healthcheck customizado

## Service 2: Worker

Crie outro service no mesmo projeto Railway usando o mesmo repositório e configure:

- Start Command:

```bash
python -m workers.trader_worker
```

Esse processo roda continuamente e atualiza:

- `worker_status`
- `worker_heartbeat`
- `cash`
- `equity`
- `positions`

## Service 3: Postgres compartilhado

Para o `web` e o `worker` enxergarem o mesmo estado em produção, adicione também um service de Postgres no mesmo projeto Railway.

Quando o Railway criar a variável `DATABASE_URL`, o projeto passa a usar o banco automaticamente para compartilhar:

- `bot_state`
- `paper_state`
- `paper_trades`
- `trader_orders`
- `investor_orders`
- `bot_log`

Sem `DATABASE_URL`, o projeto continua em modo local usando arquivos na pasta `storage/`.

## Variáveis e storage

- `DATABASE_URL`: fornecida automaticamente pelo Postgres do Railway
- nenhum volume compartilhado é necessário para o estado principal quando o banco estiver ativo
- a pasta `storage/` continua sendo o fallback local para desenvolvimento

## Fluxo recomendado

1. Fazer merge do PR com as correções do paper trading.
2. Deployar o `web` com o comando do Streamlit.
3. Adicionar um service de Postgres no mesmo projeto.
4. Confirmar que `DATABASE_URL` foi injetada no `web` e no `worker`.
5. Deployar o `worker` com `python -m workers.trader_worker`.
6. Abrir a interface e mudar o bot para `RUNNING`.
7. Conferir se o painel mostra o worker como `online`.

## Checklist de validação

- O site abre no domínio do Railway.
- O botão `Rodar agora` funciona.
- `web` e `worker` têm a variável `DATABASE_URL`.
- O `worker_status` muda para `online`.
- O `worker_heartbeat` atualiza sozinho.
- `cash` e `equity` deixam de ficar travados.
- Trades começam a aparecer quando houver sinal.
