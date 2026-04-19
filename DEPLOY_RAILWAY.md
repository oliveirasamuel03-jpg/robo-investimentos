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

## Variáveis e storage

Os dois services precisam usar o mesmo código e o mesmo estado persistido do projeto. Se você adicionar volume/storage no Railway depois, mantenha a pasta de dados persistente entre reinícios.

## Fluxo recomendado

1. Fazer merge do PR com as correções do paper trading.
2. Deployar o `web` com o comando do Streamlit.
3. Deployar o `worker` com `python -m workers.trader_worker`.
4. Abrir a interface e mudar o bot para `RUNNING`.
5. Conferir se o painel mostra o worker como `online`.

## Checklist de validação

- O site abre no domínio do Railway.
- O botão `Rodar agora` funciona.
- O `worker_status` muda para `online`.
- O `worker_heartbeat` atualiza sozinho.
- `cash` e `equity` deixam de ficar travados.
- Trades começam a aparecer quando houver sinal.
