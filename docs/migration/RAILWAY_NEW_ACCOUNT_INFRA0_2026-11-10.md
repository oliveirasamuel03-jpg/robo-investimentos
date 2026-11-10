# FASE INFRA-0 — Preparação controlada da conta Railway (2026-11-10)

> **Escopo:** preparação de infraestrutura, não implantação. `PAPER TRADING` obrigatório. `main` intocada. Nenhuma autorização para negociação real, mudanças de estratégia, limiares, capital, ticket, score ou provider de mercado.
>
> Repositório exclusivo: `oliveirasamuel03-jpg/robo-investimentos`. Não acessar nem alterar `painel-ir-softr-tools` ou outros projetos.

## Identificadores verificados

- Repositório: https://github.com/oliveirasamuel03-jpg/robo-investimentos
- Commit de origem da branch de migração: `f0c932c21212064dcef37e1c928b0b188adf440f` (FASE 2.6B.3, REPORT_ONLY).
- Branch de preparação: `atlas/infra-0-railway-account-migration`.
- Conta/workspace Railway novo: `Samuel Oliveira's Projects`; ID `4e937e0d-7701-4de4-8d18-b1c61bd5b47d`.
- Projeto Railway novo já existente: `precious-integrity`; ID `2211b14a-9a43-48e5-93f5-5014995b557d`.
- Environment: `production`; ID `23687181-a9ff-4365-9430-f6b6f6914b43`.
- Serviço `web` **somente staged**: `c4b9f978-eeec-44cb-a599-8ea1155862d1`.
- Serviço `work` **somente staged**: `f652c296-e4a1-4f9e-9db8-c359ea3d729f`.
- Serviço PostgreSQL **não criado** na conta nova. Nenhum volume provisionado. Sem deploy.

## Configurações staged para futura homologação

| Item | web | work |
|---|---|---|
| Start command | `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT` | `python -m workers.trader_worker` |
| Restart policy | `NEVER` | `NEVER` |
| `AUTH_REQUIRED` | `true` | `true` |
| `BROKER_MODE` | `paper` | `paper` |
| `BROKER_PROVIDER` | `paper` | `paper` |
| GitHub source | **pendente** | **pendente** |
| `DATABASE_URL` | **pendente** | **pendente** |
| Credenciais Twelve Data | **pendente, inserir pelo usuário diretamente no destino** | **pendente** |

Todas essas configurações estão pendentes de commit no Railway e **não foram aplicadas**. Não executar `accept-deploy` antes de aprovação financeira e homologação de segurança.

## Bloqueio confirmado de integração GitHub no Railway

A tentativa de configurar o repositório como fonte staged foi rejeitada com: `No GitHub account linked. Please connect a GitHub account to perform this action.`

**Ação manual:** Railway nova conta → Settings / Integrations / GitHub, autorizar o usuário `oliveirasamuel03-jpg` e o repositório `robo-investimentos`. Esta autorização Railway↔GitHub é distinta da conexão GitHub↔ChatGPT. Não selecionar deploy automático no processo.

## Gasto e orçamento

O usuário relatou aproximadamente R$ 300 em fatura da conta Railway antiga. O destino deve ter meta de custo zero em desenvolvimento e **planejamento máximo de US$ 10 por ciclo**. A fatura total pode incluir plano, Compute, Railway Agent, impostos e encargos; **Compute Hard Limit não controla automaticamente todas essas parcelas**.

**Ação manual antes de qualquer execução:** novo Railway workspace → Usage → Set Usage Limits: configurar hard limit de Compute em US$ 10, alerta antecipado, e o menor limite de Railway Agent permitido. Exigir confirmação da configuração por inspeção do painel; os conectores atuais não expõem alteração de teto de cobrança. Não contratar plano sem consentimento explícito sobre preço.

**Proibidos enquanto bloqueado:** executar Railway Agent, criar deploy, ligar web/work, provisionar Postgres/volume, gerar serviços pagos, aceitar alterações staged, alterar credenciais de provider, executar ordens, criar PR com preview automático, tocar na conta antiga.

## Banco anterior: pendência crítica

O antigo projeto Railway `jubilant-beauty` tinha web, work e Postgres com volume `postgres-volume` de 5.000 MB provisionados; os serviços apareceram offline na inspeção prévia. O **volume listado NÃO significa backup validado**. Precisamos avaliar acesso legítimo aos dados, dump PostgreSQL consistente, verificação de restauração e custos antes de qualquer destruição/transferência. Nunca iniciar dois workers gravando no mesmo estado.

## Critérios de desbloqueio do primeiro deploy

1. Integração Railway↔GitHub confirmada, source na branch separada, sem auto-preview.
2. Teto financeiro e encargos verificados por humano; orçamento explicitamente aprovado.
3. Banco de origem recuperado ou decisão documental explícita de iniciar novo banco vazio, sem fingir migração de histórico.
4. `DATABASE_URL` interno e credenciais necessárias configurados no destino, sem exposição em GitHub.
5. Autenticação **obrigatória** (`AUTH_REQUIRED=true`); nunca desativar login tornando usuário anônimo administrador.
6. `PAPER` e bloqueios de negociação real verificados por testes; nenhuma alteração operacional ou de score/threshold.
7. Web primeiro e worker depois; sem execução real; testes de heartbeat/feed/Twelve Data e logs.
8. Revisão de segurança antes de qualquer PR/merge; manter a `main` sem alterações diretas.
9. Se GitHub connector bloquear um merge, não contornar com CLI ou push manual.

## Status de preparação

- **Concluído:** auditoria básica do destino, branch isolada, criação staged de `web` e `work`, comandos de inicialização e variáveis de segurança staged.
- **Pendente manual:** Railway↔GitHub, limite de gastos.
- **Pendente técnico:** recuperação/backup de dados, Postgres, variáveis de banco e mercado, validação de testes.
- **Proibido nesta fase:** ativar recursos, aceitar deploy ou avançar estratégia para FASE 2.6C.

Este arquivo não contém credenciais nem dados privados e **não substitui** backup.
