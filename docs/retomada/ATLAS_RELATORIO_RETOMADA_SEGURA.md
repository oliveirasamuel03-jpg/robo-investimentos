# ATLAS — Retomada segura INFRA-1

> Atualização de 11/11/2026: [acesso, recuperação e financeiro](ATLAS_ACESSO_RECUPERACAO_FINANCEIRO.md).
> Acesso antigo negado; backup não comprovado; imagem histórica PostgreSQL 18;
> billing sem sessão autenticada. Homologação externa permanece bloqueada.

> Continuidade: [INFRA-1.1 — persistência e recuperação local](ATLAS_INFRA_1_1_PERSISTENCIA.md).
> A nova fase corrige gravações parciais e corrupção JSON silenciosa e acrescenta
> 19 verificações reais em PostgreSQL local, além de 349 testes offline aprovados.
> O relatório INFRA-1 abaixo fica preservado como registro histórico da primeira entrega.

Data: 10/11/2026. PAPER_ONLY / COST_CONTROLLED / NO_PRODUCTION_DEPLOY.
Escopo exclusivo: `oliveirasamuel03-jpg/robo-investimentos`.

## 1. Resumo executivo

Preparação de código e validação local concluídas; homologação externa BLOQUEADA.
Foram corrigidos bootstrap administrativo público, bypass administrativo anônimo,
habilitação de real mode pela UI e testes desatualizados. Nenhuma estratégia,
score, threshold, capital, ticket, limite de posições ou provider operacional mudou.
331 testes passaram com rede bloqueada. Isso NÃO autoriza dinheiro real nem deploy.

Esta é uma auditoria dirigida de código, configurações e testes, não uma certificação
integral de segurança, pentest ou comprovação de recuperação de dados antigos.

## 2. Inventário do aplicativo

| Componente | Implementação | Diagnóstico confirmado |
|---|---|---|
| Entrada web | `app.py`, Streamlit 1.44.1 | Funciona localmente; renderização AppTest e servidor HTTP |
| Login | `pages/0_Login.py`, `core/auth/*` | Funciona após correção; senhas bcrypt, bootstrap por ambiente |
| Trader | `pages/1_Trader.py` | Renderiza offline autenticado; integração LIVE não verificada |
| Controle do Bot | `pages/4_Controle_do_Bot.py` | Guarda administrativa; renderiza com administrador de fixture |
| Histórico | `pages/5_Historico.py`, `core/trader_reports.py` | Funciona com dados locais simulados; histórico antigo indisponível |
| Worker | `workers/trader_worker.py` | Importação e heartbeat simulado aprovados; loop contínuo não iniciado |
| Motor | `engines/trader_engine.py`, `paper_trading_engine.py` | Ciclos PAPER testados com fixtures; envio real não implementado no caminho auditado |
| Broker | `core/broker.py` | 12 combinações de provider/mode/flag: execução e envio sempre false |
| Estado | `core/state_store.py` | Testes locais aprovados; compartilhamento remoto não verificado |
| Persistência | `core/persistence.py` | JSON/CSV local testado indiretamente; PostgreSQL real não testado |
| Feed | `core/market_data.py`, `core/providers/*` | Twelve Data configurado por padrão, fallback Yahoo/sintético; testes isolados |
| Multi-timeframe | `core/multi_timeframe_data_fetcher.py` | Mocks/guards testados; nenhum diagnóstico recebeu autoridade nova |
| Alertas/monitoramento | `core/alerts.py`, `core/production_monitor.py` | Cooldown, falhas, heartbeat e fallback testados; entrega externa não testada |
| Relatórios/retenção | `core/email_reports.py`, `core/retention.py` | Testes aprovados; memória com grande histórico e SMTP/Resend não homologados |
| Diagnósticos | módulos de estrutura, BOS, pivô, Fibonacci, shadow e microajuste | Suíte aprovada; fases preservadas, sem avanço para 2.6C |

113 arquivos Python previamente rastreados. Inventário inclui inspeção das quatro
páginas, pontos de autenticação, broker, motor, persistência, configuração e worker.

## 3. Estado atual confirmado e rastreabilidade

- `main` consultada no GitHub: `f0c932c21212064dcef37e1c928b0b188adf440f`.
- Preparação INFRA-0: `78857cb7ce2660685ee99d5d7f1a1ddb9031e7e1`, filha da main.
- Branch de trabalho: `atlas/infra-1-estabilizacao-offline`, criada dessa preparação.
- Branch nova evita atualizar a branch já selecionada nas alterações staged do Railway.
- PRs #70–#77 consultados: todos mergeados; código já inclui 2.6B.1, 2.6B.2 e
  clareza de relatório 2.6B.3. O resumo antigo limitado ao PR #70 não descreve o HEAD.
- Nenhum PR criado: configuração de previews externos não confirmada. Nenhum merge.
- Evidências e código pertencem ao mesmo commit desta entrega, identificável pelo
  histórico Git deste arquivo; não se deve interpretar o SHA da base como SHA da correção.

## 4. Alterações realizadas

- `core/auth/session.py`: autenticação obrigatória inclusive localmente; `AUTH_REQUIRED=false`
  não cria mais administrador anônimo; sessões de bypass antigas são invalidadas.
- `pages/0_Login.py`: removido cadastro público do primeiro administrador. Sem bootstrap
  configurado, mostra indisponibilidade e para; bootstrap existente por ambiente permanece.
- `pages/1_Trader.py`: botão de habilitar real mode desativado e ação protegida no servidor;
  desabilitar flag antiga continua possível com administrador autenticado e senha.
- `.python-version`: Python 3.12, versão validada localmente; runtime antigo usava 3.13.
- `requirements-dev.lock`: snapshot das versões instaladas, inclusive transitivas.
- `scripts/test_offline.py`: ambiente limpo, storage temporário, recusa `.env`, bloqueio
  de rede Python/processos e autoload de plugins pytest desativado.
- `scripts/local_web.py`: demo autenticada em loopback, storage descartável, sem credenciais
  herdadas de DB/provider/email e com bloqueio de conexões externas.
- Testes: imports corrigidos, fixtures atualizadas ao contrato `load_market_data_result`,
  cenários de alertas/feed corrigidos e cobertura INFRA-1 adicionada.
- `DEPLOY_RAILWAY.md`: aviso de que instruções antigas não podem ser executadas nesta fase.

## 5. Evidências dos testes

Ambiente: Python 3.12.14; instalação `requirements-dev.txt` concluída sem serviços externos
de execução. Download de pacotes não é teste de provider. Comandos executados:

```bash
.venv/bin/python scripts/test_offline.py -q
.venv/bin/python -m compileall -q app.py core engines pages workers scripts tests
.venv/bin/python -m pip check
git diff --check
```

| Rodada | Resultado |
|---|---|
| Inicial | 3 erros de coleta: `from conftest` em pacote `tests` |
| Após imports | 296 aprovados / 5 reprovados; fixtures antigas e mocks ineficazes |
| Final | **331 aprovados, 0 reprovados, 42 warnings, 6,57 s** |
| Compileall | Aprovado |
| pip check | Nenhuma dependência quebrada |
| Streamlit real local | `/_stcore/health`: HTTP 200, `ok`; processo encerrado com código 0 |
| UI via AppTest | Entrada, Trader, Controle, Histórico e Login sem exceção |

Evidências: [coleta inicial](evidencias/baseline-collection.txt) e
[saída completa final](evidencias/pytest-final.txt).

O guard final bloqueou **38 tentativas** de rede/processo. Há caminhos secundários de
fetch/fallback que ainda tentam consultar rede; as exceções são tratadas pelo app.
Nenhuma dessas tentativas teve permissão de sair pelo transporte Python auditado.
O harness não é um sandbox universal contra bibliotecas nativas arbitrárias: destina-se
à suíte inspecionada, sem credenciais, subprocessos ou provedores pagos.
Passar offline não comprova funcionamento LIVE. Python 3.13, PostgreSQL remoto,
restauração real, entrega de emails, navegador completo e estabilidade de 72h não foram testados.

## 6. Problemas corrigidos

1. Visitante podia criar administrador quando o store estava vazio: bloqueado.
2. Configuração de desenvolvimento concedia admin anônimo: removida.
3. UI oferecia habilitação de real mode: bloqueada sem alterar motor/estratégia.
4. Três testes não eram coletados: imports alinhados ao pacote.
5. Testes mockavam `load_prices`/`load_market_data_map`, mas o ciclo usa
   `load_market_data_result`: fixtures corrigidas. Teste de relatório agora comprova
   abertura da posição PAPER antes de testar saída e relatório.
6. Cooldown não preparava pré-requisitos; fixtures de saúde deixavam feed UNKNOWN;
   teste de gráfico misturava sucesso do worker com sucesso do gráfico: corrigidos
   os cenários, preservando comportamento operacional.

O erro `url_pathname` observado durante AppTest vinha do cache global de descoberta
de páginas entre entrypoints diferentes. Fixture reinicia esse cache; nenhuma mudança
de navegação produtiva foi feita para mascarar a limitação do harness.

## 7. Problemas pendentes e débitos técnicos

- Persistência local faz read-modify-write sem lock e sem gravação atômica: não usar
  web e worker separados compartilhando arquivos como solução definitiva.
- `replace_json_rows` usa DELETE seguido de INSERT sob autocommit: falha intermediária
  pode deixar histórico parcial. Correção transacional exige fase de persistência e testes PostgreSQL.
- JSON inválido pode retornar default silenciosamente; risco de confundir corrupção com store vazio.
- Grandes módulos/UI e acoplamento de estado tornam regressões difíceis de isolar.
- Warnings `utcnow`, resample `H` e concat pandas; não bloqueiam estes testes, mas devem ser tratados.
- Snapshot lock sem hashes; auditoria CVE completa não realizada. Python 3.13 não homologado.
- Não há workflows `.github` rastreados no checkout; isso não comprova ausência de integrações externas.
- Fluxos de autenticação precisam de rate limiting e política de senha em futura revisão.
- Flag real antiga no estado não é saneada automaticamente; exigir inspeção antes de restaurar.
- Documentação INFRA-0 aponta outro projeto histórico. Ela é evidência antiga, não destino autorizado.

## 8. Auditoria de segurança

Guards: login para Trader/Histórico/entrada; administrador para Controle do Bot.
Senhas persistidas com bcrypt, testes sem credenciais reais. Bootstrap por ambiente
nunca autentica visitante automaticamente. Credenciais administrativas devem ser
inseridas fora do Git e removidas do ambiente após provisionamento seguro do usuário.

Broker mantém `can_submit_orders=false` e `execution_enabled=false`, inclusive sob
providers/flags live simulados. Não foi encontrado caminho de envio real nos módulos
de execução auditados. Isso não é autorização para configurar broker real.
Controles shadow/study/report preservados; nenhum arquivo de estratégia foi alterado.
Falha de bootstrap encontrada antes da publicação e corrigida apenas na branch.

## 9. Situação Railway

Destino exclusivo: `perceptive-integrity` / `cba6525d-a2b3-428e-ada2-5978590bda2b`.
Environment `production`: `8c980ba6-72ce-4de3-a6d6-4827b77ce333`.
Serviço `robo-investimentos`: `2d3c3373-911c-4844-a728-81d5055297ee`.
Workspace observado: Samuel Oliveira's Projects; ID fornecido pelo usuário
`4e937e0d-7701-4de4-8d18-b1c61bd5b47d` não revalidado por endpoint financeiro.

- Um serviço; último deploy `821211f6-531b-4f5a-a25c-2ae80dbb6d29`, CRASHED.
- Logs: execução fora de `streamlit run`, `NoSessionContext` em `st.switch_page`.
- Nove alterações seguem STAGED, patch `56534ac3-4dc4-4220-8fbf-f7f7b12bd393`.
- Source staged correto; branch staged ainda INFRA-0. **Não contém estas correções INFRA-1.**
- Start staged: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`.
- Restart staged: NEVER. Uma réplica planejada não significa serviço saudável ativo.
- AUTH_REQUIRED, BROKER_MODE e BROKER_PROVIDER aparecem staged, com valores redigidos.
  API de variáveis live retornou somente nomes Railway. Valores staged não certificados.
- Zero volumes, buckets, domínios e proxies TCP; sem serviço worker/Postgres no destino.
- Nenhuma mutação Railway nesta execução. Não aceitar as nove alterações como se já
  incluíssem a correção de autenticação. Nunca selecionar main para resolver isso.
- Ausência histórica de ordens reais não é verificável sem banco/auditoria de origem.

## 10. Situação financeira

Trial, US$ 5 de crédito e consumo previamente US$ 0: **informados pelo usuário**,
não reconfirmados em faturamento. Nenhum recurso pago, deploy, Agent ou serviço foi
ativado por esta tarefa; custo externo novo provocado por provisionamento: US$ 0.
Não é possível certificar o saldo ou a fatura atual a partir dos reads disponíveis.

Conector não expõe controle financeiro. CLI Railway ausente; navegador do projeto
mostrou Login/404, sem sessão para acessar billing. Não foi iniciado login nem upgrade.
Documentação oficial pesquisada distingue limites Compute e Agent:
https://docs.railway.com/pricing/cost-control
Compute hard limit derruba workloads; Agent hard limit desabilita Agent separadamente.
A busca retornou esses trechos, mas fetch da página completa falhou; mínimo permitido,
elegibilidade Trial e limites efetivamente configurados continuam NÃO VERIFICADOS.

Proposta não aplicada: Compute hard limit <= US$ 10 na futura fase paga; Agent no
menor valor permitido, sem ativação. Não configurar se exigir contratação. Compute
não deve ser tratado como teto da fatura total (plano, impostos, Agent e outros itens).

## 11. PostgreSQL e recuperação

Sem acesso à conta antiga ou banco nesta tarefa; sem backup recuperável comprovado.
Volume antigo de 5 GB é informação histórica, não prova de backup.

Schema esperado pelo código (não inspecionado no banco): `app_state(namespace TEXT PK,
payload_json TEXT, updated_at TIMESTAMPTZ)` e `app_rows(seq BIGSERIAL PK, namespace TEXT,
row_json TEXT, created_at TIMESTAMPTZ)`, índice namespace/seq.
Namespaces incluem auth_users_store, paper_state, paper_trades, bot state e tabelas
CSV convertidas para `table:<nome>`. `STATE_SCHEMA_VERSION` é marcador de estado,
não um sistema versionado de migrações SQL.

**Não abrir o app contra a origem para inspecionar:** `_ensure_db_schema` executa DDL
e loaders podem importar arquivos locais quando não encontram linhas remotas.

Procedimento preparado, NÃO executado:
1. Confirmar acesso legítimo, versão do servidor, propriedade e custo, sem iniciar serviços antigos.
2. Inventariar schema/contagens com transação read-only via cliente SQL, não via loaders do app.
3. Obter dump consistente por ferramenta compatível; manter senha em mecanismo seguro,
   nunca comando/log/Git. Preservar checksum e cópia original protegida.
4. Restaurar exclusivamente em banco isolado e autorizado, validar tabelas, sequência,
   namespaces, JSON, usuários e hashes, posições, trades, timestamps e contagens.
5. Exigir uma única autoridade de escrita; nenhum worker ligado durante restauração.
6. Antes de aplicação, revisar a criação automática de schema e importação local.

Sem recuperação: decisão explícita de iniciar histórico novo, bootstrap seguro,
estado STOPPED e PAPER, sem copiar exemplos/archives para runtime. Banco novo
**não representa migração do histórico antigo**. Nenhum banco foi criado.

## 12. Execução local gratuita e reproduzível

Em checkout desta branch, Python 3.12, sem `.env` de produção:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python scripts/test_offline.py -q
.venv/bin/python scripts/local_web.py
```

O launcher solicita senha temporária local de pelo menos 12 caracteres, usuário
`local-demo`; autenticação permanece obrigatória. Conectar apenas em
`http://127.0.0.1:8501`; Ctrl+C encerra e descarta dados temporários.
Não contém histórico antigo. Feed externo bloqueado; eventual fallback não comprova LIVE.
`--no-bootstrap` serve apenas para smoke de servidor/login indisponível.
Não usar `python app.py`. Não executar worker contínuo nesta fase.

## 13. Preparação para homologação Railway

Revisar commit INFRA-1; verificar previews/webhooks/autodeploy antes de PR ou
mudança de source. Depois, somente preparar source staged para a branch revisada,
conferir as nove mudanças e valores PAPER/auth, startup e Python. Sem Accept Deploy.
Resolver persistência/admin/financeiro antes de solicitar janela curta de web.

## 14. Critérios para habilitar web PAPER

Autorização específica de deploy/custo/duração; revisão de segurança; source com
correções INFRA-1; autenticação e bootstrap protegido; persistência escolhida e
testada; BROKER_MODE/BROKER_PROVIDER paper conferidos; sem credenciais de broker real;
política de reinício e plano de encerramento; previews automáticos controlados.
Health HTTP sozinho não homologa login, estado ou segurança.

## 15. Critérios para habilitar worker PAPER

Web homologada; banco compartilhado recuperado ou histórico novo explicitamente
aprovado; somente um worker; estado e posições revisados; feed Twelve Data autorizado,
budget e credenciais seguros; heartbeat/logs/safety coerentes durante ciclo observado;
nenhum shadow com autoridade; teste prolongado aprovado em orçamento separado.
Não ligar worker com filesystem efêmero como solução para falta de PostgreSQL.

## 16. Próxima fase recomendada

INFRA-1.1: revisão das correções de segurança e persistência, teste PostgreSQL isolado
gratuito quando houver runtime adequado, preparação da recuperação e confirmação
financeira. Manter PAPER, sem deploy e sem avanço estratégico para 2.6C.

## 17. Passos manuais indispensáveis

- Acesso autenticado ao billing para confirmar Trial/crédito/limites sem contratar.
- Acesso legítimo a backup/export da conta antiga, se disponível sem custo/reativação.
- Decisão sobre histórico novo se não houver recuperação.
- Inserção de credenciais administrativas somente em canal protegido na homologação.
- Autorização específica para qualquer deploy futuro. Não há pedido de autorização
  para deploy agora: ainda existem bloqueios técnicos e financeiros.

## 18. Bloqueio e rollback

Bloquear avanço com auth anônima, flag real ativa, autoridade shadow, segredo exposto,
falha crítica de testes, saldo/limites desconhecidos, restauração não validada ou
deploy automático não controlado. Na falha de homologação futura, interromper a
janela autorizada e preservar logs/dados; nunca apagar ou sobrescrever histórico.

Rollback de código: revert auditável somente em branch, sem main/merge/deploy nesta
fase. Não reintroduzir o bootstrap público como solução para login. Não há rollback
remoto necessário agora: nenhuma infraestrutura foi modificada.
