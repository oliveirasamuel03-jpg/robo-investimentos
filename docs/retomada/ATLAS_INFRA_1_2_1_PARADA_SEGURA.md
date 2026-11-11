# ATLAS — INFRA-1.2.1: parada segura, contenção e integridade operacional

Referência: 11/11/2026, UTC.
**PAPER_ONLY / SECURITY_FIRST / COST_CONTROLLED / NO_PRODUCTION_DEPLOY.**
Repositório exclusivo: `oliveirasamuel03-jpg/robo-investimentos`.

## 1. Resumo executivo

Correções implementadas e homologadas localmente: STOP pode persistir enquanto
um ciclo aguarda feed; o draft é revalidado antes da confirmação financeira;
parada recebida, persistida e reconhecida têm estados e mensagens diferentes.
Saldo, posições, trades, relatório e espelho da plataforma continuam na mesma
transação PostgreSQL. Não houve alteração dos cálculos de entrada/saída.

Entrega para revisão Atlas, **sem aprovação de Web/Worker 24h**, execução real ou
segurança absoluta. Ainda existem contenção durante commit, limites do backend
de arquivos e ausência de chave única de idempotência externa.

- HEAD inicial conferido: `5d7165217e42ff5121b5444f956d45a877118734`.
- Base: `atlas/infra-1-2-integridade-concorrencia`.
- Branch nova: `atlas/infra-1-2-1-parada-segura`.
- Commit de implementação/testes:
  [`02c70031171b1f944dccdf2b45052bdf7e3c231f`](https://github.com/oliveirasamuel03-jpg/robo-investimentos/commit/02c70031171b1f944dccdf2b45052bdf7e3c231f).
- Evidências finais: **480 testes offline aprovados; 46 aprovados em PostgreSQL
  18.6 e os mesmos 46 em PostgreSQL 16.15**. Nenhuma falha final.

## 2. Problema original reproduzido antes da correção

O script [reproduce_infra121_legacy_stop.py](../../scripts/reproduce_infra121_legacy_stop.py)
carrega uma cópia temporária obtida por `git archive 5d71652`, usa PostgreSQL 18.6
local descartável e prende o worker em um fake feed por `threading.Event`.
Enquanto o Event de liberação permanece fechado, o STOP real tenta adquirir o
lock global e recebe `psycopg.errors.LockNotAvailable` por timeout do PostgreSQL.
O estado continua RUNNING: **a intenção não foi persistida**. Ao liberar o feed,
uma exceção sintética encerra o ciclo com rollback; zero trades foram confirmados.

[JSON da reprodução](evidencias/infra121-legacy-stop.json) e
[log](evidencias/infra121-legacy-stop.txt). Não se usou sleep arbitrário para
inferir a ordem: o worker confirmou sua entrada no feed, e o resultado do STOP
foi recebido antes da liberação desse feed.

## 3. Mapa de concorrência

O [mapa anterior](evidencias/infra121-concurrency-before.md) foi registrado antes
da alteração funcional dos ciclos. A INFRA-1.2 adquiria
`atlas:persistence-unit` no início de `run_trader_cycle`; `run_paper_cycle`
aninhava savepoint na mesma conexão. Feed principal, fetch intraday shadow,
cálculo, trades, relatórios, sincronização e leitura do relatório permaneciam
sob esse lock. O tempo incluía toda a duração da consulta externa, sem limite
superior imposto pelo lock; outro escritor esperava até `lock_timeout=5s`.

STOP, heartbeat, configurações, usuários, logs e retenção dependiam do mesmo
lock. O worker verificava status antes do ciclo; não havia request_id/ACK. Reset
podia substituir STOPPED por DEFAULT. As telas não distinguiam commit de STOP
de reconhecimento posterior do worker.

| Etapa atual | Trabalho | Garantia e concorrência |
|---|---|---|
| Preparação curta | Registra `cycle_activity.PREPARING`, lê controles/configuração/risco e snapshot PAPER | Transações pequenas concluídas antes do feed. Heartbeat preserva STOP. |
| Aquisição e cálculo | Mesmo feed/provider, indicadores, estratégia e estudos; cria draft em memória | Nenhuma transação de escrita permanece aberta durante consultas externas. O draft não é outra fonte persistente de verdade. |
| Validação | Adquire unidade transacional e relê estado atual | STOP primeiro; broker exclusivamente PAPER; revisão da intenção, configurações, risco/posições/saldo atuais e CAS do estado PAPER. |
| Commit | Estado PAPER, trades, relatório, ordens e posições espelhadas | Uma transação PostgreSQL; falha reverte tudo. Savepoints preservados. |
| Observabilidade | Atualização de feed/contexto e retorno da UI | Não confirma operações novas. Mensagem considera se o commit ocorreu antes de STOP posterior. |

A duração do lock deixou de incluir feed/cálculo da estratégia. O commit ainda
faz serialização, leitura de histórico e sincronização de ordens; sua duração
não tem garantia absoluta de cinco segundos. Os testes comprovam ordenação e
recusa, não são benchmark de produção. A configuração global de lock não foi
relaxada para esconder contenção.

## 4. Correções realizadas

- Removidos os decorators que envolviam todo `run_trader_cycle` e
  `run_paper_cycle`. Consultas externas e cálculo produzem apenas um draft;
  trades e relatórios financeiros só são publicados no final.
- `_commit_paper_draft` relê STOP e broker sob a mesma barreira do commit. STOP
  persistido descarta o draft inteiro e permite ACK auditável, sem operação nova.
- Revisão de controle também é conferida em chamada direta a `run_paper_cycle`:
  um draft anterior a STOP → ACK → START não é reaproveitado silenciosamente.
- No fluxo trader, fingerprint inclui intenção, configurações operacionais,
  risco, macro alert, posições, saldo e capital existentes. Mudança durante feed
  gera `PersistenceConflictError`, sem retry ou alteração de thresholds.
- CAS do estado PAPER ocorre antes de append. PostgreSQL confirma também o
  espelho da plataforma na mesma transação; falhas posteriores revertem o save.
- `run_paper_cycle` recusa aquisição chamada dentro de transação de persistência
  já aberta, evitando reintroduzir consultas externas sob o lock.
- Guard de STOP protege `state_store` e a API genérica de persistência do
  namespace `bot_state`; saves/reset comuns não removem uma parada confirmada.
- Conexões sem `connect_timeout` explícito recebem default de 5s; DSN que já
  define timeout mantém seu valor. Isso não converte falha em sucesso.
- Retorno de ciclo bem-sucedido inclui `committed=True`. Se STOP ocorre depois
  desse commit, a UI informa o commit anterior sem prometer cancelamento retroativo.

## 5. Protocolo de parada segura

Única fonte persistente: `bot_state.stop_control`, com request_id, requested_at,
persisted_at, acknowledged_at, actor/worker_id e eventos. O protocolo não cria
outra tabela, fila, worker ou schema. Timestamps são registros da aplicação
criados na transação; a evidência de persistência é o commit e sua verificação,
não o relógio Python isoladamente.

| Estado | Evidência e mensagem |
|---|---|
| RECEBIDA | A interface recebeu o clique e informa que ainda aguarda persistência. Não significa parada. |
| PERSISTED | `request_stop` concluiu transação com STOPPED e request_id, depois releu esse estado. Novos commits PAPER já devem ser recusados. O worker pode ainda estar adquirindo/calculando dados. |
| ACKNOWLEDGED | Worker/ciclo observou STOP sob a barreira financeira e persistiu ACK da mesma solicitação. A UI só chama parada confirmada se também verificar STOPPED. |
| UNCONFIRMED | Timeout, conflito, falha de escrita/conexão ou falha de leitura de confirmação. A UI não declara sucesso, não repete a ação e orienta recarga/verificação. |
| RESUMED | START explícito, com revisão da intenção realmente exibida ao operador. Nova solicitação pendente de ACK não pode ser retomada. |

`request_stop` retorna recibo apenas fora de uma transação PostgreSQL externa;
chamada aninhada não produz falsa confirmação. Repetir STOP/ACK da mesma parada
preserva ID e não duplica eventos. Uma falha de confirmação pode ocorrer **depois
do commit**: UNCONFIRMED não prova rollback nem autoriza reenvio cego.

Durante feed lento o STOP pode persistir imediatamente em relação ao término do
feed; não precisa aguardar essa consulta. Se o commit já possui o lock, STOP
espera ou retorna UNCONFIRMED por timeout. Um commit que terminou antes da parada
persistir continua válido. Não há cancelamento retroativo.

`cycle_activity` registra ID, PREPARING/IDLE/FAILED e timestamps. PREPARING é o
último registro conhecido, não prova de processo vivo. ACK prova a barreira
contra commits futuros, não encerramento do socket/feed ou do processo Python.
Um draft que voltar do feed depois da parada será descartado. Reinício importa
estado persistido e reconhece STOP antes de executar ciclo; não reativa RUNNING.

START exige Snapshot persistido e revisão original de bot_status/request/ACK/
resume. Heartbeat não invalida essa revisão; STOP ou ACK posterior à tela exige
recarga. Se modo foi fornecido, alteração concorrente de modo também é recusada.
Reset preserva STOP e sua auditoria, inclusive para STOPPED legado sem protocolo.

## 6. Arquivos modificados

Produção:
`core/persistence.py`, `core/state_store.py`, `core/stop_control.py`,
`core/ui_mutations.py`, `engines/trader_engine.py`, `paper_trading_engine.py`,
`workers/trader_worker.py`, `pages/1_Trader.py`, `pages/4_Controle_do_Bot.py`.

Homologação:
`scripts/test_infra12_postgres.py`, `scripts/reproduce_infra121_legacy_stop.py`,
`scripts/check_infra121_strategy_equivalence.py`, `tests/conftest.py`,
`tests/test_stop_control.py`, `tests/test_ui_mutations.py`,
`tests/test_infra121_engine.py`, `tests/test_infra121_idempotency.py`,
`tests/test_infra121_postgres.py`.
Documentação: este relatório, proposta separada de idempotência e
`docs/retomada/evidencias/infra121-*`.
Nenhum teste antigo teve seu assert alterado para mascarar falha; os 451 casos
anteriores permanecem coletados. Nenhuma dependência nova foi exigida.

## 7–8. Testes executados e evidências

| Execução final | Coletados | Aprovados | Falhas/erros | Skips | Warnings | Tempo |
|---|---:|---:|---:|---:|---:|---:|
| Offline completo | 526 | 480 | 0 | 46 | 89 | 26,36s |
| PostgreSQL 18.6 | 46 | 46 | 0 | 0 | 19 | 48,02s |
| PostgreSQL 16.15 Alpine | 46 | 46 | 0 | 0 | 19 | 45,33s |

São **526 casos distintos**: 451 anteriores + 60 novos offline + 15 novos
PostgreSQL. Os 46 skips offline são os testes de integração que foram executados
nas duas versões; não foram contados como aprovados no offline. Nenhum desses
casos ficou sem executar. Reexecuções e equivalência não foram somadas a esse total.

Novos offline: 20 do protocolo, 34 de UI (incluindo botões reais via Streamlit
AppTest), 4 de idempotência e 2 do guard de engine. O runner offline bloqueou
124 tentativas de rede/processo. Não houve envio a provider/broker real.
Warnings de datetime.utcnow e pandas (`H`, concat vazio) foram mantidos.

Logs/JUnit: [offline](evidencias/infra121-offline.txt),
[PG18](evidencias/infra121-postgres18.txt), [PG16](evidencias/infra121-postgres16.txt)
e os XML de mesmo prefixo. [Checks](evidencias/infra121-checks.txt): compileall,
pip check e git diff --check com exit code zero. Os logs TXT tiveram somente
espaços finais/linhas vazias finais normalizados para higiene Git.

Registro de iterações: primeira integração PG18 com 45 casos passou sem falhas;
a última acrescentou rollback de SELL/relatório e observação do caminho shadow.
Durante desenvolvimento, uma primeira execução isolada de idempotência teve
3 passed/1 failed porque o helper de commit ainda não havia sido implementado;
a suíte desse módulo terminou com 4 passed. Subconjuntos posteriores passaram
(78 casos do motor/integridade; 49 de protocolo/estado; 34 finais de UI). Uma
execução offline intermediária passou com 440 casos e 45 skips antes da conclusão
dos testes de UI/guards. Esses resultados não substituem as execuções finais.
A referência anterior de 420 offline/31 PG por versão permanece nos artefatos
INFRA-1.2; não é apresentada como nova execução completa da baseline nesta fase.

### Matriz determinística A–L

| Requisito | Exercício real e resultado |
|---|---|
| A/B | Event prende feed; STOP persiste com worker ainda bloqueado; conexão de escrita ausente no feed; release termina com draft descartado. |
| C | Escritor prende transação; STOP chega ao lock, aguarda commit anterior e depois persiste. Nenhuma compensação retroativa. |
| D | STOP persistido antes de liberar o feed; nenhum append financeiro; ACK da mesma solicitação ao validar draft. |
| E | STOP e heartbeat concorrem por Barrier; ambos campos preservados. |
| F | Formulário desatualizado e STOP concorrente; conflito sem overwrite/sucesso/retry; configuração atual preservada. |
| G | `OperationalError` de conexão injetado durante STOP; recibo UNCONFIRMED, uma tentativa, nenhum falso ACK. A suíte antiga também mantém interrupção real de backend e reconexão. |
| H | Timeout PostgreSQL real ao disputar lock; STOP não persistido nem declarado concluído. |
| I | Processo Python novo importa o engine usado pelo worker e faz ciclo único com STOP persistido; sem feed/trade e sem reativação automática. |
| J | Falhas após append real de BUY e após SELL/relatório, antes de COMMIT; rollback de todas as partes. Retry explícito não duplica. |
| K | Dois processos sincronizados por barreira PostgreSQL atualizam STOP e heartbeat; intenção e atualização preservadas. |
| L | Ciclos repetidos depois de STOP não chamam feed nem produzem novas entradas. |

Adicionais: trader desabilitado/risco bloqueado durante feed rejeitam draft;
STOP → ACK → START durante aquisição invalida draft antigo; tentativa de broker
não PAPER recusa publicação; API genérica não contorna guard; cálculos e replay
BUY/SELL permanecem consistentes. Os testes usam Event/Barrier ou barreira de
banco para estabelecer a ordem, sem sleep arbitrário como evidência principal.

## 9. PostgreSQL 18 e 16

Servidores reais descartáveis **18.6** e **16.15**, imagens oficiais Docker cujos
digests constam nos logs. Listener publicado só em 127.0.0.1, porta efêmera,
credencial aleatória sintética, tmpfs e banco novo por teste. Runner não aceita
DSN externo e elimina DATABASE_URL/credenciais herdadas antes de iniciar.
Imagens criadas para o ensaio foram usadas sequencialmente para evitar falta de
disco; nenhum volume antigo foi removido. Contêineres de teste/reprodução foram
removidos. Nenhum worker contínuo ou serviço remoto foi iniciado.

Repetição local com imagens previamente disponíveis:

```bash
.venv/bin/python scripts/test_offline.py -q
.venv/bin/python scripts/test_infra12_postgres.py --version 18
.venv/bin/python scripts/test_infra12_postgres.py --version 16 --alpine
.venv/bin/python scripts/reproduce_infra121_legacy_stop.py
.venv/bin/python scripts/check_infra121_strategy_equivalence.py
.venv/bin/python -m compileall -q core engines workers pages paper_trading_engine.py scripts tests
.venv/bin/python -m pip check
git diff --check
```

Os runners recusam `.env` local; os dois scripts de auditoria também usam dados
sintéticos e armazenamento temporário. Não substituir as conexões sintéticas
por DATABASE_URL remota. Nenhuma migração remota ou schema novo foi aplicado.

## 10. Segurança PAPER e integridade

Runners impõem `BROKER_MODE=paper` e `BROKER_PROVIDER=paper`. O guard do commit
revalida provider, modo configurado, live_requested e real_mode_enabled usando
estado mais recente. Uma configuração incompatível causa recusa; nenhuma ordem
real é enviada e nenhum broker real é habilitado.

Não se modificaram capital, ticket, score, thresholds, min_signal_score,
max_open_positions, provider, critérios de entrada/saída ou autoridade de estudos.
Os módulos SHADOW_ONLY/STUDY_ONLY/DIAGNOSTIC_ONLY/PLANNING_ONLY não ganharam
poder operacional. Não se avançou FASE 2.6C nem se ligaram flags experimentais.

[Equivalência sintética](evidencias/infra121-strategy-equivalence.json): cinco
comparações exatas entre baseline `5d71652` e código atual passaram, com relógio/
IDs fixados apenas no ensaio: estratégia real com feed fictício, BUY, posição
repetida, SELL/contabilidade e fechamento repetido. Warnings das duas execuções
foram preservados. Isso demonstra esses cenários, não equivalência universal.

## 11. Conflitos nas telas

As duas páginas têm Parar dedicado, feedback de solicitação recebida, resultado
de persistência e reconhecimento posterior. `run_ui_mutation` executa uma vez;
em conflito/timeout/falha mostra resultado não confirmado, orienta recarga e
revisão, sem imprimir erro com credenciais, sobrescrever ou repetir a operação.
A sessão guarda só rótulo de intenção pendente, que nunca reenvia o comando.

Todos os badges, títulos, métricas e logs de status das duas telas usam
“Parada pendente de confirmação” para STOPPED sem ACK verificável, inclusive
legado; o rádio de ação mostra “Solicitar parada”. Quatro AppTests exercitam
os rótulos PERSISTED/ACK das duas páginas.

START usa a revisão realmente exibida antes do rerun de clique. Configuração,
watchlist, pausa, reset, validação, administração de usuários e desabilitar modo
real têm tratamento conservador de falha. Isso não altera os valores desses
controles. A autorização de administrador existente foi preservada.

O feedback dá prioridade à evidência de commit: `committed=True` com estado final
STOPPED informa que o ciclo confirmou antes da parada. Resultado sem evidência
suficiente não recebe sucesso presumido. Dois AppTests simulam o resultado dessa ordem de commits
e verificam o feedback nos botões reais; outros verificam STOP com falha, formulário em conflito
e ausência de reenvio ao atualizar a página.

## 12. Idempotência e duplicação

UUID de posição não é chave de intenção idempotente. Os testes cobrem repetição
normal de BUY/SELL, rollback de cálculo/commit e recálculo explícito com posição
atual, sem duplicar trade ou relatório nos cenários ensaiados. CAS recusa draft
concorrente antes dos appends. Não foi adicionado índice único nem deduplicação
retroativa do histórico.

A [proposta separada](ATLAS_INFRA_1_2_1_IDEMPOTENCIA_PROPOSTA.md) apresenta
cycle_request_id/event_key, compatibilidade com registros antigos, índice parcial
futuro, atualização, testes e rollback. **Nenhum DDL dessa proposta foi aplicado.**
Uma conexão perdida no instante do COMMIT pode deixar resultado desconhecido;
a interface exige reconsulta e não faz retry automático. Não há promessa de
exactly-once para reenvio arbitrário, SQL externo ou versões antigas escritoras.

## 13–14. Riscos residuais e limitações

| Severidade | Limite e tratamento |
|---|---|
| Alta | Arquivos locais continuam sem rollback entre múltiplos arquivos. O lock de controle impede nova publicação depois de STOP, mas falha entre save e append ainda pode divergir saldo/histórico. Não liberar operação prolongada ou Web/Worker separados nesse backend. |
| Alta | SQL manual/versão antiga não adota guard/locks. Exigir versão coordenada dos escritores; o protocolo é de aplicação, não autorização contra Python/SQL privilegiado ou prova criptográfica de identidade do worker. |
| Média | Transação financeira longa, retenção e outros writers ainda podem causar timeout do STOP. UNCONFIRMED é visível; não se promete latência máxima de parada nem cancelamento de feed. |
| Média | Fingerprint conservador inclui metadados de risco/macro; atualizações equivalentes podem recusar draft. Medir esse efeito antes de reduzir o conjunto, preservando campos operacionais e a revisão da intenção. |
| Média | Não há chave única de idempotência externa; resultado desconhecido de COMMIT exige leitura e investigação antes de qualquer ação manual. |
| Média | ACK indica barreira financeira, não processo morto ou fim do feed. PREPARING pode permanecer após falha de banco/crash. Não tratar heartbeat/fase como confirmação de STOP. |
| Média | Leituras independentes de vários namespaces não são um snapshot agregado único. Diagnóstico financeiro deve usar contexto coerente e histórico completo quando necessário. |
| Baixa | CAS de conteúdo ainda admite ABA fora da revisão de intenção; tokens após rollback exigem recarga. Guards não substituem autorização da UI, backup ou auditoria externa. |

Nenhum histórico real foi excluído. O banco antigo não foi acessado nem
recuperado; sua condição continua desconhecida nesta missão. Os backups e
restores da suíte permanecem exclusivamente sintéticos.

## 15. Critérios de rollback

Considerar rollback se STOP deixar de persistir durante feed sem lock de escrita,
se draft anterior à parada confirmar, se houver divergência no ledger/relatório,
feedback indevido de sucesso ou recusas recorrentes que inviabilizem controle.
Antes de qualquer rollback operacional futuro: coordenar suspensão dos escritores,
preservar backup verificado, verificar requests pendentes e não misturar versões.
Reverter código reintroduz a contenção INFRA-1.2; não autoriza reativar o worker.
Não apagar eventos STOP, trades ou controle como compensação; não remover locks
para ocultar timeout. Não há migração de schema que exija DROP/TRUNCATE.
Nenhum rollback remoto foi executado.

## 16. Próxima recomendação e pendências Web/Worker 24h

Revisão Atlas deste protocolo e das evidências; homologar duração de commits,
recusa por fingerprint, perda de conexão no COMMIT e verificação manual de
solicitação pendente. Projetar idempotência persistente em fase separada, antes
de retries de intenção. Verificar política de previews/autodeploy em ação própria
antes de qualquer PR/publicação operacional. Web/Worker 24h, múltiplos workers,
infra remota, dinheiro real e FASE 2.6C permanecem sem liberação.

## 17. Commit final e rastreabilidade

Implementação/testes: `02c70031171b1f944dccdf2b45052bdf7e3c231f`, filho direto
da referência `5d71652`. A correção final de todos os rótulos da UI é
[`4bbf618dde50e6006f75d7d55737cf42feada7d3`](https://github.com/oliveirasamuel03-jpg/robo-investimentos/commit/4bbf618dde50e6006f75d7d55737cf42feada7d3).
O commit final de documentação é o commit que introduz
este arquivo e as evidências; obter seu SHA exato com:

```bash
git log -1 --format='%H %s' -- docs/retomada/ATLAS_INFRA_1_2_1_PARADA_SEGURA.md
```

O SHA final também é fornecido na entrega ao Atlas. Nenhuma alteração da main,
merge ou PR foi realizada. Commits da fase pertencem somente à branch nova.

## 18. Ausência de deploy e custos externos

Nenhum comando/ferramenta de deploy, restart Railway, alteração de variável
remota, provisionamento, compra ou upgrade foi executado. Nenhum PR foi criado,
nenhuma alteração foi enviada à main e nenhum workflow de deploy foi adicionado.
As ações GitHub se limitaram à branch solicitada e seus commits com `[skip ci]`.
O projeto Railway não foi modificado nem o PostgreSQL antigo acessado. Todos os
servidores de homologação foram locais descartáveis. **Nenhum novo custo externo
foi contratado e nenhum deploy foi executado nesta fase.**
