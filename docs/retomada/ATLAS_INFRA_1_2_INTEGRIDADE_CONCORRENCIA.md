# ATLAS — INFRA-1.2: integridade, concorrência e homologação local

Referência: 11/11/2026, UTC. **PAPER_ONLY / COST_CONTROLLED / NO_PRODUCTION_DEPLOY.**
Repositório exclusivo: `oliveirasamuel03-jpg/robo-investimentos`.

## Resultado e rastreabilidade

Implementação e homologação local executadas, com ressalvas explícitas. Entrega
preparada para revisão do Atlas; **a separação operacional Web/Worker não está
liberada por este relatório**. Os limites de concorrência e do fallback local
abaixo precisam de aceite técnico antes do encerramento da fase. Não há aprovação
para dinheiro real, deploy, banco remoto, múltiplos workers ou FASE 2.6C.

- HEAD inicial verificado: `c0b1f68214114460f0b07300044ac067e35b09d5`, igual ao informado.
- Base: `atlas/infra-1-1-persistencia-segura`.
- Branch exclusiva: `atlas/infra-1-2-integridade-concorrencia`.
- Commit de implementação e testes: [`20d1cd36613fa5d1245acd8d1848e061fdddc30d`](https://github.com/oliveirasamuel03-jpg/robo-investimentos/commit/20d1cd36613fa5d1245acd8d1848e061fdddc30d).
- O commit que introduz este relatório reúne as evidências finais; identificar por
  `git log -1 -- docs/retomada/ATLAS_INFRA_1_2_INTEGRIDADE_CONCORRENCIA.md`.
- Nenhum PR, merge, deploy, alteração da main ou chamada operacional ao Railway.
- Nenhum acesso a outro repositório, PostgreSQL antigo ou DATABASE_URL herdada.

## 1. Arquitetura auditada

| Componente | Contrato e concorrência |
|---|---|
| `core/persistence.py` | Alterna entre arquivos JSON/CSV e PostgreSQL. Banco mantém `app_state(namespace,payload_json,updated_at)` e `app_rows(seq,namespace,row_json,created_at)`, com payloads TEXT. |
| `core/state_store.py` | `bot_state`: controle, segurança, heartbeat, diagnóstico e espelho de posições PAPER. Web e worker faziam read-modify-write do documento inteiro. |
| `core/auth/users_store.py` | Usuários no namespace `auth_users_store`; bootstrap, inclusão, papéis e bloqueios competiam pelo mesmo documento. |
| `paper_trading_engine.py` | Estado PAPER, posições por ativo e histórico limitado de patrimônio; trades em outro namespace. Antes gravava trades/relatórios antes do estado final. |
| `engines/trader_engine.py` | Checa STOPPED, executa ciclo PAPER, sincroniza posições/ordens e risco. Sincronização de ordens faz read-deduplicate-replace por timestamp/ativo/lado. |
| `workers/trader_worker.py` | Loop existente, heartbeat e manutenção. Não foi iniciado nenhum worker. Retenção publicava catálogo depois de mover os registros. |
| `core/retention.py`, `core/trader_reports.py` | Leitura/cópia/substituição de históricos e resumos semanais. Arquivos de runtime/archive pertencem aos namespaces existentes. |
| Páginas, `core/alerts.py`, `core/swing_validation.py` | Alguns saves integrais continuam; snapshots desatualizados passam a ser recusados. |

Não foi criada outra fonte de verdade, outro worker, tabela ou coluna. O espelho
em `bot_state` já existia. Os novos tokens de conteúdo ficam em atributos Python
de `StateSnapshot`, fora do JSON persistido.

## 2–3. Problemas encontrados e severidade

| Severidade antes da correção | Falha | Tratamento |
|---|---|---|
| Crítica | Worker/Web sobrescreviam estado novo com leitura antiga, inclusive STOPPED/segurança. | CAS, atualização transacional e recusa explícita de conflito. |
| Crítica | Trade/relatório persistido sem saldo/posição correspondente após interrupção. | Uma transação PostgreSQL cobre o ciclo PAPER e sua sincronização com a plataforma. |
| Alta | Duas remoções/rebaixamentos podiam eliminar ambos os administradores; bootstrap/usuários podiam duplicar ou desaparecer. | Guardas e mutações sob o mesmo lock/transação, rechecagem do bootstrap. |
| Alta | Store de usuários semanticamente inválido era filtrado e depois regravado sem os registros descartados. | Corrupção, duplicatas e tipos inválidos interrompem a operação, preservando os dados. |
| Alta | Namespace ausente importava automaticamente JSON/CSV antigo do disco. | PostgreSQL inicia com defaults/linhas vazias, sem importação implícita. |
| Alta | Retenção podia perder append entre leitura e substituição, ou publicar arquivo sem catálogo. | Transação cobre leitura, arquivo, remoção, resumo e catálogo do worker. |
| Alta | CSV corrompido virava tabela vazia; JSON aceitava NaN/Infinity/overflow e chaves repetidas. | Erro explícito, preservação do original e serialização JSON estrita. |
| Alta | Append local e inicialização de arquivos disputavam escrita e perdiam registros. | Locks POSIX por caminho estável e inicialização dentro do lock. |
| Média | DDL concorrente e cache global de schema independente da URL. | DDL serializado/transacional e cache associado à URL. |
| Média | SQL de banco novo verificava somente `public`. | Recusa relações em todos os schemas de usuário. |

As três regressões da base reproduzidas de fato com dados fictícios foram:
snapshot antigo desfez STOPPED; CSV truncado foi substituído silenciosamente;
dois appends concorrentes preservaram apenas **1 de 2** registros. O código antigo
foi carregado diretamente de `git show c0b1f682:core/persistence.py` em diretório
temporário; uma barreira sincronizou as duas leituras do histórico. Evidência:
[reproduções](evidencias/infra12-legacy-reproductions.json).

## 4. Correções realizadas e impacto na arquitetura

- `update_json_state` executa leitura, callback, validação e escrita sob transação.
  Autenticação e atualizações pequenas de estado usam esse contrato.
- PostgreSQL usa advisory lock transacional global antes de locks de namespace.
  A escolha conservadora serializa todos os escritores cooperantes e evita
  inversão de ordem em logs, estado, autenticação e retenção aninhados. Não é uma
  solução de alta capacidade. `lock_timeout=5s` produz erro explícito; não há retry
  cego, fallback para arquivo ou continuação com defaults após falha de banco.
- `save_json_state` compara origem/conteúdo com o snapshot lido; linha existente
  também recebe `FOR UPDATE` e UPDATE condicional. Save não versionado no banco
  só insere ou aceita texto idêntico: não sobrescreve estado existente diferente.
- `persistence_transaction` reutiliza a conexão na unidade de trabalho; escopos
  aninhados usam savepoints. Uma exceção Python capturada pelo chamador não
  confirma parcialmente a subunidade que falhou.
- Ciclo PAPER, ciclo trader, reset/sincronização e retenção usam essa unidade no
  PostgreSQL. Nenhum cálculo de estratégia foi alterado. Rollback pode deixar
  lacunas normais na sequência BIGSERIAL; sequência contínua não é invariante.
- Substituição valida histórico atual antes de DELETE e confirma tudo de uma vez;
  corrupção exige investigação, não limpeza automática.
- Arquivos locais usam `RLock` + `flock` em sidecar estável, arquivo temporário,
  fsync, rename e fsync do diretório. Inicialização e append compartilham o lock.

## 5. Arquivos modificados

Produção: `core/persistence.py`, `core/state_store.py`,
`core/auth/users_store.py`, `core/integrity_diagnostics.py`, `core/retention.py`,
`paper_trading_engine.py`, `engines/trader_engine.py`, `workers/trader_worker.py`.

Homologação: `scripts/sql/new_history_schema.sql`,
`scripts/test_infra12_postgres.py`, `tests/test_infra12_persistence.py`,
`tests/test_state_auth_concurrency.py`, `tests/test_integrity_diagnostics.py`,
`tests/test_infra12_postgres.py`, `tests/test_infra12_retention_postgres.py`.
Este relatório e os arquivos `docs/retomada/evidencias/infra12-*` registram a execução.
Os testes antigos não foram alterados para esconder falhas. Dependências de
produção e requisitos operacionais permanecem nos arquivos existentes.

## 6. Testes executados e evidências

| Execução final | Coletados | Aprovados | Falhas/erros | Skips | Warnings | Tempo |
|---|---:|---:|---:|---:|---:|---:|
| Baseline verdadeiro do commit c0b1f682, via git archive | 349 | 349 | 0 | 0 | 42 | 16,76s |
| Suíte offline final | 451 | 420 | 0 | 31 | 48 | 11,82s |
| PostgreSQL 18.6 local final | 31 | 31 | 0 | 0 | 1 | 18,05s |
| PostgreSQL 16.15 local final | 31 | 31 | 0 | 0 | 1 | 17,52s |

São **451 casos distintos**: 349 anteriores + 71 novos offline + 31 de integração.
Os 31 skips offline foram executados separadamente nas duas versões PostgreSQL;
nenhum desses casos ficou pendente. As três reproduções da base são verificações
adicionais, não foram somadas ao total pytest. Não se somam reexecuções como testes
distintos. O runner offline bloqueou 50 tentativas de rede/processo (baseline: 38).

Evidências: [baseline](evidencias/infra12-baseline.txt),
[offline](evidencias/infra12-offline.txt), [PG18](evidencias/infra12-postgres18.txt),
[PG16](evidencias/infra12-postgres16.txt), com JUnit XML de cada execução final.
[Checks](evidencias/infra12-checks.txt): compileall, pip check e git diff --check,
todos com exit code 0. Warnings: datetime.utcnow legado e depreciações pandas
(`H` e concat de frame vazio). Não foram suprimidos. Os logs TXT tiveram apenas espaços finais e linhas vazias
finais normalizados para higiene Git; resultados e mensagens foram preservados.

Ocorrências intermediárias não omitidas: houve coleta com imports temporariamente
incompletos durante a integração dos patches; uma tentativa de baseline falhou
com ENOSPC (5 failed/204 passed/140 errors/18 warnings), e outra suíte em andamento
foi afetada pelo mesmo espaço temporário (7 failed/159 passed/19 skipped/236 errors).
O [excerto de ENOSPC](evidencias/infra12-baseline-environment-failure.txt) mantém
hash e resumo do log integral. Imagens Docker criadas nesta sessão foram removidas
sequencialmente; não foram removidos volumes antigos. A primeira rodada PG18 teve
17 passed/2 failed por fixtures novas (diretório ausente e suposição incorreta de
STOPPED como default). As fixtures foram corrigidas para o contrato real PAUSED;
o comportamento de produção não foi alterado. Todas as execuções finais acima
foram repetidas após essas correções, sem falhas.

### Cobertura dos requisitos A–L

- A: 8 sessões incrementam o mesmo estado até 96; 4 processos Python até 32;
  80 appends únicos; 24 substituições concorrentes sem misturar históricos.
- B/C/D: falhas Python e constraint SQL revertem estado/histórico; outro namespace
  fica preservado; leitor não enxerga alterações ainda não confirmadas.
- E: interrupção de backend PostgreSQL, falha explícita e conexão nova bem-sucedida.
- F/G: JSON inválido, NaN/Infinity/overflow/chaves duplicadas e CSV truncado;
  quoting CSV e roundtrip JSON/CSV; nenhum dado real.
- H/I: banco vazio, recusa em banco com relações inclusive fora de public,
  não importação de arquivo antigo e recusa de save desatualizado/não versionado.
- J: pg_dump/pg_restore sintéticos com comparação das duas tabelas e sequência
  avançando corretamente após restauração. Não é backup de dados do usuário.
- K/L: gravação/leitura STOPPED, default legado seguro PAUSED, PAPER obrigatório e
  broker sem autorização para ordens reais. O teste STOPPED impede chamar ciclo.
- Adicionais: último administrador concorrente, username único, savepoint após
  exceção capturada, timeout sem overwrite, rollback no motor PAPER real depois
  do append do trade, e retenção interrompida/concorrente com append.

## 7. Compatibilidade PostgreSQL e banco novo

PostgreSQL **18.6** (imagem oficial `postgres:18`) e **16.15**
(`postgres:16-alpine`) foram executados gratuitamente no Docker local. Digests
estão nos logs. Contêiner exclusivo por execução, porta aleatória publicada só
em 127.0.0.1, credencial sintética temporária, banco novo por teste e dados em
tmpfs. O runner limpa variáveis herdadas, não aceita DSN externo e remove somente
seu próprio contêiner. Nenhum contêiner de homologação permaneceu em execução.

O SQL preparado continua sem DROP/TRUNCATE, INSERT de usuários/operações,
ativação de worker ou importação. Recusa qualquer relação de usuário pré-existente
(tabela, partição, view, materialized view, sequence ou foreign table).
Foi executado somente nos bancos descartáveis. Não houve migração remota.
Um banco novo inicia **histórico do zero** e não representa recuperação do antigo.

## 8. Segurança PAPER

Ambos os runners impõem `BROKER_MODE=paper` e `BROKER_PROVIDER=paper`, descartam
credenciais herdadas e não iniciam loops Web/Worker. Testes usam preços e usuários
fictícios. Não houve ordem real, broker real, mudança de provider operacional ou
ativação de flags experimentais. A suíte existente de segurança e STOPPED passou.
BOS, Fibonacci, pivôs, multi-timeframe, webhooks e shadow continuam sem autoridade
operacional. Capital, ticket, score, thresholds, min_signal_score,
max_open_positions e regras da estratégia não foram modificados.

## 9. Integridade financeira: contrato diagnóstico

`diagnose_paper_integrity` é função pura, sem chamada no caminho operacional,
sem reparação e sem gatilho. Seus 39 testes incluem BUY/SELL do motor real com
preços sintéticos. Valida o contrato já praticado pelo código:

- Valores monetários finitos; capital, saldo e patrimônio não negativos;
  quantidade/preço/gross positivos; custo não negativo; PnL admite sinal.
- Uma posição por ativo normalizado e referência de trade; repetição de
  `(trade_id, side)` é duplicação. BUY e SELL com mesmo ID são válidos.
- SELL é comparado ao BUY disponível em ativo/quantidade e realização de PnL;
  ausência de BUY só é acusada quando o histórico foi declarado completo.
- Patrimônio = saldo + quantidade × último preço das posições; último snapshot
  de patrimônio/saldo/número de posições coincide com o estado.
- No contrato PAPER atual custo é zero e saldo varia pelo gross_value; tolerância
  de um centavo para arredondamento, com preço persistido a seis casas.
- Reconciliação global de cash/PnL/posições só é feita com
  `complete_trade_history=True`, histórico completo desde reset e em ordem.
  Consulta com limit/retention não pode ser apresentada como ledger completo.

JSON estrito é proteção de formato da persistência; os achados financeiros
adicionais continuam apenas diagnósticos/testes, sem mudar decisões de negociação.

## 10. Limitações de concorrência e riscos residuais

| Severidade residual | Limite e condição de uso |
|---|---|
| Alta | Arquivos locais têm atomicidade por arquivo, não entre saldo/trades/relatórios/catalog. Interrupção entre arquivos ainda pode divergir; não liberar Web/Worker separados nesse backend. |
| Alta | Advisory locks dependem de todos os escritores adotarem esta versão. Nunca misturar INFRA-1.1/SQL manual com estes ciclos; CAS de estado é reforçado por row lock/UPDATE condicional, mas isso não torna escritas externas um ledger transacional. |
| Média | Lock global é mantido durante fetch de mercado no ciclo; outro writer pode receber timeout de 5s. STOP concorrente pode precisar ser reenviado após recarga; nenhum commit de STOP é desfeito por snapshot antigo. Não é promessa de parada imediata durante ciclo já iniciado. |
| Média | Saves integrais de páginas/alerts/swing podem lançar conflito visível. Não há UI dedicada para resolver conflitos nem retry automático. Recarregar e reaplicar a intenção após revisão. |
| Média | CAS por conteúdo não detecta ABA (conteúdo muda e volta ao mesmo texto); dict puro local mantém substituição explícita legada. Converter Snapshot em dict perde token; no banco a sobrescrita passa a ser recusada. |
| Média | Duas leituras independentes podem observar commits diferentes; ausência de dirty read não equivale a snapshot financeiro agregado. Para auditoria coerente usar o contexto transacional existente que coordena os escritores. |
| Média | Não foi adicionado índice único de trade_id/idempotency key. Diagnóstico detecta duplicata; transação evita a duplicação causada por commit parcial, sem prometer exactly-once para reenvio arbitrário externo. |
| Baixa | Namespaces de tabelas continuam baseados no basename; caminhos novos precisam de nomes únicos. Locks locais requerem POSIX/Linux e sidecar preservado durante uso. |

Callbacks transacionais devem ser curtos e sem efeitos externos. Objetos lidos
ou gravados dentro de transação revertida devem ser descartados/recarregados:
o token em memória pode já ter avançado. Se fsync do diretório falha após rename,
o arquivo novo pode já estar visível; erro não significa que o arquivo antigo
continua publicado. Não se tenta compensar apagando ou sobrescrevendo histórico.

## 11. Recuperação do banco antigo

**Não realizada, não acessada e não validada nesta missão.** Nenhuma afirmação de
recuperação, integridade, disponibilidade ou descarte do banco/volume antigo.
Todos os backups/restores desta fase são sintéticos e locais. Credenciais e
DATABASE_URL antigos não foram usados. Recuperação real permanece fase separada.

## 12–13. Impacto e pendências de homologação

Sem schema remoto novo ou migração irreversível. A maior alteração é o contrato
explícito de atualização e falha por conflito/corrupção. Atlas deve revisar:

1. Aceite da serialização global e do timeout; medir ciclo/fetch com dados
   sintéticos antes de decidir por locks de agregados menores.
2. Tratamento de conflitos nas telas; controles de parada durante ciclo em curso.
3. Proibir coexistência de versões escritoras e definir procedimento de troca
   coordenada, sempre com um único worker, antes de qualquer publicação futura.
4. Isolar uma unidade transacional de leitura/reconciliação e estudar chave única
   de idempotência; qualquer migração futura exige proposta, testes e autorização.
5. Manter fallback de arquivos fora da separação Web/Worker; não interpretar
   diagnóstico de histórico limitado como reconciliação completa.
6. Confirmar previews/autodeploy em ação específica futura. Não foi criado PR e
   não foi consultado/alterado o projeto Railway `perceptive-integrity`.

## 14. Critérios de rollback

Considerar rollback se houver timeout recorrente, recusa inesperada de dados
legados válidos ou divergência nos ensaios de saldo/trades/retention. Antes de
qualquer rollback futuro: parar escritores de forma coordenada, preservar backup
verificado do banco correto e revisar evidências. Reverter o commit de implementação
na branch apropriada, sem apagar dados, remover locks como remendo ou misturar
versões ativas. Não há mudança de schema que exija DROP/TRUNCATE nesta fase.
Reverter código para INFRA-1.1 reintroduz riscos identificados e não autoriza
reativar worker. Nenhum rollback remoto foi executado.

## 15. Próxima fase recomendada

Revisão Atlas de INFRA-1.2 e homologação de conflitos/parada/idempotência com
PostgreSQL local, PAPER e dados sintéticos. Depois, proposta separada para
separação Web/Worker e publicação, condicionada à segurança de autodeploy e às
autorizações específicas. Não avançar estratégia, FASE 2.6C ou execução real.

### Como repetir localmente

```bash
.venv/bin/python scripts/test_offline.py -q
# Imagens oficiais precisam estar presentes; executar uma versão por vez.
.venv/bin/python scripts/test_infra12_postgres.py --version 18
.venv/bin/python scripts/test_infra12_postgres.py --version 16 --alpine
.venv/bin/python -m compileall -q core engines workers paper_trading_engine.py scripts tests
.venv/bin/python -m pip check
git diff --check
```

Os runners recusam `.env` local. Use checkout limpo e ambiente de desenvolvimento;
nunca substitua o DSN sintético por DATABASE_URL remota. Não executar o SQL em
banco existente para repetir estes testes.
