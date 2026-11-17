# ATLAS — INFRA-1.2.7: retenção e quarentena

17/11/2026 UTC. **PAPER_ONLY / SECURITY_FIRST / COST_CONTROLLED /
LOCAL_TESTS_ONLY / NO_PRODUCTION_DEPLOY.**
Repositório exclusivo `oliveirasamuel03-jpg/robo-investimentos`.

## 1. Base, branch e escopo

HEAD remoto da INFRA-1.2.6 confirmado antes de editar:
`faf3fa26b6d8c9d8edf45e8b8e8177b54dcdad05`.
A branch remota `atlas/infra-1-2-7-retencao-quarentena` já existia apontando
para essa mesma base; não foi sobrescrita por outra origem.
A árvore original continha artefatos da execução anterior. Foram preservados;
esta fase usa uma worktree limpa do **mesmo repositório**, criada nessa base e
na branch obrigatória. Nenhuma alteração em main ou outro repositório.

A autorização de execução é o comando do usuário desta fase. O documento
`ATLAS_INFRA_1_2_7_CRITERIOS_CORRETIVOS.md` da base é um plano anterior, não
uma autorização independente nem motivo para interromper o trabalho autorizado.
O protótipo de snapshot rejeitado não foi integrado ao runtime.

<!-- COMMITS -->
- Implementação: [`6d33fd7`](https://github.com/oliveirasamuel03-jpg/robo-investimentos/commit/6d33fd766bb3b4aba4f6f5712436d0ffed9a4c17),
  retenção sem reescrita e quarentena persistente.
- Endurecimento do catálogo: [`2787104`](https://github.com/oliveirasamuel03-jpg/robo-investimentos/commit/27871043b66a6a60cd28a40ad07f5d6c98e7bf9d),
  patch bruto de metadados, recusa de colisão de namespace e mensagens da UI.
- Contenção sem recibo anterior: [`abcfadb`](https://github.com/oliveirasamuel03-jpg/robo-investimentos/commit/abcfadbb5cdf4ca4b8af2c57bbaba33a0fc36f57),
  revalidação de dados confirmados, inclusive quando a gravação da barreira falha.
- O commit documental de entrega é identificado por
  `git log -1 --format=%H -- docs/retomada/ATLAS_INFRA_1_2_7_RETENCAO_QUARENTENA.md`
  e pelo link apresentado na entrega. Seu hash não é inserido no próprio conteúdo.
<!-- END_COMMITS -->

## 2. Defeitos reproduzidos na implementação anterior

Executados os 12 contraexemplos de `tests/test_infra126_postgres.py` contra a
base, antes das alterações: **12 passaram**, comprovando as limitações antigas,
não aprovando o comportamento. Evidências: `infra127-before18.txt/.xml` e
`infra127-before-contract-pg18.json`.

| Defeito | Causa anterior | Resultado corrigido |
|---|---|---|
| No-op alterava histórico e recibo | `_archive_old_rows` normalizava, deduplicava e substituía retidos mesmo sem linhas antigas; recriava seq e descartava cycle_id/colunas extras | No-op retorna antes de qualquer substituição; objetos, seq, timestamps, cycle_id e manifesto permanecem iguais |
| Mesma chave, conteúdo diferente: uma linha sumia | `keep="last"` sobre chave parcial, incluindo dedupe do próprio runtime | Originais financeiros nunca são deduplicados. Conflitos de conteúdo são identificados e preservados, com quarentena auditável |
| UNKNOWN com gate livre | `reconcile` retornava blocked sem gravar controle quando não havia intenção ativa | Quarentena própria, persistida e relida; se a gravação falhar, resultado UNCONFIRMED e revalidação obrigatória antes de nova operação |
| Arquivo quebrava o manifesto assinado | Recibos v1 assinam namespace/seq/conteúdo runtime; não há protocolo de transformação para arquivo | Arquivamento financeiro com recibos é recusado, sem mover/reassinar dados. Exige fase futura de compatibilidade |
| Histórico antigo podia virar preimage de ciclo novo | Início verificava intenção ativa, sem revalidar a prova financeira anterior | Recibo, manifesto e validações financeiras anteriores são reavaliados sob fence antes de admitir nova intenção |

Dois casos adicionais foram reproduzidos durante a revisão da própria correção:
o normalizador de estado usado para salvar catálogo alterava capital sintético de
10.000 para 1.000; o catálogo agora usa patch JSON bruto, sem normalizar campos
operacionais. Também foi comprovada colisão entre namespaces de arquivo/runtime
derivados do mesmo basename: antes uma linha virava zero; agora a transformação
é recusada e a linha permanece. Evidências negativas separadas:
`infra127-normalization-before-fix18.*`, `infra127-namespace-before-fix18.*` e
`infra127-catalog-risks-before.json`. O teste exploratório anterior
`infra127-catalog-before-fix18.*` falhou por problemas da fixture, não é usado
como prova desses defeitos.

Uma segunda revisão reproduziu **três falhas de regressão**: conflitos sem
recibo anterior e conflito entre arquivo/runtime podiam reabrir o gate se a
gravação da quarentena falhasse ou fosse desfeita pelo chamador. Os originais
não eram apagados, mas a ausência de controle podia ser tomada como permissão.
`infra127-unanchored-before18.*` preserva os contraexemplos anteriores ao terceiro
commit. A solução abaixo consulta também a evidência nos próprios dados.

## 3. Arquitetura e estados de contenção

Continua uma fonte financeira autoritativa: estado/linhas existentes no PostgreSQL.
Não foi criado outro ledger ou revisão usada como prova. A nova entrada
`app_state[paper_integrity_quarantine]` é **controle de segurança**, não estado
financeiro. Usa as tabelas existentes; nenhuma coluna, índice, trigger, role ou
migração foi acrescentada pela correção.

| Situação | Controle / comportamento |
|---|---|
| Intenção financeira pendente | `paper_commit_control.active_cycle_id`; exige prova positiva de COMMITTED ou NOT_COMMITTED |
| Divergência sem intenção ativa | Quarentena com ID próprio, ciclo relacionado, motivo e evidências; intenção e STOP não são substituídos |
| Retenção conflitante/corrompida | Quarentena de integridade; originais preservados; recibo financeiro isolado não libera conflito de arquivo |
| Persistência ainda em transação externa | `STAGED_IN_CALLER_TRANSACTION`; não é confirmação durável |
| Commit do controle e readback observados | `CONFIRMED` |
| Falha na gravação/readback/ACK | `UNCONFIRMED`; nunca anunciar que a barreira foi confirmada |
| Barreira de intenção já existente | `EXISTING_INTENT`; manter identidade atual, sem reexecutar operação |
| Quarentena não gravada, mas conflito/arquivo/controle ausente detectável nos dados | `PERSISTED_INTEGRITY_EVIDENCE`; bloqueado e tentativa de materializar contenção, sem execução financeira |

A quarentena é consultada por `gate_status`/guardas e por `begin_intent`.
`record_unit_result` a verifica novamente dentro da unidade financeira,
impedindo confirmar rascunho após contenção persistida entre intenção e commit.
Continuam na mesma transação saldo, posição, trades, ordens, relatório financeiro
e recibo. Não há retry financeiro automático.

A ordem é a do fence transacional: uma contenção persistida antes da unidade
financeira impede seu commit. Uma solicitação que só obtém o fence depois de
um commit financeiro não desfaz esse commit. Nem quarentena nem STOP prometem
cancelamento retroativo; o resultado confirmado continua sendo registrado.

O lock global e a ordem de namespaces permanecem os existentes. O timeout de
lock não foi aumentado. `assert_execution_allowed` também verifica a evidência
anterior em PostgreSQL. `begin_intent` repete a prova no fence definitivo,
impedindo aceitar uma adulteração entre preflight e criação da intenção.

A evidência de dados confirmados é lida em conexão própria, transação
`REPEATABLE READ, READ ONLY`, com limites de cinco segundos. Controle e dados
são lidos no **mesmo snapshot MVCC**, evitando confundir o primeiro commit
concorrente com recibo órfão. Essa conexão não enxerga uma unidade financeira
ainda incompleta do chamador. O snapshot é apenas uma leitura de segurança;
não é o protótipo rejeitado nem substitui a prova integral no fence. Nenhum
timeout existente foi aumentado.

## 4. Falha ao persistir e reinício

Uma falha de escrita não pode ser transformada em promessa de persistência.
`reconcile` registra UNKNOWN com confirmação de controle explícita. Uma
transação externa pode desfazer um controle que estava apenas STAGED; o
resultado não o chama de confirmado.

Para fechar essa janela após restart, a próxima tentativa financeira não se
baseia apenas na ausência da quarentena: exige prova nova do manifesto/recibo
anterior, usando o mesmo validador de estado, identidade, eventos, referências,
valores e espelho da reconciliação. Se a divergência permanece, recria contenção
quando possível e recusa a operação. PostgreSQL inacessível ou controle JSON
inválido falham fechados. Isso é leitura/controle de segurança, não retry BUY/SELL.

Quando não há recibo anterior, a leitura independente detecta duplicatas e
conteúdos conflitantes nos relatórios/ordens persistidos, recibos órfãos, identidade
de ciclo sem controle e arquivo financeiro sem custódia comprovada. Esses dados
continuam bloqueando mesmo se o latch não puder ser gravado ou se o chamador
desfizer a transação. `assert_execution_allowed` tenta registrar a evidência e
sempre recusa a execução; `begin_intent` repete a verificação sob fence. Não há
dependência apenas da RAM nem liberação por ausência de exceção anterior.

Há teste em novo interpretador Python para gravação de quarentena indisponível
e para rollback externo, sem estado de RAM compartilhado. Também há teste de
resposta perdida **injetada em Python após um COMMIT real do controle**: conexão
independente observa a barreira, mas a aplicação retorna UNCONFIRMED. Esse caso
não é apresentado como interrupção real de TCP.

O teste de backup também executa **SIGKILL e reinício real do PostgreSQL próprio**
com quarentena ativa, `fsync=on` e `synchronous_commit=on`: compara os controles e
as linhas financeiras antes/depois e verifica STOP não confirmado enquanto o
banco está indisponível. O log de recuperação do servidor é registrado, sem DSN.

## 5. Liberação, identidade e diagnósticos atrasados

Quarentena não é removida por heartbeat, START, reset, timeout ou falha de leitura.
O namespace de controle é protegido pelas mesmas guardas que protegem recibos.
JSON malformado, estrutura inválida ou evidência ausente em controle ativo não
viram default livre.

Para liberar quarentena causada por divergência de reconciliação são necessários:
prova positiva atual, pedido explícito `release_quarantine=True`, ID da geração
observada (`expected_quarantine_id`) e vínculo com o ciclo verificado. Uma
requisição antiga não libera uma geração nova. Por padrão, verificar não libera
quarentena. As evidências de episódios resolvidos são conservadas em history;
nenhum histórico de quarentena é limpo automaticamente.

Conflitos de retenção/arquivo e corrupção histórica não são resolvidos somente
porque o recibo do último ciclo está correto. Permanecem bloqueados para revisão
de integridade específica; não foi implementado reparo/deduplicação automático.
A prova de reparo no teste usa exclusivamente dados sintéticos e SQL da fixture.
Não existe API operacional nova de reparo financeiro.

Diagnóstico A atrasado não muda active_cycle_id, quarentena nem recibo atual B.
Recibo histórico é explicitamente histórico; não autoriza liberação atual. A
interface só solicita liberação com o ID que exibiu e não apresenta confirmação
financeira como liberação de quarentena ainda ativa. A reconciliação não escreve
RUNNING, não inicia worker e não substitui START do operador.

## 6. Retenção, duplicatas e arquivo

São comparados objetos completos, incluindo cycle_id e campos extras; a chave
de negócio identifica possíveis conflitos, não prova igualdade de conteúdo.
O hash usa JSON canônico: mera mudança da ordem das chaves do objeto não cria
um conflito artificial. A ordem dos eventos e sua identidade continuam relevantes.
Evidências de conflito incluem namespace, chave, identidade de origem, hashes
dos payloads, ciclo quando disponível e motivo. As linhas originais permanecem
no armazenamento; não são substituídas por hashes.

Duplicatas financeiras idênticas são contadas e preservadas no runtime, sem
limpeza automática. Preservar uma duplicata não certifica um ledger adulterado:
o teste acrescenta uma duplicata sintética e comprova que a prova anterior
impede nova operação. Nas projeções, dedupe elimina somente conteúdo integral
idêntico; conteúdos distintos com mesma chave continuam distintos.

Dados financeiros já vinculados a recibos v1 **não são arquivados**: a manutenção
retorna `FINANCIAL_ARCHIVE_REQUIRES_VERSIONED_RECEIPTS`, sem alterar manifesto,
seq, metadados ou recibos. Essa recusa de manutenção não inventa um STOP nem
torna incerto um commit financeiro previamente comprovado.

Arquivamento PostgreSQL de linhas sintéticas/legacy sem recibos é permitido
somente preservando os objetos completos e publicando o catálogo na mesma
transação da cópia/remoção. Linhas que referenciem ciclos sem seus recibos não
recebem compatibilidade presumida. Arquivamento financeiro no modo arquivo
local é recusado (`LOCAL_FINANCIAL_ARCHIVE_NOT_ATOMIC`), pois não há prova de
atomicidade multi-arquivo. Logs não financeiros continuam com arquivamento
transacional e teste de rollback/concorrrência.

Esse arquivo legacy não conserva a identidade SQL `seq` do namespace de origem
como recibo de transformação; as identidades presentes no objeto são mantidas.
Por isso não se declara equivalência de manifesto ou vínculo de recibo para
essa transformação. Históricos assinados, inclusive seus `seq`, são preservados
pela recusa da transformação; no-op preserva todos os `seq` e timestamps.

Preservar fisicamente um arquivo legacy **não autoriza operar**: enquanto houver
arquivo financeiro ou catálogo sem protocolo verificável de custódia, a execução
financeira fica bloqueada. A retenção pode ler e diagnosticar a evidência sem
apagá-la; intenção ativa/quarentena confirmada impedem a manutenção. Arquivos
com recibos v1 não são transformados. Uma reconciliação COMMITTED pode confirmar
o resultado financeiro e ainda manter a barreira de integridade do arquivo.

O job faz preflight financeiro antes de mover logs ou gerar projeções. Quando
conflito é encontrado, publica somente contenção/diagnóstico e não aceita
modificações parciais de dados. O worker distingue resultado bloqueado de
sucesso. Catálogos de ordens e relatórios semanais existentes são conservados.

Resumos e CSVs semanais não são regravados quando o conteúdo permanece igual;
generated_at representa a geração do conteúdo conservado. O teste repete o
job completo com as mesmas entradas e comprova ausência de writes de dados.
O bookkeeping do worker (última tentativa/erro e log) é separado da transformação
dos históricos; pode registrar uma tentativa sem alterar evidência financeira.

## 7. STOP e segurança PAPER

STOP permanece independente da quarentena: request, persistência/readback e
ACK são etapas distintas. Há provas de STOP com quarentena ativa/corrompida,
durante feed pausado e após concorrência. Banco indisponível/timeout não é
apresentado como STOP confirmado; a suíte anterior preserva essas regressões.
Snapshot antigo e atualização de metadados não liberam contenção nem reativam
robô parado.

BROKER_MODE=paper e BROKER_PROVIDER=paper são impostos pelo runner sanitizado.
Não foram alterados estratégia, broker/provider, capital, ticket,
max_open_positions, score, min_signal_score, thresholds, frequência ou autoridade
SHADOW_ONLY. O validador financeiro foi extraído sem mudar suas regras e
reutilizado; não virou sinal operacional. Nenhuma ordem real ou worker contínuo.

## 8. Regressões convertidas e compatibilidade

Os casos anteriores que comprovavam defeitos foram mantidos e convertidos:

- No-op agora exige bytes/seq/cycle_id/manifesto preservados e COMMITTED.
- Conflito exige as duas linhas intactas e quarentena persistida, sem keep-last.
- Backup preserva dados e controle bloqueado de recibo sintético adulterado;
  retenção já não é a causa da adulteração.
- O teste de retenção/recibo/snapshot da 1.2.5 exige recusa da transformação
  incompatível e prova atual preservada.
- Arquivo financeiro local agora exige recusa sem perda; o cenário de arquivo
  válido, com comparação de objetos e catálogo, está coberto em PostgreSQL.
- SQL não administrativo que apaga o controle ainda consegue realizar DML na
  fixture, mas a aplicação detecta recibo/identidade órfãos e bloqueia. O antigo
  contraexemplo foi convertido nessa regressão; nenhuma ACL foi alterada.

Não foram removidos casos para obter aprovação. As provas anteriores de STOP,
COMMIT incerto, TCP local, restart PG, rollback, backup/restore e PAPER continuam
na regressão. Formato dos recibos financeiros v1 e tabelas existentes preservado.

A coleta real da base encontrou **685 casos**, incluindo os contraexemplos da
1.2.6; os 667 informados como referência não incluem todos os casos presentes
nesse HEAD. A comparação é feita pelos IDs coletados, sem diminuir a base.

Arquivos de código/teste alterados nesta fase:

| Arquivo | Alteração |
|---|---|
| `core/retention.py` | No-op, preservação integral, conflitos, atomicidade do catálogo e recusa de arquivo incompatível |
| `core/commit_protocol.py` | Quarentena, prova anterior, contenção sem latch, identidade e liberação explícita |
| `core/ui_mutations.py` | Liberação vinculada à geração observada e mensagens sem falso sucesso |
| `workers/trader_worker.py` | Retenção bloqueada é registrada como falha, sem anunciar sucesso |
| `scripts/test_infra127_postgres.py` | Runner isolado que conserva as suítes anteriores e acrescenta esta fase |
| `tests/test_infra127_postgres.py` | Regressões novas, falhas, concorrência, restart e restore sintéticos |
| `tests/test_infra126_postgres.py` | Contraexemplos convertidos em comportamento seguro |
| `tests/test_infra125_postgres.py` | Retenção assinada deve ser recusada, sem legitimar manifesto alterado |
| `tests/test_retention.py` | Arquivo financeiro local recusado sem perda e projeções íntegras |
| `tests/test_infra123_identity.py` | Reconciliação histórica sem liberação da contenção atual |
| `tests/test_infra124_postgres.py` | JSON inválido detectado antes da intenção exige contenção persistida, sem alterar dados ou criar recibo |

Relatório e evidências usam somente caminhos `docs/retomada/...infra127...`.
As oito fontes de estratégia/configuração/STOP/persistência básica comparadas
em `infra127-source-audit.json` permanecem idênticas byte a byte à base.

Isso não autoriza versões antigas a escrever simultaneamente: elas desconhecem
a quarentena nova. Atualização operacional futura exige quiescência e controle
de versão de **todos** os escritores. O contrato de SQL privilegiado da 1.2.6
continua um limite explícito, não uma garantia corrigida por contador.

## 9. Execuções e evidências

<!-- FINAL_RESULTS -->
| Execução real | Aprovados | Falhas | Pulados / desmarcados | Warnings |
|---|---:|---:|---:|---:|
| Contraexemplos na base, antes da correção | 12 | 0 | 0 | 28 |
| Offline final (`infra127-offline-complete.*`) | 534 | 0 | 175 pulados, pois exigem PG | 104 |
| Suíte PG18.6 completa do runtime final (`infra127-postgres18-release.*`) | 173 | 2 expectativas de exceção antigas | 0 | 640 |
| Segurança final, após converter as expectativas (`infra127-safety-final18.*`) | 48 | 0 | 3 benchmarks desmarcados pela seleção | 157 |
| PostgreSQL 16 | 0 | não avaliado | 175 não executados | não avaliado |

**709 casos distintos cobertos ao consolidar as execuções:** 534 offline e
175 PostgreSQL 18. Os 48 da reexecução se sobrepõem à suíte completa; não são
somados como casos novos. Os dois casos com expectativas antigas passaram com
as novas assertions de preservação e contenção. **Não houve uma única execução
completa verde dos 175 após a conversão dos testes**; houve suíte completa e
reexecução de impacto, com runtime idêntico. As falhas anteriores não foram
apagadas. A base de 685 IDs foi preservada integralmente, com 24 casos novos.

Offline: 99,82 s. Suíte PG completa: 1.548,44 s. Reexecução de segurança:
175,13 s. Os resultados por ID, as 14 regressões obrigatórias e os resumos
de todas as 19 execuções estão em `infra127-checks.json`,
`infra127-regression-matrix.json` e `infra127-summary.json`. As execuções
intermediárias incluem falhas reproduzidas e erros de desenvolvimento;
não são reclassificadas como aprovações.

Compileall e pip check passaram (`No broken requirements found`);
`git diff --check` passou. Os warnings permanecem nos logs: depreciações de
`datetime.utcnow`, frequência pandas `H` e concatenação de DataFrame vazio/NA.
Não foram suprimidos nem resolvidos alterando estratégia nesta fase.

PG18.6 usa imagem oficial local `postgres:18-alpine`, digest
`sha256:9ba53c7d41ea093baf5c1ba635ba5434c4f979880527d49bb7d9237d3193085b`.
O ambiente usa Docker VFS com disco limitado. A tentativa da imagem 16 atual
falhou por espaço; a imagem 16.15 chegou a ser obtida, mas restavam apenas
309,3 MiB antes de criar outro container, insuficientes para sua camada VFS,
dados e a guarda inalterada de 256 MiB livres. Nenhum teste/servidor PG16 foi
iniciado. Foi removida somente essa imagem recém-obtida e sem uso; o container
anterior `9fb2a800a365` e seu volume foram preservados. Não houve prune global,
remoção forçada ou acesso ao banco anterior. Evidência:
`infra127-pg16-unexecuted.json` e logs de recurso. Resultados antigos de PG16
não são declarados como execução desta fase.

Backup/restore: `infra127-proof-pg18.json` contém comparação de dados, índices,
sequência, hashes de inventário, quarentena, recibos e STOP; a regressão completa
inclui os cenários anteriores de UNKNOWN/recibo ausente ou corrompido.
O mesmo arquivo registra crash real com quarentena ativa, log de recovery e
STOP UNCONFIRMED durante indisponibilidade. Os cortes reais de TCP anteriores
estão em `infra127-final-regression-123-tcp-pg18.json`; a nova perda de ACK da
quarentena é explicitamente uma injeção Python após commit real observado.

Medições fechadas do runtime final, dez repetições por volume, máximos em ms:

| Eventos | STOP verificado | Lock financeiro | ACK | Reconciliação | Metas conjuntas |
|---:|---:|---:|---:|---:|---|
| 0 | 362,268 | 232,723 | 147,772 | 81,578 | atendidas nesta amostra |
| 1.000 | 845,673 | 647,122 | 169,794 | 415,861 | atendidas nesta amostra |
| 5.000 | 3.834,763 | 3.550,523 | 511,572 | 2.636,974 | **não atendidas** |

Nenhum timeout de STOP nesses três cenários normais. No cenário adversarial
do **protótipo rejeitado**, STOP persistiu em 5.049,140 ms e o candidato teve
`LockNotAvailable`; isso não é apresentado como êxito de desempenho nem como
timeout de STOP do runtime. O protótipo continua fora do produto. Fonte:
`infra127-final-regression-124-performance-after-pg18.json` e
`infra127-final-regression-125-blockers-pg18.json`.

Como referência interna, a primeira implementação desta mesma fase havia
medido STOP de 3.417,466 ms em 5.000 eventos, também fora da meta. São execuções
em ambiente compartilhado, sem promessa de melhoria ou isolamento de host.
Uma leitura intermediária de 3,56 s, comunicada durante atualização incremental
do arquivo, foi corrigida pelo resultado fechado de 3,83 s. Não se ajustou
timeout, volume, parâmetros ou validações para aprovar medições.
<!-- END_FINAL_RESULTS -->

Comandos reproduzíveis, com dependências de desenvolvimento já instaladas:

```text
python scripts/test_infra127_postgres.py --version 18 --alpine --junitxml=docs/retomada/evidencias/infra127-postgres18-release.xml
python scripts/test_infra127_postgres.py --version 18 --alpine --infra124-safety-only --junitxml=docs/retomada/evidencias/infra127-safety-final18.xml
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 BROKER_MODE=paper BROKER_PROVIDER=paper python -m pytest -q
python -m compileall -q core engines workers pages scripts tests
python -m pip check
git diff --check
```

PG16 é pendência, não comando executado: quando houver espaço seguro,
usar o mesmo runner com `--version 16 --alpine` e novo caminho de evidência.

O runner não aceita DSN/container remoto nem credenciais herdadas; cria banco,
usuário/dados fictícios e container próprios, publica somente 127.0.0.1 e limpa
seu próprio container/volume. Os testes negativos e destrutivos afetam somente
essas fixtures. Evidências são logs, JUnit e JSON de prefixo infra127; dumps,
senhas e DSNs não são incluídos. Artefatos regenerados de fases anteriores são
copiados para novos nomes e os originais restaurados byte a byte.
O check staged identificou espaços finais emitidos pelo pytest em tracebacks.
Somente esses espaços/tabs finais dos novos TXT/XML foram normalizados para
passar `git diff --check`; nenhuma linha, falha, assertion ou warning foi removido.
Hashes anteriores/posteriores estão em `infra127-format-normalization.json`.
Os 137 arquivos de evidência das fases anteriores não foram normalizados.

Execuções de desenvolvimento, sem mascaramento: a primeira offline teve uma
falha de compatibilidade do mock de UI com argumentos novos; o código agora
passa os argumentos apenas quando há quarentena. A primeira execução estendida
PG18 teve uma falha de bootstrap do resumo semanal em banco vazio; corrigida
pela unidade transacional do helper. Logs e assertions desses casos permanecem.

A suíte completa do runtime final encontrou duas expectativas antigas de
`PersistenceDataError` para JSON inválido em ordens/relatórios sem recibo.
O novo preflight lança `ReconciliationRequired` **antes** de criar intenção e
persiste quarentena. Esses dois casos foram convertidos: além da exceção
específica, exigem todas as linhas idênticas, zero recibos e contenção confirmada
por conexão independente, com motivo `INVALID_UNANCHORED_FINANCIAL_JSON`.
A variante de `paper_trades` mantém a exceção original. Não foi usado `except`
genérico nem removida a prova de rejeição. O runtime não foi alterado para essa
conversão; a reexecução abrange o módulo de segurança afetado e todos os casos
1.2.6/1.2.7. A execução completa com as duas falhas também é conservada.

## 10. Matriz de aceitação

| Regressão | Prova / resultado exigido |
|---|---|
| No-op | 3 repetições e job completo repetido, zero API de escrita, mesmos bytes/seq/timestamps/manifesto |
| Arquivo válido | PG sem recibos: objetos integrais e catálogo no mesmo commit; dados com recibos recusados sem efeitos |
| Duplicatas iguais | Originais conservados; mudança do ledger não é legitimada por dedupe |
| Conteúdo conflitante | Ambos os originais + origem/chave/hashes conservados e quarentena |
| Controle persistente | Conexão independente e processo Python novo observam bloqueio |
| Falha da retenção | Injeção após cópia: rollback de arquivo, runtime e catálogo |
| COMMIT incerto | Suíte anterior de resposta perdida TCP/rollback; nova injeção de ACK do controle rotulada separadamente |
| Restart/reconciliação | Falha de write/rollback externo não permite financiar no novo processo |
| STOP | Persistência/ACK durante bloqueio; sem confirmação falsa em falha |
| Backup/restore | Estado/linhas/seq/índices/controle comparados; destino STOPPED e bloqueado |
| Escritores concorrentes | Barreira determinística; dois recusados, nenhum trade repetido |
| Intenção antes do commit | Quarentena persistida na fronteira força rollback da unidade financeira |
| Recibos/manifestos | No-op idêntico; planned_trades adulterado não vira preimage de ciclo novo |
| A atrasado / B atual | Gate, quarentena e recibo B idênticos após diagnóstico/reconciliação A |
| JSON/corrupt control | JSON inválido, NaN, chave repetida, schema de controle inválido: sem sobrescrita e sem execução |
| START/reset/heartbeat | Não removem quarentena nem histórico de episódios |

O status executado de cada caso está nos JUnit/JSON; cenário pulado não é prova.

## 11. Riscos remanescentes e interrupção

- Arquivo de eventos assinados por recibos v1 requer protocolo futuro de
  transformação/custódia. Sua manutenção segue bloqueada nesta fase; arquivos
  financeiros legacy sem prova também impedem novos ciclos financeiros.
- Provas integrais e preflights continuam O(N), com trabalho sob lock global.
  Não há aprovação de metas STOP/ACK/72h; não se aumentou timeout para ocultá-las.
- SQL administrativo, grants amplos e binários antigos podem contornar guardas
  Python. Detectar o controle apagado não impede SQL arbitrário capaz de alterar
  simultaneamente todas as evidências. Não houve mudança de privilégios ou schema.
  Revisão independente e atualização coordenada continuam obrigatórias.
- Arquivos locais não têm atomicidade financeira multi-arquivo comprovada.
  A recuperação durável após falha de contenção foi homologada aqui em PG,
  não declarada equivalente para arquivos locais.
- A correção não repara dados antigos, não cria vínculos ausentes, não deduplica
  histórico e não oferece exactly-once universal. Evidência insuficiente bloqueia.
- Protótipo de snapshot antigo continua rejeitado; desconhece a nova superfície
  de quarentena e não é caminho operacional suportado.
- PostgreSQL 16 não foi homologado nesta execução: falta de espaço com Docker
  VFS, sem apagar contêiner/volume anterior. Não há equivalência entre versões
  declarada com base apenas nos resultados da fase anterior.

Interromper ensaio diante de UNKNOWN inconclusivo, controle ilegível, tentativa
de writer incompatível, divergência, perda/duplicação ou STOP não confirmado.
Não usar reset, DELETE, restauração antiga, timeout ou heartbeat para liberar.

## 12. Rollback e próxima fase

Rollback de binário não pode ignorar quarentena já persistida. Antes de retornar
a versão antiga: STOP verificável, encerrar writers de forma coordenada, backup
com comparação e revisão independente das barreiras. Não limpar o namespace novo
nem restaurar backup antigo sobre novos commits. Se não houver compatibilidade
segura, permanecer bloqueado e preparar correção adiante.

Atlas deve revisar origem/chave/hashes dos conflitos, a máquina de estados, a
liberação por geração e as evidências de falha/restart/restore. Próxima fase:
homologar o contrato de atualização de writers, manutenção de eventos com recibos
e contenção/desempenho sob carga, depois de auditoria independente.

## 13. Decisão e ausência de recuperação operacional

<!-- FINAL_DECISION -->
**BLOQUEADO.** As correções locais não representam liberação operacional.
Persistem a validação não executada no PostgreSQL 16, as metas conjuntas de
contenção/desempenho não demonstradas em carga maior e a ausência de protocolo
de custódia que permita arquivar eventos financeiros com recibos v1. Escritores
antigos/SQL externo e rollback de versão ainda exigem revisão independente.

Atlas deve revisar os diffs, os contraexemplos convertidos, a contenção sem
latch, os hashes de backup/restore, as provas reais TCP/crash e os resultados
completos antes de autorizar qualquer fase posterior. Nenhum desses limites
é resolvido por apagar evidência, ignorar quarentena ou repetir uma operação.
<!-- END_FINAL_DECISION -->

Qualquer decisão deste relatório se limita à correção e evidência local.
**Não autoriza recuperação de serviços, Web/Worker prolongados ou 72 horas.**
Não houve PR, merge, deploy, operação Railway, alteração de variável remota,
migração de produção, recurso pago ou ordem real. A publicação usa apenas a
branch autorizada pelo conector GitHub, com `[skip ci]`; não se declara segura
configuração externa de autodeploy que não foi inspecionada.
