# ATLAS — INFRA-1.2.7.1: fechamento de regressão

18/11/2026 UTC. PAPER_ONLY / SECURITY_FIRST / LOCAL_TESTS_ONLY /
COST_CONTROLLED / NO_PRODUCTION_DEPLOY.
Somente `oliveirasamuel03-jpg/robo-investimentos`.

## 1. Origem e rastreabilidade

- Branch de origem remota confirmada:
  `atlas/infra-1-2-7-retencao-quarentena`.
- Commit exato, local e remoto:
  `8bdb4bd2fe099ddc59af158f8a32f190fd6b84ab`.
- Nova branch: `atlas/infra-1-2-7-1-fechamento-regressao`, criada nesse commit,
  com árvore limpa em worktree separada do mesmo repositório.
- Runner e testes novos: [`69d277d`](https://github.com/oliveirasamuel03-jpg/robo-investimentos/commit/69d277def9966f7a20b76a8a352c39d5aadf47d9).
- O commit documental final é obtido por
  `git log -1 --format=%H -- docs/retomada/ATLAS_INFRA_1_2_7_1_FECHAMENTO_REGRESSAO.md`
  e identificado na entrega. Seu próprio hash não é inserido circularmente no conteúdo.

Foram comparados os **216 arquivos de evidência preexistentes** com os blobs
do commit exato. Os 215 hashes do manifesto anterior também foram verificados.
Nenhuma divergência inicial. Evidência: `evidencias/infra1271/source.json`.
Árvore original e outras worktrees não foram alteradas.

## 2. Auditoria independente

<!-- INDEPENDENT_AUDIT -->
**PENDENTE / NÃO FORNECIDA.** Não foi localizado parecer independente da
INFRA-1.2.7 nos arquivos da origem, nem na busca de issues restrita a este
repositório para auditoria e 1.2.7. A referência foi solicitada ao usuário
durante a execução. O relatório e o inventário de fontes produzidos pela
própria execução anterior não são tratados como revisão independente.

Os testes locais seguem autorizados pelo comando desta missão. Sua execução
não supre o parecer ausente nem constitui aprovação de segurança operacional.
<!-- END_INDEPENDENT_AUDIT -->

## 3. Causa da pendência anterior

A INFRA-1.2.7 teve uma execução completa do runtime final com **173 passed e
2 failed**: dois testes esperavam `PersistenceDataError`, mas o novo preflight
detectava JSON inválido antes da intenção, persistia contenção e lançava
`ReconciliationRequired`. As expectativas foram convertidas naquela fase e
reforçadas com preservação dos dados, ausência de novos recibos e quarentena
observada por conexão independente. A reexecução de impacto teve 48 passed.
Faltava uma única rodada completa verde de 175 casos do commit entregue.

Essas conversões já estão no commit exato autorizado. **Nesta fase nenhum
teste anterior, assertion, regra financeira ou código operacional foi alterado.**
Falhas e evidências anteriores são preservadas integralmente.

## 4. Auditoria e correção do runner

O runner anterior seleciona os 175 casos por padrão, mas aceita seleções
parciais para outras tarefas e não exige, sozinho, 175 IDs únicos com zero
falhas/erros/pulos. Também permite que funções de evidência dos testes regravem
arquivos de fases anteriores no checkout em uso.

Foi adicionado `scripts/test_infra1271_postgres.py`, sem modificar o runner
antigo nem os testes PostgreSQL:

1. Usa o commit exato autorizado em uma worktree temporária detached, do mesmo
   repositório, para executar o runner anterior sem filtros.
2. Mantém tabelas, DDL de fixtures, assertions, timeouts e 175 casos originais.
3. Confere o conjunto completo de IDs contra o manifesto do commit de origem.
   Usa somente as identidades desse manifesto, nunca os resultados antigos:
   a aprovação depende do JUnit gerado pela nova execução.
4. Recusa JUnit incompleto, IDs duplicados/inesperados, header inconsistente,
   casos pulados, falhas, erros, casos desmarcados ou saída não zero.
5. Confere a versão principal observada do servidor e BROKER_MODE/PROVIDER paper.
6. Grava log, JUnit, duração, digest da imagem e resultados em diretório novo,
   sem substituir uma rodada anterior.
7. Copia artefatos novos/alterados da worktree de teste para esse diretório,
   verifica os arquivos originais e remove somente sua própria cópia descartável.
   Se a captura de evidências falhar, mantém a cópia para investigação.

O banco continua sendo criado e limpo pelo runner original: container próprio,
um CPU, 512 MiB, bind somente em 127.0.0.1, dados/credenciais sintéticos,
ambiente sanitizado e proibição de DSN ou Docker remoto. Não foi instalado
PostgreSQL nem baixada imagem nesta fase.

A reserva de disco dos testes continua **256 MiB**, sem redução. O wrapper
exige adicionalmente espaço para a imagem/camada e 768 MiB para os dados do
container antes de iniciar. Não modifica parâmetros do PostgreSQL ou do robô.

Nove testes offline novos verificam a rejeição de evidência inválida/incompleta
e a cópia de artefatos sem sobrescrever a origem. Eles não substituem nenhum
caso anterior. O PostgreSQL continua com exatamente 175 casos.

## 5. Resultado integral PostgreSQL 18

<!-- PG18_RESULTS -->
**175 passed / 0 failed / 0 errors / 0 skipped / 0 deselected**, em uma única
rodada completa do commit exato de origem, sem seleção ou reexecução parcial.
Servidor observado **PostgreSQL 18.6**, imagem oficial local `postgres:18-alpine`,
digest `sha256:9ba53c7d41ea093baf5c1ba635ba5434c4f979880527d49bb7d9237d3193085b`.

| Medida | Resultado |
|---|---|
| Duração pytest registrada no log | 1.539,37 s — 25min39s |
| Duração total do wrapper, incluindo isolamento/limpeza | 1.550,44 s |
| Warnings | 640, preservados no log |
| IDs JUnit ausentes, inesperados ou duplicados | 0 |
| Casos ocultos/desmarcados/pulados | 0 |
| Artefatos novos/alterados capturados | 12 |
| Evidências anteriores preservadas byte a byte | 216 |
| Cópia descartável removida após captura | sim |

Evidências da rodada:

- [JUnit completo](evidencias/infra1271/pg18-20261118T141016Z-7d8e6818/junit.xml).
- [Log completo](evidencias/infra1271/pg18-20261118T141016Z-7d8e6818/pytest.txt).
- [Manifesto e verificação por ID](evidencias/infra1271/pg18-20261118T141016Z-7d8e6818/run.json).
- [Artefatos capturados](evidencias/infra1271/pg18-20261118T141016Z-7d8e6818/artifacts/infra127-proof-pg18.json),
  incluindo quarentena, STOP e backup/restore sintéticos. Os demais artefatos
  estão listados com hashes em `run.json`.

O código de saída foi zero. Não foi necessário corrigir ou reexecutar esta
rodada, e nenhum caso PostgreSQL foi alterado. Os casos de conexão TCP real,
crash/recovery, backup/restore, retenção, recibos, manifestos, concorrência e
PAPER foram executados na mesma sessão. Resultado positivo de teste funcional
não significa aprovação de desempenho ou segurança absoluta.

Somando a suíte offline e a sessão PG18: **718 casos distintos**, preservando
os 709 da origem e acrescentando somente nove testes offline do wrapper.
PG16 não é incluído como execução nem como aprovação nesse total.
<!-- END_PG18_RESULTS -->

## 6. PostgreSQL 16

**RESOURCE_BLOCKED — 0 executados; 175 não executados.** A imagem oficial PG16
não está presente, nem há servidor/binaries PG16 instalados no ambiente.
Somente `postgres:18-alpine` está disponível. A execução do wrapper para 16
encerrou antes de criar container/banco, com código 3 e registro explícito.

O Docker usa VFS. A fase anterior já demonstrou que extrair a imagem 16 deixou
apenas 309,3 MiB antes mesmo da camada do novo container e seus dados. O ambiente
continua com cerca de 2 GiB livres antes desta rodada; não há margem conservadora
para repetir download, camada VFS, dados e a reserva inalterada. Não foi feito
pull, prune, remoção de volume/imagem anterior, alteração do daemon ou contratação.

Evidências: `evidencias/infra1271/runner16.txt`, `pg16-resource.json` e o
`run.json` da tentativa PG16. Os resultados antigos de 16 não são recontados
como homologação atual. O container anterior `9fb2a800a365` e seu volume são
preservados sem inicialização ou leitura do banco.

## 7. Regressão offline e verificações

**543 passed, 175 skipped, 104 warnings**, em **98,36 s**.
Os pulados são os casos PostgreSQL, executados separadamente; nenhum pulo é
contabilizado como aprovação. Os 543 incluem todos os 534 casos offline da
origem e nove testes novos do wrapper. Evidência: `offline.txt` e `offline.xml`.

STOP, UNKNOWN_COMMIT, retenção, quarentena, recibos, manifestos, invariantes,
identidade de ciclos, concorrência e PAPER seguem na cobertura preservada.
Os testes reais TCP, crash/recovery e backup/restore pertencem à suíte PG175.

Compileall, pip check e diff check passaram; `No broken requirements found`.
Os warnings preexistentes de `datetime.utcnow`, resample pandas `H` e concatenação
de DataFrame vazio/NA permanecem visíveis. Não foram suprimidos.

Somente espaços/tabs finais emitidos nos novos TXT/XML foram normalizados para
o check de whitespace do Git. Nenhuma linha, warning, assertion ou resultado foi
removido; os hashes antes/depois estão em `format-normalization.json`. O JUnit
foi revalidado depois dessa formatação. As evidências anteriores não foram tocadas.

## 8. Arquivos modificados

Somente adições nesta fase:

| Caminho | Finalidade |
|---|---|
| `scripts/test_infra1271_postgres.py` | Wrapper de fechamento e isolamento da rodada completa |
| `tests/test_infra1271_runner.py` | Nove verificações offline do wrapper |
| `docs/retomada/ATLAS_INFRA_1_2_7_1_FECHAMENTO_REGRESSAO.md` | Este relatório |
| `docs/retomada/evidencias/infra1271/` | Fonte, recursos, logs, JUnit, resumos e artefatos novos |

Arquivos preexistentes, módulos operacionais, scripts antigos e todas as
assertions PostgreSQL permanecem idênticos à origem. Não há migração nova,
protocolo de custódia novo ou alteração de privilégios SQL da aplicação.
Os estudos SQL já existentes na suíte rodam apenas em fixtures descartáveis.

## 9. Reprodução

Com a branch de fechamento, dependências já instaladas e imagem oficial local:

```text
python scripts/test_infra1271_postgres.py --version 18 --alpine
python scripts/test_infra1271_postgres.py --version 16 --alpine
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 BROKER_MODE=paper BROKER_PROVIDER=paper python -m pytest -q --junitxml=offline.xml
python -m compileall -q core engines workers pages scripts tests
python -m pip check
git diff --check
```

Não usar DSN remoto, imagem externa arbitrária, filtros `-k`, flags de seleção,
retry financeiro ou credenciais reais. O wrapper não faz pull automático.
Cada invocação cria um diretório exclusivo sob `evidencias/infra1271/`;
`run.json` identifica origem, imagem, resultados e duração, e `junit.xml`
comprova os IDs. Falha na rodada exige investigação e nova rodada completa,
sem substituir ou ocultar a evidência negativa.

## 10. Riscos remanescentes e próxima correção mínima

Não houve trabalho de otimização nesta missão. Permanecem os resultados da
origem em 5.000 eventos: máximo de STOP **3.834,763 ms**, lock financeiro
**3.550,523 ms** e reconciliação **2.636,974 ms**, acima das metas conjuntas.
A execução completa conserva os benchmarks antigos como casos obrigatórios;
passar suas assertions funcionais não equivale a atingir metas de desempenho.
Nenhum timeout, histórico, capital, ticket ou frequência foi ajustado.

Persistem também: custódia/transformação de eventos vinculados a recibos v1,
escritores antigos, SQL privilegiado, quiescência de atualização e rollback de
versão que desconheça quarentena. Esses pontos não foram corrigidos nem
liberados por esta missão.

Próximo passo mínimo: concluir a mesma rodada integral no PG16 em ambiente
local gratuito com espaço suficiente, preservando volumes e reserva; obter o
parecer independente do commit validado. Depois, Atlas deve definir fase
explícita para as pendências de custódia/escritores e contenção. Não ampliar
automaticamente o escopo para alterações financeiras ou infraestrutura remota.

## 11. Decisão e limites

<!-- FINAL_DECISION -->
**APROVADO COM RESSALVAS — exclusivamente para o fechamento da regressão
local PG18/offline.** Foi encerrada a pendência de uma única rodada completa
verde de 175 casos, preservando todas as assertions e a evidência anterior.

As ressalvas são PG16 `RESOURCE_BLOCKED` e parecer independente ainda não
fornecido/localizado. Esta execução não substitui esse parecer nem declara
aceitação de segurança. Permanecem os bloqueios operacionais de desempenho,
custódia e contrato de escritores da INFRA-1.2.7, sem qualquer liberação.
<!-- END_FINAL_DECISION -->

Nenhuma decisão deste documento autoriza recuperação operacional ou 72 horas.
Nenhum PR, merge, deploy, Railway, banco/volume de produção, novo serviço pago,
worker operacional ou ordem real. Nenhuma quarentena operacional foi liberada.
Subprocessos finitos e controles sintéticos exercitados pelos testes pertencem
exclusivamente aos bancos descartáveis. Broker/provider e estratégia permanecem
os da origem; BROKER_MODE=paper e BROKER_PROVIDER=paper são impostos na rodada.
