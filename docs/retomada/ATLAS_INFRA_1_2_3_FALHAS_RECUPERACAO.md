# ATLAS — INFRA-1.2.3: falhas, recuperação e durabilidade

Data: 12/11/2026, UTC. **PAPER_ONLY / SECURITY_FIRST / COST_CONTROLLED /
LOCAL_TESTS_ONLY / NO_PRODUCTION_DEPLOY.**
Repositório exclusivo: `oliveirasamuel03-jpg/robo-investimentos`.

## 1. Resumo executivo

Foram implementados e exercitados testes de socket TCP real, encerramento
inesperado do PostgreSQL, recuperação, backup/restauração e ordenação de ciclos.
Uma falha de identidade em diagnósticos atrasados foi reproduzida e corrigida.
Saldo, posições, BUY/SELL, relatórios e bloqueios foram preservados nos cenários
locais comprovados. Reconciliação não reexecutou operações financeiras.

**A homologação prolongada permanece BLOQUEADA.** A carga de 5.000 eventos
históricos apresentou contenção excessiva e timeout de STOP. O tratamento foi
conservador: não anunciou persistência nem ACK. Testes que verificam esse
comportamento podem passar mesmo com o orçamento de desempenho reprovado.
Esta entrega não aprova 72 horas, Web/Worker contínuos, produção ou dinheiro real.

## 2. Versão, branch e commits auditados

- HEAD remoto consultado antes da criação da branch, por `git ls-remote`:
  `atlas/infra-1-2-2-commit-incerto` =
  `e62a3a2be39d70782ce07d50e59027fc2c32e886`, igual ao HEAD local limpo.
- Branch exclusiva: `atlas/infra-1-2-3-falhas-recuperacao`, criada dessa referência
  localmente e pelo conector GitHub. Sem main, merge ou PR.
- Implementação e testes: [`1a6c67e9b719536102f9794736daa1a93f9177c8`](https://github.com/oliveirasamuel03-jpg/robo-investimentos/commit/1a6c67e9b719536102f9794736daa1a93f9177c8).
- O commit final é o que introduz este relatório e as evidências. Identificável por
  `git log -1 --format='%H %s' -- docs/retomada/ATLAS_INFRA_1_2_3_FALHAS_RECUPERACAO.md`;
  o SHA e o link concretos também acompanham a entrega final.
- Relatórios INFRA-1.2.1/1.2.2 e proposta de idempotência revisados. Sem migração
  nova de schema ou unicidade retroativa aplicada nesta fase.

## 3. Falhas originais reproduzidas

Depois de A e B confirmados, `post_commit_failure(A)` podia reabrir a barreira com
`active_cycle_id=A`, embora `last_cycle_id=B`. A reconciliação explícita de A
comparava seu manifesto antigo ao estado mais recente sem qualificá-lo como
histórico. Execução dos testes novos contra o código original: **2 falharam,
1 passou**, antes de qualquer correção.
[Evidência da regressão original](evidencias/infra123-identity-before.txt).

Durante a construção do procedimento de backup houve **8 falhas e 74 aprovações**
no primeiro ensaio expandido: o helper supunha incorretamente o nome da sequência.
Foi corrigido para descobrir a sequência realmente vinculada a `app_rows.seq`
pelo catálogo PostgreSQL. Assertions de igualdade/avanço foram mantidas.
[Registro do defeito do helper](evidencias/infra123-backup-helper-before.txt).
O arquivo preserva extrato e hash do log temporário completo; não apresenta esses
ensaios como aprovações finais.

## 4. Falhas reais de comunicação comprovadas

Proxy PostgreSQL v3 de teste, cliente e destino exclusivamente `127.0.0.1`.
Fecha sockets reais com SO_LINGER; não cria a exceção da aplicação manualmente.
Dados de autenticação são encaminhados opacamente, nunca gravados. SSL/GSS são
desabilitados apenas nessa conexão descartável para observar o protocolo.

| Cenário | Evidência independente no servidor | Resultado da aplicação / reconciliação explícita |
|---|---|---|
| Corte antes de enviar COMMIT | Recibo PREPARED, nenhum trade | UNKNOWN_COMMIT; depois NOT_COMMITTED comprovado pela preimagem |
| Corte durante transmissão | Escreve 7 dos 12 bytes do frame no socket e interrompe; servidor mantém PREPARED, nenhum trade | UNKNOWN_COMMIT; depois NOT_COMMITTED |
| COMMIT recebido, resposta retida | Proxy observa CommandComplete COMMIT; outra conexão consulta recibo COMMITTED e um trade | UNKNOWN_COMMIT; depois COMMITTED, sem repetir BUY |
| Desconexão após ACK conhecido pelo driver | Um trade e resultado positivo já conhecidos | COMMITTED preservado; conexão seguinte recupera leitura |
| Diagnóstico após commit financeiro | Conexão diagnóstica é cortada e SELECT falha de fato no libpq | POST_COMMIT_DIAGNOSTIC_FAILURE; reconciliação sem novo trade |

Não se afirma que os 7 bytes tenham chegado integralmente ao servidor antes do
RST; a prova é de transmissão parcial tentada e ausência do commit financeiro
observada separadamente. Não é uma falha de WAN, firewall ou serviço externo.

Nos três primeiros cenários o ciclo roda em subprocesso limitado a uma execução.
Após receber UNKNOWN ele termina com `os._exit(79)`, sem cleanup Python. Outro
intérprete, sem memória do anterior, encontra a barreira e rejeita novo ciclo.
A falha financeira vem do socket, e a terminação do processo é provocada pelo
harness. Nenhum worker contínuo é iniciado.

Evidências: [TCP PG18](evidencias/infra123-tcp-pg18.json),
[TCP PG16](evidencias/infra123-tcp-pg16.json).

## 5. Cenários apenas simulados e limites da reprodução

As injeções Python das fases anteriores foram preservadas e reexecutadas, incluindo
falha de BODY, perda simulada de resposta e falha de relatório/apresentação.
Continuam sendo injeções Python, não interrupções reais de rede. O teste legado
de INFRA-1.2.2 continua extraindo a versão antiga em diretório temporário próprio.

Mercado, sinais e posições são fixtures sintéticas determinísticas; chamadas de
feed reais não são necessárias. Events, IPC e consultas de `pg_locks` ordenam os
pontos de falha. Esperas curtas servem apenas a polling de readiness/lock; não
substituem evidência de concorrência. Não se exige administração do anfitrião.

## 6. Correções, arquitetura e mapa de concorrência

| Etapa | Proteção e alcance |
|---|---|
| Aquisição de mercado | Fora da transação financeira; nenhum cálculo de estratégia foi alterado |
| Intenção | PREPARED e gate duráveis antes de tentar o commit financeiro |
| Confirmação | Lock transacional `atlas:persistence-unit`, revalidação STOP/PAPER/risco/CAS; saldo, posições, trades, relatório e recibo no mesmo commit |
| Reconciliação atual | Mesmo lock; recibo, plano e manifesto canônico devem concordar; falha de leitura mantém bloqueio |
| Diagnóstico atrasado A | Compara identidade atual sob lock; escreve somente auditoria no recibo A se B já o sucedeu; não muda gate/recibo B nem financeiro |
| Reconciliação histórica A | Só aceita resolução anterior explicitamente verificada; devolve `historical=True`, sem comparar com financeiro de B ou liberar B |
| UI histórica | Mensagem informativa distingue evidência anterior de reconciliação do estado atual; não exibe sucesso financeiro atual |
| Backup | STOP verificado na origem, fence global de escritores, snapshot exportado read-only, dump consistente |
| Restore | Banco novo isolado; comparação exata antes da quarentena STOP; não executar worker nem START |

A identidade passa a prevalecer sobre a ordem de chegada de um diagnóstico.
Os testes com Events atrasam A até B confirmado ou B com commit incerto. Conferem
igualdade integral de gate B, recibo B, manifesto e contagem financeira antes e
depois; reconciliar A não libera B. A reconciliação histórica retorna a resolução
**anteriormente** registrada, não uma nova afirmação sobre o saldo atual.
[Evidência PG18](evidencias/infra123-ordering-pg18.json) e
[PG16](evidencias/infra123-ordering-pg16.json).

Não há segunda fonte de verdade, novo schema, retry financeiro, novo worker,
limpeza automática de recibos ou modificação de critérios de trading.

## 7. PostgreSQL 18 e 16: execução e recuperação

| Execução final | Total | Aprovados | Falhas/erros | Skips | Warnings capturados | Tempo |
|---|---:|---:|---:|---:|---:|---:|
| Offline | 600 | 518 | 0 | 82 | 105 | 58.71 s |
| PostgreSQL 18.6 Debian | 82 | 82 | 0 | 0 | 103 | 181.08 s |
| PostgreSQL 16.15 Alpine | 82 | 82 | 0 | 0 | 103 | 170.14 s |

Logs/JUnit: [offline](evidencias/infra123-offline.txt), [XML](evidencias/infra123-offline.xml);
[PG18](evidencias/infra123-postgres18.txt), [XML](evidencias/infra123-postgres18.xml);
[PG16](evidencias/infra123-postgres16.txt), [XML](evidencias/infra123-postgres16.xml).

Os 500 testes offline e 66 PostgreSQL anteriores foram preservados. Há 18 casos
offline novos e 16 PostgreSQL novos: **600 casos distintos**, com os 82 casos PG
repetidos em duas versões. Os 82 skips da execução offline são casos não
executados naquela rodada; foram executados explicitamente nos dois servidores.
Warnings não foram suprimidos: depreciações existentes, especialmente
`datetime.utcnow()` em estado/controles. Logs e XML contêm os avisos capturados pelo pytest pai. Processos filhos seguem
os filtros padrão Python; essa contagem não é prova de ausência universal de avisos.
Compileall, pip check e diff --check terminaram com código zero.

Containers próprios: 1 CPU, 512 MiB, porta publicada apenas em loopback, senha
efêmera, dados fictícios. Volume anônimo substitui tmpfs para ensaiar restart do
mesmo cluster com a mesma porta. O runner remove só seu container e volume;
fixture remove apenas DBs novos dos prefixos sintéticos, dentro desse cluster.
Não se usa prune, volume antigo, credencial real ou DATABASE_URL recebida.

O processo PostgreSQL do container é morto por SIGKILL e reiniciado de fato.
`fsync=on`, `synchronous_commit=on`; logs mostram interrupção, redo e retomada de
conexões. Manifesto e trades são iguais após recuperação; STOP e UNKNOWN mantêm
as mesmas identidades; escrita financeira bloqueada continua rejeitada. STOP
solicitado durante indisponibilidade retorna UNCONFIRMED. Uma reconciliação
explícita comprova o commit anterior, sem operar, e STOP continua vigente.
[Recuperação PG18](evidencias/infra123-recovery-pg18.json),
[PG16](evidencias/infra123-recovery-pg16.json).

Isso comprova recuperação de processo/cluster local com o volume retido. Não
comprova sobrevivência à perda física do disco, energia ou do anfitrião.

## 8. Backup e restore: procedimento e provas

Procedimento em `scripts/local_synthetic_backup.py`, usado pelos testes. Recusa
DSN fora de 127.0.0.1, usuário/banco não sintético, container não pertencente à
invocação e porta diferente da publicada. Não é ferramenta para banco remoto.

1. Persistir e reler STOP na origem. Se falhar, abortar o procedimento. Isso
   impede que uma intenção PREPARED copiada ainda faça commit na origem depois
   do snapshot por um escritor cooperante. Não reiniciar a origem em RUNNING.
2. Adquirir advisory lock global de sessão; exportar snapshot REPEATABLE READ,
   READ ONLY; conferir novamente STOP e presença do namespace de controle.
3. Inventariar todas as linhas das duas tabelas, timestamps, sequência vinculada
   e definições de índices; executar `pg_dump -Fc --snapshot` no container próprio.
4. Criar banco de destino novo com nome aleatório. Confirmar vazio; executar
   `pg_restore --single-transaction --exit-on-error`, sem clean/drop de destino.
5. Comparar inventário completo e hashes antes de qualquer alteração de quarentena.
   Exigir igualdade; então solicitar e reler STOP de quarentena, preservando o
   bloqueio UNKNOWN e a identidade da intenção. Só retornar o destino se tudo passar.

O dump fica em diretório temporário privado, modo 0600; não é publicado, contém
somente fixtures e é removido com o teste. Evidências guardam hashes e métricas,
sem DSN/senha/hash de senha/dados pessoais. Sequência e índices são comparados;
uma inserção sintética não financeira confirma avanço da sequência no restore.

Cinco cenários por versão: sem intenção pendente; UNKNOWN; STOP com ACK; recibo
pendente ausente; recibo pendente com JSON inválido. Cada fixture tem 3 trades
(BUY/SELL/BUY), 1 relatório fechado, 1 posição aberta, histórico, recibos e usuário
sintético com senha hash. Dados completos são iguais antes da quarentena.
UNKNOWN válido só se resolve explicitamente com prova; ausência/corrupção do
recibo mantém UNKNOWN e impede escrita. Nenhum restore inicia operação.
[Evidências PG18](evidencias/infra123-backup-pg18.json),
[PG16](evidencias/infra123-backup-pg16.json).

O procedimento depende de origem parada, escritores cooperantes e destino sem
consumidores até a quarentena. Não autoriza copiar um snapshot antigo, permitir
novos commits na origem e depois promover esse restore sem nova análise. Não
há recuperação do banco antigo: ele não foi acessado nem alterado.

## 9. Integridade financeira e PAPER

Invariantes do contrato existente permanecem cobertas: números finitos, saldo e
patrimônio coerentes, posição por ativo, referências BUY/SELL e relatórios,
manifesto de estado/histórico e atomicidade das unidades PostgreSQL. Os novos
ensaios acrescentam comparação exata após falha/restore e ausência de duplicação
nas reconciliações. BUY e SELL podem compartilhar trade_id da posição;
`UNIQUE(trade_id)` isolado continuaria incorreto.

`BROKER_MODE=paper` e `BROKER_PROVIDER=paper` são impostos pelo runner e conferidos
nas fixtures. Rede Python do ensaio PG é limitada a 127.0.0.1; a suíte offline
bloqueia tentativas de rede/processos externos. Download gratuito das imagens
oficiais é preparação, não endpoint da aplicação em teste. Nenhuma ordem real,
provider real consultado ou flag experimental ativada. Score, thresholds,
min_signal_score, capital, ticket, max_open_positions, provider e estratégia do
produto não foram alterados. SHADOW/STUDY/DIAGNOSTIC/PLANNING continuam sem
novo poder operacional; sem avanço para 2.6C.

## 10. STOP

STOP e barreira financeira são controles distintos. UNKNOWN não impede o operador
de solicitar STOP. Confirmação exige persistência e leitura verificável; ACK exige
reconhecimento. Banco indisponível/timeout não é sucesso. Reinício e restore não
reativam o bot. Os testes anteriores de START/reset/heartbeat/configuração,
conflito CAS, feed lento e rollback continuam presentes.

Nos ensaios de desempenho, STOP é solicitado enquanto a transação financeira já
tem o lock; `pg_locks` comprova o waiter antes da liberação controlada. Um commit
já ordenado antes do STOP pode concluir; não se promete cancelamento retroativo.
Quando houve timeout, o recibo retornou não confirmado e nenhum ACK foi alegado.
O timeout observado está preservado em
[evidência própria](evidencias/infra123-stop-timeout-pg16.json) e
[log da rodada](evidencias/infra123-postgres16-stop-timeout.txt), mesmo se outra
rodada tiver latência menor. O backup subsequente solicita STOP novamente de forma explícita para sua própria
quarentena; não converte a primeira solicitação falha em aprovação retroativa.

## 11. UNKNOWN_COMMIT e limites de idempotência

Falha de socket na tentativa de COMMIT fica UNKNOWN mesmo nos casos em que a
conexão independente descobre rollback. A aplicação não faz essa inferência sem
reconciliação explícita sob fence. Reinício, heartbeat, START/reset e restore não
liberam a intenção. Falha de leitura mantém execução financeira bloqueada.

Recibo e manifesto válidos permitem COMMITTED ou NOT_COMMITTED, sem repetir
BUY/SELL. Recibo ausente/corrompido, identidade diferente e divergência financeira
mantêm UNKNOWN. POST_COMMIT_DIAGNOSTIC_FAILURE preserva confirmação conhecida.
Não existe liberação por tempo, deduplicação de histórico ou retry financeiro.

Não há exatamente-uma-vez universal: inserções SQL privilegiadas, escritores
antigos/não cooperantes, reutilização arbitrária de intenção e serviços externos
não são cobertos. A proposta de índice/chave de idempotência continua separada.

## 12. Desempenho, contenção e crescimento

| PG | Eventos históricos | COMMIT ACK máx. ms | Lock máx. ms | STOP persistido / retorno ms | ACK ms | Reconciliação ms | Proposta |
|---|---:|---:|---:|---:|---:|---:|---|
| 18 | 0 | 0.95 | 193.2 | 226.4 / 266.4 | 100.4 | 39.8 | Amostra dentro |
| 18 | 1000 | 1.00 | 943.0 | 913.2 / 1025.0 | 107.7 | 307.7 | Amostra dentro |
| 18 | 5000 | 1.47 | 4088.7 | 3710.5 / 3836.6 | 108.8 | 1530.0 | BLOQUEADO |
| 16 | 0 | 2.67 | 287.1 | 328.7 / 393.6 | 107.4 | 36.5 | Amostra dentro |
| 16 | 1000 | 1.01 | 1012.9 | 1049.3 / 1145.7 | 112.6 | 343.5 | Amostra dentro |
| 16 | 5000 | 1.49 | 3863.1 | 3568.8 / 3686.8 | 104.3 | 1383.1 | BLOQUEADO |

Métricas completas: [PG18](evidencias/infra123-performance-pg18.json) e
[PG16](evidencias/infra123-performance-pg16.json). O campo STOP mostra tempo
até confirmação do COMMIT / tempo até retorno com releitura verificada.
Em rodada anterior, PG16/5.000 atingiu 5.064,6 ms sem persistência confirmada;
lock financeiro chegou a 6.999,3 ms. Essa prova negativa foi preservada à parte.

Tempos medidos com `perf_counter`. Cada volume tem três commits financeiros e uma
amostra de STOP concorrente por versão; não são percentis nem SLA de 72 horas.
O campo lock é duração observada pelo cliente até o ACK, um limite superior ao
lock no servidor; inclui o pequeno arranjo controlado registrado no JSON.
Tracemalloc está ativo no ensaio e adiciona custo. RSS é high-water cumulativo do
processo de testes, não consumo isolado do protocolo ou pico do servidor.

Pico Python rastreado: 1.57–10.88 MiB; RSS high-water: 200.37–236.00 MiB.
Os recibos cresceram de 1 para 3: 1.613–1.631 bytes para 4.729–4.783 bytes
somados, sem limpeza. Manifesto posterior: 503–512 bytes; seu tamanho compacto
não elimina a leitura/hash de todo o histórico. Dump sintético dos volumes:
20.474–138.373 bytes;
dump 0.163–0.272 s,
restore 0.102–0.162 s,
fence de backup 0.181–0.322 s.
Histórico com 5.000 eventos PAPER também contém 5.000 ordens e 2.500 relatórios
sintéticos; a contagem da tabela não se limita aos eventos PAPER.

Limites de execução: até 5.000 eventos financeiros históricos, 90 s por caso,
(janela medida, após gerar a fixture), 30 s para cada dump/restore,
128 MiB de alocação Python rastreada, mínimo de
256 MiB livres antes do caso, container 512 MiB/1 CPU. Não são metas operacionais.

Proposta conservadora, ainda sujeita ao Atlas: STOP verificado <=2 s; ACK <=1 s;
reconciliação <=2 s; lock financeiro observado <=1,5 s. **5.000 eventos ficam
BLOQUEADOS** pelos resultados medidos. Até 1.000 eventos é apenas teto inicial
candidato para novos ensaios curtos, não autorização: exige repetição sem timeout,
amostras suficientes, folga de recursos e projeção de crescimento para 72 horas.
Não houve mudança de capital, frequência, limites ou retenção para caber no teste.

## 13. Riscos residuais e severidade

| Severidade | Risco e situação |
|---|---|
| ALTA / bloqueio | Varredura/hash/validação de histórico sob fence cresce com o volume; STOP pode atingir lock_timeout. Carga de 5.000 não homologada. |
| ALTA / pendente | Backend local em arquivos não oferece atomicidade financeira multi-arquivo; não usar essas evidências como garantia desse backend. |
| ALTA / pendente | Origem que volte a operar após o snapshot torna promoção de backup antigo um problema de fork/época; procedimento atual exige quiescência, isolamento e revisão explícita. |
| ALTA / fora da prova | Escritores sem protocolo ou SQL privilegiado podem violar barreiras/identidades; não misturar versões. |
| MÉDIA / pendente | Acúmulo longo de recibos e repetição de verificações não foi ensaiado por 72h; apenas três recibos por volume. Não há limpeza automática. |
| MÉDIA / pendente | Queda de energia, disco cheio/corrompido, OOM, failover, TLS/proxy remoto e computador alvo não validados. |
| Corrigida | Diagnóstico antigo reabrindo gate anterior; recibo antigo tratado como evidência atual. Testes offline e PG cobrem A/B. |

## 14. Limites não validados e testes não executados

Não executados: 72h, Web/Worker contínuos, rede WAN ou criptografada, failover/réplica,
falha física de host/disco/energia, ENOSPC/OOM, múltiplos workers, banco antigo,
backup remoto, produção, computador do usuário e outros sistemas operacionais.
Não houve alteração de firewall, exigência de privilégio do anfitrião ou serviço
pago. Os ensaios de socket e SIGKILL locais estão EXECUTADOS; os anteriores com
injeção Python continuam identificados como SIMULADOS.

## 15. Testes, arquivos alterados e reprodução

Com dependências de desenvolvimento e Docker disponíveis, sem `.env` e sem DSN
externa:

```bash
.venv/bin/python scripts/test_offline.py -q
.venv/bin/python scripts/test_infra12_postgres.py --version 18
.venv/bin/python scripts/test_infra12_postgres.py --version 16 --alpine
.venv/bin/python -m compileall -q core engines workers pages scripts tests paper_trading_engine.py
.venv/bin/python -m pip check
git diff --check
```

Os runners não recebem DSN externa e geram a senha efêmera; imagens oficiais
precisam estar disponíveis no Docker. Executar versões sequencialmente para
limitar disco. Nenhum container externo deve ser parado/removido. O procedimento
de backup é exercitado automaticamente pelos cinco cenários e três volumes.

Arquivos de código alterados/adicionados:

- `core/commit_protocol.py`: identidade de diagnóstico/reconciliação históricos.
- `core/ui_mutations.py`: feedback histórico sem sucesso financeiro atual.
- `scripts/test_infra12_postgres.py`: volume próprio, porta estável, guarda de rede e suíte ampliada.
- `scripts/local_synthetic_backup.py`: snapshot, dump/restore, comparação e STOP de quarentena.
- `tests/conftest.py`: coleta de bancos sintéticos novos apenas do runner isolado.
- `tests/infra123_tcp_proxy.py`: proxy limitado a TCP loopback, sem logging de pacotes.
- `tests/test_infra123_identity.py`: regressões de identidade, UI e guardas locais.
- `tests/test_infra123_postgres.py`: TCP, processo, SIGKILL, restore, ordenação e medições.

Este relatório e arquivos `evidencias/infra123-*` registram logs, JUnit, mapas,
medições e checks. O resumo legível por máquina é
[infra123-summary.json](evidencias/infra123-summary.json).
[Verificações técnicas](evidencias/infra123-checks.json).
As engines, worker, parâmetros e telas da estratégia não tiveram código alterado;
o helper compartilhado de UI é a única mudança na apresentação.

## 16. Critérios de interrupção e rollback

Interromper ensaio diante de UNKNOWN, divergência, STOP não confirmado, timeout,
OOM/disco insuficiente, perda de recibo, novo ciclo após barreira ou qualquer
configuração diferente de PAPER. Preservar banco, recibos, logs e identidades.
Não repetir BUY/SELL, resetar gate, apagar histórico ou usar timeout como liberação.
STOP falho exige comunicação clara e contenção; não assumir parada efetiva.

Rollback de código somente em branch própria com escritores parados e revisão
explícita, preservando o banco e bloqueios. A base reintroduz a falha de identidade;
não a reativar automaticamente. Não há DDL nova para desfazer. Backup não serve
como mecanismo de apagar um commit já feito ou resolver UNKNOWN por conveniência.

## 17. Condições necessárias para futura homologação de 72 horas

Revisão independente Atlas do diff e evidências; resolver contenção observada ou
estabelecer teto com margem e projeção demonstradas; repetir medições e STOP com
histórico/recibos crescentes; validar disco/backup no ambiente alvo sem dados reais;
procedimento de reconciliação e interrupção aprovado; versões cooperantes únicas;
PAPER verificado; origem/restore isolados, STOP/UNKNOWN intactos e monitoramento
com limites acordados. Qualquer cenário sem prova continua PENDENTE/BLOQUEADO.
A decisão de autorizar 72 horas pertence ao Atlas e não é concedida aqui.

## 18. Próxima fase recomendada e ausência de deploy

Priorizar redução demonstrável do custo de verificação sob lock, com regressões
de identidade, STOP, atomicidade e manifesto antes de qualquer mudança. Não
remover comparações nem transformar diagnóstico em gatilho para melhorar números.
Depois, homologação curta repetida com carga e ensaio de crescimento de recibos;
somente então solicitar revisão Atlas para futura janela prolongada local PAPER.

Nenhum comando de deploy, Railway, PR, merge, operação real ou banco remoto foi
executado. Commits exclusivamente na branch autorizada, via conector GitHub,
com `[skip ci]`; não houve contorno por gh CLI. Não há workflows `.github` na
árvore auditada. A configuração externa de autodeploy não foi alterada nem
certificada; isso não autoriza PR, deploy ou promoção da branch. Nenhum serviço
pago, crédito ou custo externo novo contratado.
