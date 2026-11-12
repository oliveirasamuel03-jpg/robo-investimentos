# ATLAS — INFRA-1.2.4: contenção, STOP e desempenho

12/11/2026, UTC. **PAPER_ONLY / SECURITY_FIRST / COST_CONTROLLED /
LOCAL_TESTS_ONLY / NO_PRODUCTION_DEPLOY.**
Somente `oliveirasamuel03-jpg/robo-investimentos`.

## 1. Referência e resultado

HEAD remoto consultado antes de agir: `atlas/infra-1-2-3-falhas-recuperacao` =
`8dc5e5a83bb9f30aba8941dbb00ae3e757fe36f9`, idêntico ao HEAD local limpo.
Branch exclusiva criada localmente e pelo conector GitHub:
`atlas/infra-1-2-4-contencao-stop`.
Implementação/testes: [`8a32328ec1fd32582ed253b3c1110ad586ab8435`](https://github.com/oliveirasamuel03-jpg/robo-investimentos/commit/8a32328ec1fd32582ed253b3c1110ad586ab8435).
O commit final introduz este relatório/evidências, identificável por
`git log -1 --format='%H %s' -- docs/retomada/ATLAS_INFRA_1_2_4_CONTENCAO_STOP.md`;
o SHA concreto também acompanha a entrega final.

Correções implementadas e regressão funcional aprovada nos cenários executados. **Homologação de carga prolongada permanece BLOQUEADA:** há máximos fora das metas candidatas, apesar da redução da mediana e do custo da sincronização.

Esta entrega **não autoriza 72 horas**, Web/Worker contínuos, implantação ou
ordens reais. A decisão de homologação prolongada pertence à revisão Atlas.

## 2. Gargalos identificados

Auditados commit_protocol, persistence, stop_control, PAPER/trader engines,
retention e worker; relatórios e evidências da INFRA-1.2.3 revisados.
[Mapa inicial](evidencias/infra124-audit-before.md).

- Sincronização de ordens: leitura de todas as ordens, materialização em DataFrame,
  DELETE e INSERT de todo o histórico a cada ciclo, inclusive sem linhas novas.
  O custo sob lock crescia com N comandos SQL; seq/timestamp e metadados antigos
  eram reescritos desnecessariamente.
- Reconciliação: manifesto decodificava/validava históricos integrais e logo
  depois os mesmos históricos eram relidos/decodificados para verificar o ciclo.
- Verificação de duplicatas: varreduras de identidades mesmo sem candidato de
  trade/relatório. Conjunto vazio não pode duplicar identidade; isso não dispensa
  validação integral do JSON e do histórico.
- Manifesto e diagnóstico continuam O(N), assim como a leitura de ordens para
  obter as chaves existentes. Nenhuma otimização baseada somente em contagem,
  MAX(seq), cache entre transações ou hash incremental não comprovado foi aceita.

Reprodução contra produto da base, PostgreSQL 16: **2 falharam, 6 passaram,
3 deselecionados**. Falhas comprovam reescrita das linhas e releitura da
reconciliação. [Log anterior](evidencias/infra124-reproduction-before-pg16.txt).
Não houve remoção/redução de assertions antigas.

## 3. Medições anteriores

A INFRA-1.2.3 mediu STOP de aproximadamente 3,7–3,8 s em suas rodadas finais e
um timeout de 5,06 s em outra rodada, com 5.000 eventos. Essas evidências originais
foram preservadas.

Nesta fase executou-se **antes das alterações de produto** o benchmark novo,
com dez repetições por volume e versão. Os cenários usam histórico de 0, 1.000
ou 5.000 eventos PAPER, mais ordens e relatórios correspondentes. Mesmos dados,
recursos, instrumentação principal e protocolo de sincronização antes/depois.
As medições adicionais do total dentro/fora do lock foram acrescentadas no ensaio
posterior; o custo dos wrappers de profiling está incluído em ambas as versões.

[Base PG18](evidencias/infra124-performance-before-pg18.json),
[base PG16](evidencias/infra124-performance-before-pg16.json);
[logs PG18](evidencias/infra124-before-pg18.txt),
[PG16](evidencias/infra124-before-pg16.txt).

## 4. Correções incrementais implementadas

1. No PostgreSQL, sincronização acrescenta somente ordens ausentes. Mantém o
   contrato de identidade (timestamp, ativo, lado), o limite de leitura PAPER
   existente e a validação estrita de todas as ordens. Preserva os bytes das
   linhas existentes, seq, created_at e metadados. Uma sincronização sem novidade
   não escreve ordens. Backend CSV mantém o caminho existente.
2. `manifest(_observed_rows=...)` pode entregar as linhas que acabou de consultar
   e decodificar para a mesma chamada de reconciliação sob o mesmo fence. A
   reconciliação usa essas observações, sem nova leitura redundante. A chamada
   seguinte sempre consulta/valida/hash novamente. Não há cache persistido ou
   compartilhado entre ciclos, nem manifesto fornecido pelo chamador como prova.
3. Verifica-se colisão com histórico somente quando há identidades candidatas.
   Havendo BUY, SELL ou relatório, a checagem correspondente continua integral.
   Os manifestos obrigatórios em begin_intent, record_unit_result e reconcile
   continuam validando **todos** os históricos, inclusive para plano vazio.

4. Save PostgreSQL valida o documento atual uma única vez antes do CAS e entrega
   esse mesmo documento às guardas financeiras e de STOP, que só o leem. Elimina
   duas decodificações redundantes, mantendo todas as verificações. Teste com
   token CAS correspondente a JSON inválido exige recusa por corrupção, não
   permite sobrescrita por coincidência do token.

A primeira rodada incremental passou nos 96 testes PG18, porém ainda excedeu
metas em 5.000 eventos (STOP 2,083 s; lock 1,943 s). Essa evidência foi preservada
em [JSON](evidencias/infra124-performance-increment1-pg18.json) e
[log](evidencias/infra124-postgres18-increment1.txt). A eliminação das decodificações
repetidas de estado foi feita depois dessa medição, seguida de nova regressão.

Nenhum schema, índice ou coluna novos; nenhuma migração local/remota necessária.
Nenhum timeout foi aumentado: lock_timeout permanece 5 s. Saldo, posições,
trades, ordens, relatório e recibo continuam na mesma transação financeira.

## 5. Medições posteriores e comparação PG18/16

| PG | Eventos | Versão | n | STOP ms med/P95/máx | Lock ms med/P95/máx | ACK ms med/P95/máx | Reconciliação ms med/P95/máx | Metas |
|---|---:|---|---:|---|---|---|---|---|
| 18 | 0 | Antes | 10 | 280.5/297.2/297.2 | 200.2/208.2/208.2 | 103.9/118.7/118.7 | 32.9/37.1/37.1 | Dentro na amostra |
| 18 | 0 | Depois | 10 | 248.3/276.4/276.4 | 167.6/201.3/201.3 | 102.2/156.1/156.1 | 37.9/73.2/73.2 | Dentro na amostra |
| 18 | 1000 | Antes | 10 | 1056.4/1145.6/1145.6 | 934.9/992.4/992.4 | 99.5/119.2/119.2 | 320.4/369.0/369.0 | Dentro na amostra |
| 18 | 1000 | Depois | 10 | 595.3/655.6/655.6 | 452.2/506.9/506.9 | 101.7/107.1/107.1 | 268.1/294.3/294.3 | Dentro na amostra |
| 18 | 5000 | Antes | 10 | 3959.1/4347.6/4347.6 | 3830.7/4238.5/4238.5 | 112.2/226.4/226.4 | 1593.7/1674.3/1674.3 | BLOQUEADO |
| 18 | 5000 | Depois | 10 | 1763.9/3564.3/3564.3 | 1641.7/3420.2/3420.2 | 109.8/160.5/160.5 | 1312.3/1870.2/1870.2 | BLOQUEADO |
| 16 | 0 | Antes | 10 | 296.9/358.4/358.4 | 208.2/265.3/265.3 | 109.9/123.3/123.3 | 39.7/46.1/46.1 | Dentro na amostra |
| 16 | 0 | Depois | 10 | 262.4/301.1/301.1 | 180.9/197.5/197.5 | 104.2/124.7/124.7 | 36.5/58.1/58.1 | Dentro na amostra |
| 16 | 1000 | Antes | 10 | 1050.5/1122.7/1122.7 | 896.3/948.5/948.5 | 105.3/126.8/126.8 | 335.3/394.1/394.1 | Dentro na amostra |
| 16 | 1000 | Depois | 10 | 595.0/742.2/742.2 | 471.5/559.1/559.1 | 104.5/221.4/221.4 | 274.2/404.9/404.9 | Dentro na amostra |
| 16 | 5000 | Antes | 10 | 4155.8/5157.0/5157.0 | 4013.6/5023.9/5023.9 | 114.2/173.5/173.5 | 1612.9/1961.4/1961.4 | BLOQUEADO |
| 16 | 5000 | Depois | 10 | 1714.2/1994.7/1994.7 | 1604.4/1849.8/1849.8 | 108.5/119.7/119.7 | 1318.1/1477.2/1477.2 | BLOQUEADO |

PG18/5.000: mediana STOP 3959.1 → 1763.9 ms (redução de 55.4%); máximo 4347.6 → 3564.3 ms.

PG16/5.000: mediana STOP 4155.8 → 1714.2 ms (redução de 58.8%); máximo 5157.0 → 1994.7 ms.

A tabela registra o n real; foram solicitadas dez repetições por cenário. P95 é descritivo pelo método nearest-rank; com
n=10 coincide com o máximo observado e tem baixa resolução de cauda. Não é uma
estimativa de SLA ou probabilidade de timeout em 72 horas. Valores não foram
arredondados para decidir cumprimento de metas; JSON conserva amostras brutas.

[Posterior PG18](evidencias/infra124-performance-after-pg18.json),
[posterior PG16](evidencias/infra124-performance-after-pg16.json).


Latências posteriores de confirmação do COMMIT (mediana/P95/máximo, ms):

| PG | Eventos | COMMIT financeiro ACK | STOP COMMIT ACK |
|---|---:|---|---|
| 18 | 0 | 1.00/1.41/1.41 | 199.49/231.64/231.64 |
| 18 | 1000 | 1.00/1.70/1.70 | 483.97/532.70/532.70 |
| 18 | 5000 | 1.41/4.18/4.18 | 1672.16/3460.18/3460.18 |
| 16 | 0 | 1.03/1.42/1.42 | 214.00/254.74/254.74 |
| 16 | 1000 | 1.08/2.40/2.40 | 504.53/602.04/602.04 |
| 16 | 5000 | 1.34/4.82/4.82 | 1638.12/1887.99/1887.99 |

## 6. Profiling dentro e fora do lock

| PG / 5.000 | Etapa | Chamadas antes/depois | Tempo inclusivo total antes/depois, ms |
|---|---|---:|---:|
| 18 | commit_protocol._hash:inside | 264/264 | 16892.4/18205.9 |
| 18 | commit_protocol.begin_intent:outside | 10/10 | 9786.2/10573.4 |
| 18 | commit_protocol.manifest:inside | 50/50 | 46465.1/50115.5 |
| 18 | commit_protocol.reconcile:outside | 30/30 | 43626.2/37914.9 |
| 18 | commit_protocol.record_unit_result:inside | 10/10 | 9158.2/10352.0 |
| 18 | commit_protocol.reject_duplicate_events:inside | 10/10 | 3131.1/200.7 |
| 18 | persistence._decode_json:inside | 1078288/732902 | 39280.0/30320.3 |
| 18 | persistence._decode_json:outside | 202507/202507 | 6113.2/6694.9 |
| 18 | trader_engine._sync_trader_orders_into_storage:inside | 10/10 | 21672.5/2443.4 |
| 16 | commit_protocol._hash:inside | 264/264 | 17043.4/16862.6 |
| 16 | commit_protocol.begin_intent:outside | 10/10 | 10101.1/9277.1 |
| 16 | commit_protocol.manifest:inside | 50/50 | 47573.8/45232.2 |
| 16 | commit_protocol.reconcile:outside | 30/30 | 44851.5/34760.5 |
| 16 | commit_protocol.record_unit_result:inside | 10/10 | 9163.5/9205.6 |
| 16 | commit_protocol.reject_duplicate_events:inside | 10/10 | 3360.8/216.2 |
| 16 | persistence._decode_json:inside | 1078288/732902 | 40464.7/27037.6 |
| 16 | persistence._decode_json:outside | 202507/202507 | 6511.3/6031.3 |
| 16 | trader_engine._sync_trader_orders_into_storage:inside | 10/10 | 23695.4/2194.5 |

Ciclo completo posterior com 5.000 eventos (medianas observadas):

| PG | Ciclo total ms | Todos os locks do worker ms | Fora do lock estimado ms |
|---|---:|---:|---:|
| 18 | 6782.2 | 4938.0 | 1825.9 |
| 16 | 6434.1 | 4667.9 | 1717.3 |

Pico Python rastreado nas amostras posteriores: 18.11 MiB. RSS high-water máximo: 289.68 MiB. RSS não permite comparação isolada antes/depois: a base rodou só benchmarks, a regressão posterior carregou a suíte completa.

Profiling registra begin_intent, manifest, record_unit_result, reconcile,
reject_duplicate_events, sincronização de ordens, decode JSON e hash. Tempos de
pais/filhos são inclusivos e não devem ser somados. O sufixo inside/outside dos
wrappers indica o contexto na entrada; begin_intent/reconcile iniciam fora e
abrem suas transações internamente.

As amostras posteriores registram ainda tempo total do ciclo, soma dos locks
externos do worker e diferença estimada fora do lock. O tempo de lock é observado
no cliente até o ACK do COMMIT, limite superior à liberação no servidor; a
parcela externa usa esse limite e inclui conexões, aquisição sintética,
serialização/apresentação e leituras fora das transações. A pequena pausa
controlada para enfileirar STOP está registrada separadamente. Não se confunde
tempo do COMMIT com duração de toda a unidade financeira.

## 7. Segurança do protocolo e integridade financeira

PREPARED/gate durável antecedem a tentativa financeira. Recibo, plano, manifesto,
identidade, saldo, posições, BUY/SELL e relatórios continuam sendo verificados.
Formato/canonicalização dos hashes e namespaces existentes não mudaram; recibos
antigos não precisam de backfill. Gate UNKNOWN, STOP, CAS e broker PAPER continuam
sendo revalidados no mesmo lock transacional global. Não há split de commit,
novo worker, retry financeiro ou liberação por timeout.

Reuso de observações só ocorre dentro da mesma verificação cercada. Testes novos
adulteram uma linha mantendo a contagem, tanto entre chamadas quanto após gravar
o recibo e antes de enviar COMMIT; a nova leitura detecta a diferença e mantém
a execução impedida. JSON com chaves duplicadas continua recusado mesmo sem
candidatos financeiros. Duplicações BUY, SELL e relatório continuam rejeitadas.
Falha após escrever ordens/estado/recibo e antes do commit reverte toda a unidade.

A sincronização preserva todas as linhas antigas; não deduplica legado nem o
"corrige". O contrato de identidade das ordens não foi ampliado nesta fase;
exactly-once arbitrário, SQL privilegiado e escritores antigos não cooperantes
continuam fora da garantia. Propostas de idempotência mais amplas ficam separadas.

## 8. STOP, ACK e metas candidatas

Máximo posterior de STOP verificado: **3564.3 ms**. Cenários posteriores fora de ao menos uma meta: pg18:after:5000, pg16:after:5000. Cenários com timeout nos ensaios antes/depois: 0. Os resultados são funcionais e de desempenho separados; não homologam uma duração prolongada.

STOP concorrente é ordenado por Events e waiter real em `pg_locks`, sem sleep
arbitrário como prova. A transação financeira já tinha adquirido o lock antes
da solicitação. Ela pode concluir; não se promete cancelamento retroativo.
Persistência só é afirmada após COMMIT e releitura, ACK após reconhecimento
verificado. Nenhuma métrica de timeout ou leitura falha é convertida em sucesso.
O teste interrompe novas repetições financeiras do cenário se STOP não persistir.

Metas avaliadas: STOP verificado <=2 s; ACK <=1 s; reconciliação <=2 s;
lock financeiro observado <=1,5 s. São critérios candidatos para as amostras,
não parâmetros do robô nem garantias de disponibilidade. Contenção com muitos
escritores, retenção pesada e volumes maiores ainda requer ensaios próprios.

## 9. Crescimento dos recibos e manutenção

| PG | Recibos | Payload bytes | Lookup ms med/P95/máx | Reconciliação ms med/P95/máx | Manutenção ms (n=1) |
|---|---:|---:|---|---|---:|
| 18 | 1 | 955 | 8.5/11.8/11.8 | 16.6/22.1/22.1 | 52.9 |
| 18 | 32 | 30820 | 7.7/12.6/12.6 | 15.5/19.6/19.6 | 46.3 |
| 18 | 128 | 122760 | 9.7/26.9/26.9 | 21.9/179.0/179.0 | 64.9 |
| 16 | 1 | 955 | 8.9/14.8/14.8 | 18.4/65.7/65.7 | 48.1 |
| 16 | 32 | 30820 | 8.7/9.9/9.9 | 17.6/24.2/24.2 | 52.9 |
| 16 | 128 | 122760 | 9.3/11.0/11.0 | 16.9/26.0/26.0 | 57.0 |

Cada versão cria progressivamente 1, 32 e 128 intenções reais PREPARED pelo
protocolo; nenhuma unidade financeira é tentada. Cada preimagem é relida sob
fence e comprovada NOT_COMMITTED, sem BUY/SELL. São recibos de intenções abortadas,
não 128 ciclos de trading. Dez leituras por identidade e dez reconciliações
por patamar medem acesso e verificação. A manutenção de retenção existente é
exercitada; todos os recibos são comparados antes/depois e permanecem intactos.

Espaço reportado é soma dos payloads UTF-8; não mede bloat físico/índices atribuível
somente aos recibos. PK app_state(namespace) já permite leitura seletiva. Não há
limpeza, TTL ou exclusão. Milhares de recibos, backups prolongados e crescimento
por 72 horas continuam pendentes. Retenção/arquivamento futuro precisa de um
protocolo próprio: inventário, evidência histórica preservada, compatibilidade
com gate ativo, backup verificável e rollback sem apagar recibos.

## 10. Testes finais e evidências

| Execução final | Total | Aprovados | Falhas/erros | Skips | Warnings | Tempo |
|---|---:|---:|---:|---:|---:|---:|
| Offline | 619 | 522 | 0 | 97 | 105 | 39.42 s |
| PostgreSQL 18.6 | 97 | 97 | 0 | 0 | 171 | 298.67 s |
| PostgreSQL 16.15 | 97 | 97 | 0 | 0 | 171 | 293.11 s |

Logs/JUnit: [offline](evidencias/infra124-offline.txt), [XML](evidencias/infra124-offline.xml);
[PG18](evidencias/infra124-postgres18.txt), [XML](evidencias/infra124-postgres18.xml);
[PG16](evidencias/infra124-postgres16.txt), [XML](evidencias/infra124-postgres16.xml).

São **619 casos distintos**: base de 600 preservada, mais 4 offline e 15 PG.
Os casos PG são repetidos em duas versões; skips offline não são aprovações.
Warnings capturados são mantidos nos logs, sobretudo depreciações preexistentes
de datetime.utcnow(). Processos filhos têm filtros padrão Python; contagens do
pytest pai não comprovam ausência universal de avisos.

Regressões executadas incluem feed lento/STOP, STOP/COMMIT, heartbeat/configuração,
START/reset contra gate, rollback, resposta perdida, TCP real, novo intérprete
Python, SIGKILL/restart do PostgreSQL, backup/restore, recibo ausente/corrompido,
diagnóstico atrasado A/B e reconciliação sem repetir trades. Os resultados novos
desses cenários da INFRA-1.2.3 são copiados para arquivos infra124-regression-*;
as evidências históricas originais permanecem preservadas.

Compileall, pip check e git diff --check concluíram com código zero.
[Verificações](evidencias/infra124-checks.json),
[resumo estruturado](evidencias/infra124-summary.json).

## 11. Reprodução, limites e arquivos

```bash
.venv/bin/python scripts/test_offline.py -q
.venv/bin/python scripts/test_infra12_postgres.py --version 18
.venv/bin/python scripts/test_infra12_postgres.py --version 16 --alpine
.venv/bin/python scripts/test_infra12_postgres.py --version 18 --infra124-benchmark-only
.venv/bin/python -m compileall -q core engines workers pages scripts tests paper_trading_engine.py
.venv/bin/python -m pip check
git diff --check
```

Runner recusa .env/DSN externa, sanitiza ambiente, impõe BROKER_MODE=paper e
BROKER_PROVIDER=paper, publica PostgreSQL somente em 127.0.0.1 e limita sockets
Python de teste a loopback. São imagens oficiais gratuitas, 1 CPU/512 MiB,
containers/volumes próprios e removidos ao final. Não toca bancos/containers
externos. Limites por cenário: 5.000 eventos, 10 repetições, janela medida
240 s (exclui seed), pico Python rastreado <128 MiB e disco livre inicial
>256 MiB. Tracemalloc e profiling adicionam custo; RSS é high-water cumulativo
do processo, não pico isolado do protocolo/servidor.

Código modificado: `core/commit_protocol.py`, `core/persistence.py`,
`engines/trader_engine.py`,
`scripts/test_infra12_postgres.py`; novos `tests/test_infra124_postgres.py` e
`tests/test_infra124_statistics.py`. Este relatório e `evidencias/infra124-*`
contêm logs, medições, JUnit, falhas anteriores e verificações. Nenhuma mudança
em parâmetros, estratégia, broker, provider, cálculos de entrada/saída,
stop_control, worker, retenção ou módulos shadow. Os últimos foram
auditados, sem necessidade demonstrada de alteração de produto nesta fase.

## 12. Riscos residuais e situações não validadas

Varreduras/hash/diagnóstico continuam O(N) sob fence. Acima de 5.000 eventos,
concorrência sustentada de UI/logs/retenção, crescimento prolongado de recibos,
falta de disco/OOM e custo de manutenção com histórico grande não foram
homologados. Arquivos locais não têm atomicidade multi-arquivo; não lhes atribuir
garantias PostgreSQL. SQL privilegiado/escritores sem protocolo podem violar
barreiras. Backup exige origem parada e destino isolado até quarentena; não
promover snapshot antigo após novos commits na origem sem revisão de época/fork.

Não executados: 72h, worker contínuo, rede WAN/TLS/failover, queda física de
energia/disco/anfitrião, computador do usuário e outros sistemas operacionais.
Nenhum banco antigo foi acessado; recuperação desse banco continua fora do escopo.

## 13. Critérios de interrupção e rollback

Interromper testes/ensaios diante de UNKNOWN, divergência, STOP sem confirmação,
timeout, recursos insuficientes ou configuração diferente de PAPER. Preservar
dados, recibos e evidências; não repetir BUY/SELL, resetar gate ou usar timeout
para liberar execução. Não reduzir história nem desativar validações para obter
aprovação de desempenho.

Amostras com máximos mais altos permanecem no conjunto; não há causa de
infraestrutura demonstrada para a variação e ela não foi descartada como ruído.

Rollback de código em branch própria, escritores parados, banco preservado e
revisão explícita. Nenhum DDL para desfazer. A base reintroduz a reescrita cara e
os tempos anteriores; não retomar automaticamente. Não apagar histórico/recibos
nem restaurar banco antigo por conveniência.

## 14. Recomendação para homologação

5.000 eventos seguem BLOQUEADOS nesta entrega. Uma fase posterior pode estudar
snapshot consistente e revisão transacional por namespace para reduzir trabalho
sob lock, mas terá de demonstrar que todos os escritores, inclusive retenção,
atualizam a revisão e que adulterações continuam detectáveis. Isso é proposta,
não protocolo/migração aplicado. Não substituir provas por cache ou contagem.

Revisão Atlas do diff e dados brutos; confirmar limites e margem no ambiente
alvo com ensaios curtos adicionais, carga de UI/retenção e crescimento realista
de recibos. Preservar testes de corrupção, STOP, resposta perdida e restauração
em qualquer otimização futura. Só considerar janela prolongada após autorização
posterior explícita do Atlas. Metas funcionais e de desempenho são avaliações
distintas; amostras dentro do orçamento não aprovam 72 horas.

## 15. Ausência de deploy e preservação PAPER

Nenhum comando Railway, deploy, PR, merge, banco remoto, serviço pago ou ordem
real foi executado. Main não foi alterada; só a branch autorizada recebe commits
via conector GitHub, com [skip ci], sem gh CLI/contorno de bloqueio. Configuração
externa de autodeploy não foi alterada nem certificada. Nenhum worker contínuo
foi iniciado. Capital, ticket, scores, thresholds, min_signal_score,
max_open_positions, provider e frequência configurada não foram alterados.
SHADOW/STUDY/DIAGNOSTIC/PLANNING permanecem sem nova autoridade; sem avanço 2.6C.
