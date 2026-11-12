# INFRA-1.2.5 — baseline e auditoria de escritores

HEAD remoto consultado antes de qualquer mudança: branch
atlas/infra-1-2-4-contencao-stop = 64ca64f25020983546bfbce8427e7adac437ddf1.
HEAD local limpo e idêntico. Branch nova atlas/infra-1-2-5-snapshot-transacional.
Nenhum arquivo de produto foi alterado para o estudo.

Baseline preservada: evidências originais INFRA-1.2.4, 619 casos distintos,
522 offline, 97 PG por versão. Manifesto integral ainda executa cinco vezes por
ciclo no benchmark anterior: begin_intent (1), record_unit_result (1),
reconciliações (3, incluindo a reconsulta explícita do ensaio).
Reproduções novas current/snapshot usarão os mesmos dados e recursos.

Escritores cooperantes:
- persistence_transaction: lock global antes de namespaces/savepoints.
- save/update_json_state: CAS, guardas STOP e financeiras, app_state.
- append/replace_json_rows e tabelas CSV: app_rows, incluindo DELETE/INSERT.
- PAPER: estado, trades, relatórios, recibo na mesma unidade financeira.
- trader: ordens incrementais e espelho saldo/posições no mesmo commit.
- STOP/ACK/START, heartbeat e UI: bot_state compartilhado, guardas de revisão.
- retenção: move/substitui tabelas, arquiva relatórios/ordens/logs e cria resumos.
- worker: usa essas APIs; diagnóstico antigo preserva a identidade mais recente.

Limite da revisão em Python: SQL direto, escritor antigo, COPY, TRUNCATE,
ALTER/namespace move e restauração podem contornar instrumentação de aplicação.
Revisão apenas na aplicação NÃO pode ser prova. DDL local com triggers será
avaliado, incluindo rollback e bypass privilegiado; nunca aplicado ao produto.

Candidata isolada: verificador original em RR READ ONLY, plano de metadados em
memória e comparação exata de payload/seq/namespace sob fence antes de publicar.
Inclui gate/recibo/estado, não só contagem ou MAX(seq). Identidade do servidor,
instância e banco impede transporte de prova entre restart/restore.
Fences SQL de tabelas evitam DML externo durante a publicação, mas ampliam
contenção. Hipótese de bloqueio a ser testada, não aprovada antecipadamente.
Não otimiza record_unit_result/commit financeiro; não presume resolver sua meta.
