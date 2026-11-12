# Auditoria inicial INFRA-1.2.4

Somente robo-investimentos. Referência remota consultada antes de agir:
`atlas/infra-1-2-3-falhas-recuperacao` = `8dc5e5a83bb9f30aba8941dbb00ae3e757fe36f9`.
HEAD local limpo e idêntico. Branch nova `atlas/infra-1-2-4-contencao-stop`.

Caminho crítico auditado: manifest, begin_intent, record_unit_result,
reject_duplicate_events, reconcile; persistence_transaction e locks de namespace;
STOP/ACK; finalizador PAPER, espelho e sincronização de ordens; retenção e worker.

1. `_sync_trader_orders_into_storage` materializa todas as ordens, executa DELETE
   e INSERT de todo o histórico mesmo sem linha nova. Reatribui seq/timestamps,
   refaz serialização e N comandos de escrita sob o lock global.
2. `reconcile` valida/decodifica históricos no manifesto e em seguida os consulta
   e decodifica novamente. Pode reaproveitar as observações da mesma chamada,
   sem cache entre chamadas/transações e sem confiar em contagem/MAX(seq).
3. `reject_duplicate_events` carrega históricos mesmo quando o conjunto de
   identidades candidatas está vazio. A checagem de JSON íntegro continua no
   manifesto integral obrigatório; não confundir teste de identidade vazio
   com dispensa de validação histórica.
4. Hashes/manifestações integrais continuam O(N). Evitar remoção de provas,
   metadados incrementais não autenticados ou migração ampla nesta fase.
5. `retention.run_retention_job` continua global e pode mover tabelas financeiras
   para arquivos/namespaces. Não é autorizado otimizar apagando/arquivando mais;
   conflitos com recibos ativos precisam permanecer bloqueados.
6. Recibos usam PK app_state(namespace): lookup por ciclo não lê todos os recibos.
   Crescimento/backup/retenção serão medidos, sem exclusão.

Reprodução PostgreSQL 16 anterior às alterações de produto: 2 failed, 6 passed,
3 deselected; ordens existentes foram reescritas e a reconciliação duplicou leitura.
Logs separados preservam a falha. Nenhum teste antigo/assertion foi removido.

Refinamento após a primeira rodada incremental: save_json_state PostgreSQL e
_guard_named_state decodificavam a mesma versão atual até três vezes. As duas
guardas são somente leitura; uma validação estrita compartilhada é suficiente.
A integridade do JSON atual deve prevalecer mesmo com CAS coincidente.
