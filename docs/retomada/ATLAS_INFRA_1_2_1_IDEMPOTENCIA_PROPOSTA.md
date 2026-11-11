# ATLAS INFRA-1.2.1 — proposta de idempotência futura

**PROPOSTA NÃO APLICADA. PAPER_ONLY / SECURITY_FIRST / NO_PRODUCTION_DEPLOY.**
Não houve migração de schema local de aplicação ou remota, backfill, alteração de
histórico, chave única nova, broker real ou mudança de estratégia nesta proposta.

## Contrato atual auditado

`paper_trading_engine._new_trade_id` gera `ASSET` + 12 caracteres hexadecimais de
`uuid4`. O ID identifica uma posição; BUY e SELL daquela posição compartilham o
mesmo ID. Não é uma chave estável para a intenção de executar um ciclo: recalcular
um BUY após falha gera outro UUID. `app_rows` possui apenas sequência, namespace,
JSON TEXT e timestamp; não há unicidade de `trade_id` nem de relatório fechado.
Relatórios de SELL usam append, e a sincronização de ordens usa
(timestamp, asset, side), sem constraint SQL correspondente.

No fluxo cooperante atual, posição já aberta impede outro BUY para o mesmo ativo;
posição removida e cooldown impedem repetir aquele SELL/entrada imediata. A
unidade transacional PostgreSQL e CAS de estado evitam a duplicação causada por
um commit parcial ou draft concorrente obsoleto. Isso não significa exactly-once
para retries arbitrários, duas versões diferentes de escritor, inserções SQL
externas ou conexão perdida no instante do COMMIT.

## Modelo proposto de chave

1. Gerar um `cycle_request_id` UUID na recepção explícita de uma solicitação de
   ciclo, antes do primeiro cálculo. O mesmo ID deve acompanhar uma reconsulta
   dessa solicitação após timeout; não gerar outro automaticamente.
2. Definir `event_key` determinística por evento lógico e versão de protocolo:
   `v1:<cycle_request_id>:BUY:<asset>` para entrada; para saída,
   `v1:<cycle_request_id>:SELL:<position_trade_id>:<asset>`.
3. Relatório fechado usa chave derivada da mesma saída, por exemplo
   `v1:report:<closing_event_key>`. BUY e SELL com o mesmo `trade_id` continuam
   válidos; `UNIQUE(trade_id)` isolado seria incorreto.
4. Persistir o estado, evento, relatório e recibo final da solicitação na mesma
   transação. Uma nova tentativa com a mesma chave e conteúdo diferente deve
   falhar explicitamente; não corrigir diferenças com overwrite.

Proposta mínima para o armazenamento existente: metadados opcionais
`cycle_request_id UUID` e `idempotency_key TEXT` em `app_rows`, com índice único
parcial `(namespace, idempotency_key) WHERE idempotency_key IS NOT NULL`. Os
metadados são derivados do evento na mesma transação, não um segundo ledger.
Detalhar futuramente validação de consistência JSON/metadados e identidade do
recibo no `app_state`; não executar DDL nesta fase. Um índice único meramente
sobre o texto JSON não identifica a mesma intenção com timestamp/UUID novo.

Uma identidade de saída estável por posição pode permitir chave de SELL sem
`cycle_request_id`, mas isso pressupõe ausência de saídas parciais. Esse pressuposto
precisa ser documentado e testado antes da escolha definitiva; esta proposta não
introduz saídas parciais nem altera critérios de entrada/saída.

## Compatibilidade e atualização

- Inventariar dados antigos em modo somente leitura, inclusive IDs ausentes,
  duplicados e relatórios sem referência. Não apagar, reatribuir IDs nem deduplicar
  automaticamente o histórico.
- Registros antigos mantêm metadados NULL; o índice parcial abrange apenas eventos
  do protocolo novo. Não declarar unicidade retroativa ou ledger antigo íntegro.
- Homologar DDL e backfill opcional exclusivamente com cópias sintéticas em PG18/16.
  Planejar locks/tempo de criação do índice em proposta operacional separada.
- Coordenar parada de escritores, backup verificável e versão de protocolo antes
  de qualquer implantação futura autorizada. Versões antigas que gravem NULL
  podem contornar o contrato; não misturar escritores.
- Reconsulta de resultado incerto deve primeiro buscar o recibo pela chave estável.
  Se a leitura falhar, manter **UNCONFIRMED**. Não repetir o ciclo cegamente e não
  apresentar sucesso a partir de intenção em memória.
- STOP persistido prevalece sobre uma nova execução/retry. Reconsulta de um
  resultado já confirmado apenas informa o commit anterior; não o desfaz e não
  cria novas operações depois da parada.

## Testes necessários para a fase futura

- Dois processos com o mesmo `cycle_request_id`, mesma entrada e versões de payload
  iguais: um evento lógico, mesmo recibo e nenhum saldo/relatório duplicado.
- Mesma chave com payload diferente: erro explícito e rollback completo.
- Mesma posição com BUY e SELL: ambos aceitos, segunda cópia de cada lado recusada.
- Rollback após trade e após relatório, antes do commit; reenvio da mesma chave.
- Perda de conexão antes/durante/depois do COMMIT e leitura posterior do recibo.
- Timeout de lock seguido de reconsulta, sem retry automático.
- STOP entre cálculo e commit e antes de retry; nenhum evento adicional confirmado.
- Histórico antigo com IDs ausentes/duplicados permanece byte a byte preservado;
  novos eventos ainda recebem proteção de unicidade.
- PG18/16, dump/restore, sequência e incompatibilidade de escritores antigos.

## Critérios de rollback da proposta

Interromper a implantação futura se houver colisão inesperada de chave,
incompatibilidade de payload, indisponibilidade por locks ou divergência de
saldo/posição/evento. Parar escritores, preservar evidências e backup e retornar
à versão compatível somente após revisar o efeito sobre garantias de idempotência.
Não usar DROP/TRUNCATE ou remoção de histórico como rollback. Colunas opcionais e
índice podem permanecer inativos até decisão explícita; não misturar versões
ativas durante a troca. A volta ao protocolo anterior remove a garantia nova e
não autoriza reativação automática.

## Alcance da evidência desta fase

Os testes INFRA-1.2.1 demonstram apenas os cenários cobertos: repetição com posição
já aberta, repetição após fechamento/cooldown, falha de cálculo sem commit e
recusa de draft obsoleto antes de append. Rollback após append real e falhas de
conexão exigem e são verificados no PostgreSQL descartável. Arquivos locais não
oferecem atomicidade entre saldo, trades e relatórios; não atribuir a esse backend
as garantias transacionais do PostgreSQL.
