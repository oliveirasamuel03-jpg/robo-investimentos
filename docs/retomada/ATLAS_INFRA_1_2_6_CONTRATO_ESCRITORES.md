# ATLAS — INFRA-1.2.6: contrato de escritores, retenção e integridade

Data da auditoria documental: 12/11/2026.
Escopo: **PAPER_ONLY / STUDY_ONLY / READ_ONLY_AUDIT / NO_PRODUCTION_DEPLOY**.
Repositório exclusivo: oliveirasamuel03-jpg/robo-investimentos.
Branch: atlas/infra-1-2-6-contrato-escritores.
HEAD inspecionado antes deste documento: e8bbed5e0b574ee8dcda1394c7fa5e656d35d7b4.
Base da fase: cbd9a17fe40648cd6843792f7842c66da3f915d2 (INFRA-1.2.5).

## 1. Parecer executivo

**ESTUDO DOCUMENTADO COM BLOQUEIOS CRÍTICOS; NÃO HOMOLOGADO.**

A fase identificou riscos de integridade que antecedem qualquer nova otimização de desempenho:

- **P0 / CRÍTICO — quarentena não durável diante de divergência após retenção.** Um teste de estudo constrói histórico sintético, executa retenção sem linhas para arquivar, observa mudança no manifesto e recebe UNKNOWN_COMMIT/blocked de reconcile(); contudo gate_status() continua sem bloqueio persistido e assert_execution_allowed() não recusa a próxima autorização. Isso demonstra um caminho incompatível com a premissa fail-closed. Não se executou um segundo ciclo financeiro nesse teste.
- **P1 / ALTO — deduplicação destrutiva de relatórios conflitantes.** Outro teste injeta duas linhas com a mesma chave derivada, mas conteúdo financeiro diferente; a rotina de retenção preserva apenas uma, mesmo sem arquivar linha alguma. A perda de uma evidência não é reparo de integridade.
- **P1 / ALTO — contrato de escritores insuficientemente imposto no banco.** As guardas Python e advisory locks dependem de clientes cooperantes; acesso SQL direto com privilégios de UPDATE nas tabelas de controle pode modificar dados sem participar do protocolo.
- **P1 / ALTO — retenção sem alterações de negócio altera identidade financeira.** A normalização/reconstrução pode omitir metadados como cycle_id e regravar linhas/seq, invalidando recibos existentes.

Por isso, **não avançar para 72 horas, Web/Worker contínuos, FASE 2.6C ou dinheiro real**. Não executar a estratégia operacional para comprovar esses defeitos. Priorizar uma correção PAPER local na INFRA-1.2.7.

## 2. Limite exato da evidência

Esta revisão inspecionou o código e a definição dos testes na branch GitHub, sem rodar Python, PostgreSQL ou Docker nesta sessão. Não existem ainda no commit de referência os logs finais de execução da INFRA-1.2.6. Os cenários são **contraexemplos codificados**, não uma confirmação independente de que o pytest foi executado nesta etapa.

Evidências estáticas primárias:

| Achado | Fonte inspecionada |
| --- | --- |
| Retenção sem arquivo muda manifesto e não grava barreira | tests/test_infra126_postgres.py:36-56 |
| Deduplicação descarta um relatório conflitante | tests/test_infra126_postgres.py:59-75 |
| UNKNOWN ativo impede manutenção dentro da mesma transação | tests/test_infra126_postgres.py:78-94 |
| Backup reproduz inconsistência, não a corrige | tests/test_infra126_postgres.py:97-117 |
| Colisão de namespace derivado de basename | tests/test_infra126_postgres.py:119-128 |
| Inventário read-only não é verificador financeiro estrito | tests/test_infra126_postgres.py:131-144 |
| SQL direto altera gate sem participar do lock advisory | tests/test_infra126_postgres.py:147-165 |
| Teste local de papéis mínimos, versões, pending e CAS | tests/test_infra126_postgres.py:168-276 |
| Escritores detectados por AST, com limitações explícitas | scripts/studies/infra126_inventory.py |
| DDL de estudo de privilégio restrito | scripts/studies/infra126_writer_contract_local.sql |

Os 667 casos distintos (528 offline e 139 PostgreSQL por versão) são **referência histórica registrada da INFRA-1.2.5**, não contagem atual de INFRA-1.2.6. Não somar testes novos sem runner/log/JUnit; não contar skips como aprovações. CI remoto do commit consultado não foi confirmado.

## 3. Inventário e contrato de escritores (A–G)

Os contratos abaixo derivam da revisão de fontes e do inventário AST. Não são ainda uma política de ACL instalada. Colunas: A namespace; B operações; C lock; D validações; E unidade transacional; F revisão; G incompatibilidade.

| Escritor | A / B: dados e operações | C / D / E: coordenação e prova | F / G: revisão e restrições |
| --- | --- | --- | --- |
| core/persistence.py | app_state, app_rows e arquivos JSON/CSV; save/CAS/append/replace | PostgreSQL: advisory global antes de namespaces, transação e savepoints; valida JSON/STOP/gate; local: atomicidade por arquivo apenas | Não há revisão financeira confiável. SQL direto ou versões antigas não cooperam |
| core/commit_protocol.py | paper_commit_control, paper_cycle_receipt:<id>; intent PREPARED, recibo, reconciliação, diagnóstico | Intenção durável antes da unidade financeira; recibo COMMITTED no mesmo commit do ledger; reconciliação fenced e comparações integrais | Identidade de ciclo e hashes, não revisão universal; não liberar gate por timeout |
| paper_trading_engine.py e engines/trader_engine.py | paper_state, bot_state espelhado, paper_trades, trader_orders, trade_reports | Revalida STOP/PAPER/risco/CAS; saldo, posições, trades, relatórios e recibo no mesmo commit PostgreSQL | Não misturar versões; não habilitar broker real nem gatilhos shadow |
| core/stop_control.py / core/state_store.py | bot_state, STOP/ACK/START, heartbeat, configurações | CAS/revisão de controle e persistência/readback; STOP confirmado difere de ACK | Reset ou save genérico não podem desfazer STOP; caminhos legados são incompatíveis |
| pages/, app.py, worker | Delegam operações ao estado, UI, motor PAPER e retenção | Herdam as garantias das APIs; worker singular é contrato candidato | UI não é autoridade autônoma de operação; evitar escritores de versões diferentes |
| core/retention.py | Trade reports, ordens, logs, archives, resumos e catálogo | run_retention_job sob transação PostgreSQL; reescreve tabelas runtime e arquivo | A rotina atual pode mudar manifesto e perder metadados/linhas; é bloqueio P0/P1 |
| core/auth/users_store.py | auth_users_store | Guarda de credenciais/roles e API de persistência | Admin do app não é superusuário PostgreSQL |
| bot_engine.py legado | Arquivos legados próprios | Sem o mesmo CAS/fence/garantia multi-arquivo | Nunca concorrer com protocolo financeiro atual ou importar seus dados sem validação |
| backup sintético | app_state, app_rows, sequence, índices, dump/restore local | Origem parada; snapshot consistente; restore isolado + quarentena STOP | Backup preserva inclusive dados inconsistentes; não representa reconciliação |
| SQL direto / manutenção privilegiada | Qualquer tabela para a qual haja permissões de DML/DDL | Advisory locks são voluntários para escritores SQL externos | Exigir papéis mínimos e contrato coordenado; administradores continuam fora da prova universal |

O inventário de scripts/studies/infra126_inventory.py identifica chamadas Python, SQL literal, sinks de arquivos e chamadores alcançáveis estáticos. **Não provar ausência de escritores**: SQL dinâmico, reflexão, módulos externos, ferramentas administrativas e contas com permissão direta continuam sujeitos a auditoria manual.

## 4. Mapa de transações, STOP e UNKNOWN

- core/persistence.py:125-164 adquire o lock global atlas:persistence-unit antes dos locks de namespace; timeout segue 5 segundos.
- core/commit_protocol.py:97-115 determina bloqueio de execução essencialmente por active_cycle_id no gate persistido; indisponibilidade de leitura deve impedir a execução.
- core/commit_protocol.py:235-339 faz reconciliação sob fence; uma falha pode retornar blocked=True sem persistir novo gate. Caso o gate já estivesse fechado por intenção pendente, isso conserva o bloqueio. **Quando o gate estava livre e a divergência surgiu posteriormente, esse retorno isolado não estabelece quarentena durável**, conforme o contraexemplo da fase.
- core/retention.py:165-214 lê, normaliza/deduplica e substitui o runtime independentemente de haver linhas arquivadas. core/retention.py:552-624 agrega relatórios, ordens, logs, weekly summaries e catálogo.

A fronteira entre verificar uma intenção pendente e auditar integridade de histórico já resolvido precisa ser explícita. Um recibo histórico **não** deve virar prova de estado financeiro atual; ao mesmo tempo, uma divergência relevante detectada contra dados atuais não pode ser meramente informativa se os escritores permanecem habilitados.

## 5. Matriz de achados e critérios de aceitação corretiva

| ID | Severidade | Evidência/impacto | Correção candidata e critério indispensável |
| --- | --- | --- | --- |
| P0-01 | CRÍTICA | Retenção sem arquivo altera manifesto; UNKNOWN retornado sem gate durável; autorização financeira ainda possível | Um detector de divergência atual deve estabelecer quarentena financeira persistente atômica, preservada no restart. Se falhar ao gravar/ler a quarentena, a execução deve falhar fechada. Provar com conexão independente e novo processo |
| P1-02 | ALTA | Deduplicação colapsa linhas com mesma chave e payload conflitante | Rejeitar manutenção, com rollback integral e preservação das duas evidências; nunca escolher silenciosamente keep=last. Garantir contagens/seq/payload antes/depois ou isolamento de quarentena sem reparo |
| P1-03 | ALTA | Retenção sem trabalho refaz linhas e descarta cycle_id | Operação estritamente no-op quando nada muda: bytes, seq, cycle_id, hash, recibo e gate iguais. Para arquivo real, protocolo explícito de epoch/prova sem eliminar evidências; não publicar uma mudança de manifesto incompatível |
| P1-04 | ALTA | SQL externo com UPDATE direto pode alterar gate mesmo sob advisory | Papel runtime mínimo sem DML/DDL direto sensível e sem SUPERUSER/BYPASSRLS; mecanismo transacional controlado/testável em PG local; teste com conexão SQL não cooperante deve ser recusado |
| P1-05 | ALTA | Manutenção/retention e arquivos históricos têm contratos diferentes | Mapear todas as transições runtime→archive e o significado de identidade, atualização e recibo. Backup/restore não pode liberar UNKNOWN nem promover snapshot antigo após novo commit |
| P2-06 | MÉDIA | Namespace de tabela derivado de basename pode colidir entre diretórios | Reservar namespaces canônicos únicos e testar colisão sem migração remota; proposta de mudança de nomes deve preservar compatibilidade |
| P2-07 | MÉDIA | Inventário read-only de JSON não aplica exatamente o parser estrito do runtime | Não usar auditoria exploratória como prova de integridade; usar validador financeiro estrito e comparar rejeições de NaN/chaves duplicadas |

**Distinção de responsabilidade:** a correção de P0/P1 deve ocorrer em fase de implementação posterior explicitamente autorizada e com testes de PostgreSQL; esta fase de estudo não altera o runtime.

## 6. Contrato de confiança e papéis propostos

1. Um único conjunto de binários/protocolo, um único worker autorizado, PostgreSQL como fonte canônica; broker exclusivamente PAPER.
2. Escrita financeira somente por contrato controlado, com papéis separados de proprietário NOLOGIN, aplicação com privilégios mínimos e auditor somente-leitura. O protótipo SQL usa marcadores artificiais, **não implementa permissões sobre o ledger atual**.
3. Qualquer acesso de administrador, restauração, migração ou execução de versão antiga exige janela de manutenção, STOP e verificação integral; não pode ser tratado como escritor aderente sem auditoria.
4. Privilégios efetivos devem ser testados por conexão própria, incluindo negativa para UPDATE/DELETE/TRUNCATE/DDL/SET ROLE e recusa de escrita de versão antiga ou intenção pendente. O projeto ainda não tem esse rollout operacional.
5. Permissões elevadas de manutenção são parte do modelo de confiança do ambiente, não prova absoluta de inviolabilidade. Não prometer proteção contra administrador malicioso ou acesso físico ao host.
6. START, reset e UI não podem limpar divergências/UNKNOWN por conveniência. Liberação requer prova atual, identidade correta, aprovação explícita e confirmação durável.

## 7. Três alternativas de arquitetura

| Caminho | Garantia e vantagem | Limitação principal | Decisão |
| --- | --- | --- | --- |
| A — Verificação integral atual + fail-closed + retenção estrita | Preserva recibos/hash existentes e detecta alterações nos dados verificados; menor mudança de schema | O(N), locks custosos e STOP fora da meta em 5.000 eventos | **Primeira correção recomendada (INFRA-1.2.7)**, mesmo sem resolver performance |
| B — Ledger append-only versionado + vínculo auditável archive/epoch | Pode tornar retenção não destrutiva e preparar prova incremental auditável | Nova arquitetura/DDL/migração, versões mistas, idempotência e backup exigem prova extensa | **Estudo/implementação futura, após P0/P1** |
| C — Revisão transacional por namespace + ACL + auditoria integral | Potencial para reduzir verificações frequentes no caminho crítico sob hipótese de writers controlados | Contador sozinho não prova conteúdo, bypass privilegiado é possível; retenção/restore exigem semântica de epoch | **Não usar isoladamente como autoridade**; candidato posterior com validação independente |

Não integrar o protótipo rejeitado da INFRA-1.2.5. Fences amplos de tabela pioraram STOP e a prova provisória MVCC não basta para publicar confirmação.

## 8. Plano de correção em fases, sem misturar desempenho

**INFRA-1.2.7, fase específica PAPER:** corrigir P0-01, P1-02 e P1-03, preservando hashes e recibos legados; endurecer guardas do fluxo de retenção e de reconciliação atual; demonstrar rollback, quarentena, idempotência e ausência de retry. Investigar P1-04 e preparar política/testes locais, sem impor migração operacional ou alterar privilégios reais sem fase explícita.

**Após revisão Atlas:** somente quando os bloqueios de integridade estiverem fechados, escolher arquitetura futura de escalabilidade (B ou C sob contrato auditado). Não aumentar timeout, reduzir histórico, apagar recibos ou mudar frequência/capital para atingir metas.

Aceitação mínima da INFRA-1.2.7:
- P0 e P1 reproduzidos contra a base e corrigidos no novo código; não basta apenas passar testes que descrevem o defeito.
- Falha de gravação/releitura da quarentena mantém execução financeira bloqueada, inclusive após restart. STOP continua possível e não é confundido com UNKNOWN_COMMIT.
- Retenção sem arquivo não muda namespace canônico, bytes/seq/identidades e manifestos; retenção conflitante não apaga nenhuma evidência.
- Retenção com arquivo tem contrato de integridade e compatibilidade aprovado ou permanece explicitamente bloqueada; não forçar habilitação em produção.
- Nenhuma operação financeira automática em testes de reconciliação, nenhum segundo BUY/SELL após COMMIT incerto; rollback preserva todos os namespaces.
- Pelo menos a suíte anterior completa e os cenários novos nos PostgreSQL 18 e 16, com logs/JUnit e comparação independente dos resultados; teste offline não substitui PostgreSQL.
- Estratégia, scores, thresholds, capital, ticket, max_open_positions, provider, broker e shadow authority intactos.
- **Nenhuma autorização de 72 horas** até desempenho, recuperação, integridade e ambiente alvo cumprirem critérios separados.

## 9. Medições e riscos de desempenho herdados

A INFRA-1.2.4 (produto operacional preservado) documentou 619 casos distintos, 522 offline e 97 PG por versão, melhoria mediana do STOP de 55,4% no PG18 e 58,8% no PG16 com 5.000 eventos, mas máximo observado após correção de 3,564 s PG18 e lock acima de 1,5 s em ambos. A INFRA-1.2.5 documentou 667 casos distintos (528 offline, 139 PG por versão) e rejeição do protótipo; STOP adversarial até 5,044 s. Metas candidatas continuam: STOP verificado <=2 s, ACK <=1 s, lock financeiro <=1,5 s, reconciliação <=2 s; precisam de amostras novas em ambiente alvo. Integridade vem antes da otimização.

Ainda não foram testados: operação contínua 72h; ambiente do computador do usuário; quedas físicas de energia/disco; múltiplos workers; produção; dinheiro real. Storage local não tem atomicidade multi-arquivo. Escritores SQL privilegiados e legados permanecem fora de garantia universal.

## 10. Estratégia de migração e rollback futuros

- Não existe migração autorizada nesta fase. Para mudança futura: congelar writers, STOP confirmado, inventário de schema/permissões e integridade inicial, backup validado, teste de restore em destino isolado, implantação coordenada de uma versão única.
- Qualquer novo campo/índice/protocolo requer compatibilidade com recibos anteriores e rollback que **não** reative writers incompatíveis com gates pendentes. Preserve arquivos/ledger/recibos antes e depois.
- Na falha: permanecer STOPPED/quarentena, preservar logs e conteúdo, evitar rollback parcial de código que ignore o novo contrato. Nunca usar restore como apagamento de resultado financeiro confirmado.

## 11. Arquivos adicionados nesta fase e condição de encerramento

- scripts/studies/infra126_inventory.py: inventário AST com contratos e limitações.
- scripts/studies/infra126_writer_contract_local.sql: protótipo de marcador/ACL local; não financeiro.
- scripts/test_infra126_postgres.py: wrapper do runner PostgreSQL local.
- tests/test_infra126_postgres.py: contraexemplos de retenção, bloqueio, escritores e ACL.
- tests/test_infra126_study.py: limites de AST/arquivo legado.
- Este relatório é **documentação adicional**, não correção operacional.

**Status de encerramento:** revisão de arquitetura e matriz de requisitos documentadas. A conclusão de testes da INFRA-1.2.6 permanece **PENDENTE** até logs finais de pytest, PostgreSQL 18/16 e checks. Não declarar a fase plenamente validada por esta escrita documental. Nenhum código operacional foi alterado por este documento.

## 12. Restrições imutáveis

PAPER obrigatório. Sem ordens reais, broker real, ajustes de score/threshold/min_signal_score/capital/ticket/max_open_positions/provider, conversão de SHADOW/STUDY/DIAGNOSTIC em gatilho operacional, alteração de main, PR, merge, deploy, Railway, DB remoto/volume/variáveis, cobrança externa ou worker contínuo.

A próxima decisão do Atlas é auditar as evidências quando o executor voltar a estar disponível e autorizar uma INFRA-1.2.7 corretiva **somente em nova branch**, com testes. Até lá, manter o ambiente operacional desligado.
