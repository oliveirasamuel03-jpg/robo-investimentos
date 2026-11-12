# ATLAS — Critérios de implementação da INFRA-1.2.7 (PLANEJAMENTO, SEM EXECUÇÃO)

Data: 12/11/2026. Somente repositório oliveirasamuel03-jpg/robo-investimentos.
Documento de planejamento associado à INFRA-1.2.6; **nenhuma alteração de runtime é autorizada por este arquivo**.
Modo: **PAPER_ONLY / PLANNING_ONLY / SECURITY_FIRST / COST_CONTROLLED / NO_PRODUCTION_DEPLOY**.

## 1. Missão futura

Remover duas classes de risco financeiro antes de nova otimização:

- P0: divergência financeira detectada após manutenção/retenção não pode deixar o robô livre para nova unidade financeira por ausência de fence durável.
- P1: retenção jamais pode modificar o ledger/relatórios quando for no-op, nem descartar silenciosamente conteúdo financeiro conflitante.

Como objetivos secundários: demonstrar contrato de escritor mínimo em PostgreSQL local, preservar STOP/UNKNOWN, identidade do ciclo e recibos anteriores, e não piorar disponibilidade sem identificar o motivo.

**Status atual: especificado, NÃO implementado, NÃO testado.** O baseline funcional conhecido é INFRA-1.2.5: 667 casos distintos documentados. INFRA-1.2.6 adicionou definições de testes e protótipo local, mas resultados finais de execução ainda não foram registrados.

## 2. Ordem obrigatória da futura execução Codex

1. Verificar HEAD e ler docs/retomada/ATLAS_INFRA_1_2_6_CONTRATO_ESCRITORES.md. Trabalhar somente robo-investimentos; nunca painel-ir-softr-tools. Criar branch separada a partir do HEAD auditado da INFRA-1.2.6. Jamais alterar main.
2. Executar e conservar evidências dos contraexemplos INFRA-1.2.6 **contra a implementação antiga**, sem confundir testes que comprovam defeito com aprovação. Registrar a condição inicial.
3. Especificar exatamente a transição de estados entre integridade normal, divergência atual, quarentena persistida e liberação após prova. Incluir separação entre controle STOP do operador, intenção UNKNOWN_COMMIT e quarentena de integridade. Proibir liberação por timeout, UI, reset comum ou falha posterior de diagnóstico.
4. Implementar correção mínima e transacional da retenção e do bloqueio; NÃO introduzir ledger paralelo, revisão arbitrária, trigger de produção ou migração remota.
5. Substituir/adaptar os testes que anteriormente afirmavam comportamento defeituoso por testes de comportamento esperado, **preservando a reprodução histórica em teste pinado ao commit anterior ou artefato de evidência**. Não reduzir as assertions.
6. Executar suíte offline, PostgreSQL 18 e 16 locais e testes de fault injection/restart/backup/restore quando disponíveis. Não marcar skips como aprovação.
7. Auditar segurança PAPER, diff e evidências; documentar perigos restantes. Nenhum PR, merge, deploy ou Railway.

## 3. Máquina de estados do bloqueio financeiro (requisito de projeto)

Estados conceituais, não schema aprovado:

- READY_VERIFIED: última verificação necessária foi concluída, gate/recibo estão coerentes, nenhuma intenção pendente; não é autorização autônoma para gerar sinal.
- INTENT_PENDING_UNKNOWN: ciclo financeiro pendente/incerto; nenhum retry ou novo commit permitido; STOP continua disponível.
- INTEGRITY_QUARANTINE: divergência observada em dados de controle/histórico ou prova histórica usada indevidamente como atual; bloquear novas unidades até prova explícita e durável.
- RECONCILIATION_VERIFIED: prova positiva com identidade e manifesto compatíveis; separar ato de verificar da gravação confiável da resolução.
- UNAVAILABLE_UNCONFIRMED: PostgreSQL/lock/recibo inacessível ou confirmação perdida; não anunciar nem remover bloqueio por inferência.

Requisitos de concorrência:
- Operação financeira deve checar a restrição atual dentro do mesmo fence transacional no instante do commit, além dos preflights.
- Quarentena não deve sobrescrever active_cycle_id de outra intenção, nem ressuscitar recibo histórico; definir campo/estado distinto se necessário.
- Se a aplicação detecta divergência e a persistência da quarentena falha, aquela requisição não pode prosseguir; no reinício, a proteção durável precisa ser reconstituída pela política de leitura/prova atual antes de qualquer novo commit.
- Se a ausência de banco impedir leitura, falhar fechado; STOP deve informar solicitação não confirmada, nunca sucesso.
- Nenhum caminho permitido deve poder limpar a quarentena por START, heartbeat, reset, arquivamento, atualização de interface ou diagnóstico tardio.
- Liberação explícita deve comprovar identidade atual, ledger, relatórios, recibos e estado da origem sob fence, com auditoria; não executar BUY/SELL na reconciliação.

## 4. Contrato da retenção financeira

- Se não há dados a arquivar nem transformação legitimamente necessária, nenhuma substituição de tabela pode ocorrer. Exigir igualdade de payload, sequência, cycle_id, contagem e manifesto antes/depois.
- Não escolher arbitrariamente keep=last em conflito financeiro. Identidade igual e conteúdo diferente deve provocar conflito, rollback de toda a manutenção e preservar ambas as linhas.
- Para ordens e relatórios já referenciados por recibos, manter identidade/versionamento e garantia de recuperação. Se o arquivo financeiro real não puder ser reconciliado com os recibos atuais, **bloquear essa manutenção específica** em vez de aplicar conversão destrutiva.
- Distinguir retenção de logs/diagnósticos não financeiros de mutações em namespaces incluídos em prova financeira; não silenciar erros de manutenção.
- Nunca alterar score, parâmetros ou frequência para reduzir volume; nenhuma exclusão retroativa de histórico para obter desempenho.
- Impedir archive/runtime fork por restore antigo com origem ativa; testes de backup confirmam dados, não autorização para desbloquear divergência.

## 5. Matriz de testes de aceitação

| ID | Cenário | Resultado esperado |
| --- | --- | --- |
| T01 | Retenção de relatório sem item elegível | No-op de bytes/seq/cycle_id/manifesto/recibo e nenhuma mudança no gate |
| T02 | Duas linhas financeiras com mesma chave e conteúdo diferente | Erro explícito, rollback integral; ambas as linhas preservadas |
| T03 | Mutação financeira relevante sem intenção ativa, detectada por auditoria | Quarentena durável, execução recusada pelo mesmo processo e novo processo |
| T04 | Divergência + erro ao gravar a quarentena | Não executar ciclo; após recuperação, comprovar o estado antes de liberar; nunca indicar bloqueio durável sem confirmação |
| T05 | UNKNOWN_COMMIT já pendente durante retenção | Gate, recibos e conteúdo inalterados; STOP pode ser persistido |
| T06 | Diagnóstico atrasado do ciclo A enquanto B está pendente | Nenhuma liberação/alteração do gate B, sem novo trade |
| T07 | Retenção com arquivamento válido e recibo correspondente | Ou prova transacional completa da compatibilidade ou manutenção recusada sem efeitos parciais |
| T08 | Falha entre writes de archive e runtime | Rollback PostgreSQL completo; arquivos locais explicitamente não homologados para multi-arquivo |
| T09 | Backup/restore com quarentena ou recibo ausente | Destino STOPPED e bloqueado, sem RUNNING automático e sem retry financeiro |
| T10 | Reinício do worker em estado bloqueado | Nenhum BUY/SELL, heartbeat pode funcionar sem liberar |
| T11 | Concorrência STOP, retenção e commit | STOP não falsamente confirmado; nenhum commit após STOP confirmado |
| T12 | Role de aplicação sintética sem UPDATE/DELETE/DDL sensível | SQL direto proibido; aplicação de protocolo controlado apenas se implementado localmente |
| T13 | Retenção preservando relatório com campos extras de identidade | cycle_id e todos os campos financeiros indispensáveis intactos; rejeitar perda silenciosa |
| T14 | Histórico com corrupção de JSON, NaN, chave duplicada, colisão de namespace | Falhar fechado; não normalizar silenciosamente ou sobrescrever |
| T15 | Banco desconectado ou timeout ao verificar/quarentenar | Resultado UNCONFIRMED e falha segura; sem retry financeiro |
| T16 | Equivalência de sinais/estratégia em feed sintético | Sem alteração de score, thresholds, capital, ticket, max_open_positions, provider, broker ou autoridade shadow |

Além desses casos, preservar as provas anteriores de TCP real, COMMIT confirmado/resposta perdida, SIGKILL/restart de PG, arquivamento, invariantes financeiras, STOP e restauração. O protocolo deve funcionar em PG18 e PG16 locais descartáveis.

## 6. Condições de bloqueio e liberação

Fase corretiva NÃO aprovada quando ocorrer qualquer um:
- Quarentena não persistida mas interface diz que está confirmada, ou novo ciclo financeiro aceita histórico divergente.
- Falha ou perda de evidência de relatórios, recibos, seq, ciclo ou posições.
- Restrição UNKNOWN/STOP contornada por START, reset, heartbeat ou writer incompatível.
- Migração remota, alteração de broker, ordem real, estratégia ou parâmetros.
- Testes que passam apenas porque continuam afirmando a presença do defeito.
- Nenhuma prova PostgreSQL executada; nesse caso relatar PENDENTE, não APROVADO.

Mesmo se T01–T16 forem aprovados, **não libera 72 horas**. Será necessário, em etapa posterior, resolver STOP/lock com carga, testar longo prazo com um worker e banco alvo e revisão Atlas independente.

## 7. Documentação/evidências exigidas na implementação

Criar docs/retomada/ATLAS_INFRA_1_2_7_RETENCAO_QUARENTENA.md e evidências associadas:
- HEAD/base/branch, reprodução anterior, correções, diff e mapa de estados.
- Log da suíte, JUnit, testes pg18/pg16, warnings e skips.
- Evidência de conexão independente para quarentena/STOP.
- Backups/restore, perda real/simulada de conexão, concorrência.
- Baseline 667 + testes INFRA-1.2.6 executados + novos casos, sem somar execuções duplicadas por versão.
- Riscos ainda não resolvidos: O(N), prazos STOP, privilégio externo, retention epoch, storage local.
- Segurança PAPER e confirmação de nenhuma alteração main/PR/merge/deploy/Railway.

## 8. Custos e limites de autorização

Somente novo branch e alterações explicitamente autorizadas na futura fase; não executar esta especificação automaticamente.
Não criar infraestrutura paga, serviço Railway, banco remoto, variáveis remotas, worker contínuo, migração ou ordem real.
Não contornar bloqueios do conector GitHub via gh CLI, merge local ou push manual.
O objetivo é **provar que falhas financeiras fecham o sistema antes de torná-lo mais rápido**.
