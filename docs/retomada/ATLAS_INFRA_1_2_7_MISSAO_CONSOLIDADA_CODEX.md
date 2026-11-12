# ATLAS — INFRA-1.2.7: missão corretiva consolidada para execução futura

Data: 12/11/2026. Documento exclusivo do repositório `oliveirasamuel03-jpg/robo-investimentos`.
Branch documental: `atlas/audit-2-consolidacao-seguranca-estrategia`.
**Status: PLANNING_ONLY. Nenhuma implementação, execução de testes, deploy, PR, merge ou migração é realizada por este documento.**

## 1. Critérios de entrada e base

Prioridade absoluta: **INFRA-1.2.7 — Correção de integridade PAPER da retenção e de quarentena financeira persistente**, antes de novas otimizações ou mudanças de estratégia. A versão candidata de base é `atlas/infra-1-2-6-contrato-escritores`, commit auditado `faf3fa26b6d8c9d8edf45e8b8e8177b54dcdad05`. Na execução, verificar se o HEAD mudou e **não sobrescrever mudanças concorrentes**.

Fontes obrigatórias:
- `docs/retomada/ATLAS_INFRA_1_2_6_CONTRATO_ESCRITORES.md`;
- `docs/retomada/ATLAS_INFRA_1_2_7_CRITERIOS_CORRETIVOS.md`;
- `docs/retomada/ATLAS_PC_2_AUDITORIA_SEGURANCA_RECUPERACAO.md`;
- `docs/retomada/ATLAS_BR_1_MAPA_BRANCHES_GATES.md`;
- contraexemplos `tests/test_infra126_postgres.py`;
- relatórios/artefatos das fases 1.2.3–1.2.5.

**Não usar a ponta da branch de UI/SEC como base automática do runtime**. Preservar documentação nela, consultando por ref. Não introduzir o protótipo INFRA-1.2.5 que foi rejeitado para integração.

## 2. Ordem de implementação e gatilhos de parada

1. Confirmar somente repositório autorizado, branch exclusiva nova e SHA inicial; validar diff contra INFRA-1.2.6.
2. Reproduzir falhas P0/P1 **antes** de corrigir, em PostgreSQL local descartável, guardando relatório de evidência que distingue “falha reproduzida” de “teste passou”.
3. Congelar estratégia e parâmetros críticos; documentar hashes/equivalência. Não tocar broker real, provider ou thresholds.
4. Corrigir no-op da retenção (sem escrita quando não há item elegível), preservar seq/payload/cycle_id/recibos, recusar deduplicação de relatórios conflitantes e assegurar rollback integral.
5. Corrigir a necessidade de **quarentena durável** em divergência financeira atual mesmo com intenção original encerrada, sem falsificar ACK do STOP, sem sobrescrever intenção do ciclo B com diagnóstico atrasado do A, sem retry BUY/SELL. Ler e gravar sob fence do PostgreSQL.
6. Separar explicitamente STOP de negociação de freeze de manutenção; o worker atualmente chama retenção quando STOPPED/PAUSED e após falha. Backup consistente exige writers quiescidos, inclusive manutenção/retention. O ajuste deve preservar funcionalidades administrativas seguras e nunca liberar gate.
7. Testar versão antiga/writer SQL não cooperante. Se correção efetiva de privilégios exigir DDL/migração de banco, **parar e registrar PLANNING_ONLY**, sem ampliar escopo automaticamente.
8. Executar regressões offline e PostgreSQL **16 e 18** em bases sintéticas, injeção de falha, restart, perda de resposta do COMMIT, retenção concorrente, recuperação/restore isolado, STOP sob contenção, logs/JUnit e conferência de diffs.
9. Não tentar “resolver” STOP acima de 2 segundos aumentando timeout, apagando histórico, abaixando critérios ou tornando snapshots shadow autoridade operacional. Registrar desempenho pendente sem negar o bloqueio 72h.
10. Entregar código e evidências somente na nova branch; nenhuma PR/merge/deploy/Railway/DB remoto/72h, mesmo que testes passem. Atlas fará auditoria independente.

## 3. Comando preparado para Codex — NÃO EXECUTAR ENQUANTO INDISPONÍVEL

O bloco abaixo está pronto para copiar **quando Codex/Work voltarem**; será revisado contra o HEAD real antes da execução.

```text
Use somente o repositório robo-investimentos.
Não acesse nem altere painel-ir-softr-tools.
Trabalhe em branch separada.
Não altere a main diretamente.

ATLAS — ROBÔ INVESTIMENTO
FASE INFRA-1.2.7 — INTEGRIDADE DA RETENÇÃO E QUARENTENA DURÁVEL

PAPER_ONLY / SECURITY_FIRST / COST_CONTROLLED / LOCAL_SYNTHETIC_ONLY
NO_PRODUCTION_DEPLOY / NO_REAL_ORDERS / NO_REMOTE_DATABASE

Repositório exclusivo:
oliveirasamuel03-jpg/robo-investimentos

Base candidata:
atlas/infra-1-2-6-contrato-escritores
faf3fa26b6d8c9d8edf45e8b8e8177b54dcdad05

Criar nova branch:
atlas/infra-1-2-7-retencao-quarentena

Antes de alterar qualquer arquivo:
- Confirmar o HEAD real da base.
- Ler documentos INFRA-1.2.6, critérios INFRA-1.2.7, PC-2 e BR-1 por seus refs.
- Não usar a cadeia UI/PC/SEC como base automática de código operacional.
- Reproduzir e registrar contraexemplos, preservando a situação anterior.
- Se não existir PostgreSQL local sintético disponível, produzir relatório de BLOQUEIO; não acessar banco remoto.

MISSÃO 1 — RETENÇÃO
- Corrigir no-op que hoje regrava runtime sem arquivamento.
- Não remover/normalizar campos financeiros de identidade.
- Não aceitar deduplicação que elimina registros financeiros conflitantes.
- Recusar e fazer rollback de manutenção incompatível com recibos.
- Preservar inalterados seq, cycle_id, hash, manifesto, histórico e status quando não existir mudança elegível.
- Se arquivamento real exigir epoch/schema novo não autorizado, BLOQUEAR essa parte, sem migração remota.

MISSÃO 2 — QUARENTENA
- Diferenciar intenção UNKNOWN_COMMIT e divergência de integridade posterior.
- Quarentena deve ser durável, consultável em conexão/processo independente e fail-closed.
- START, reset, UI e diagnóstico antigo não podem liberar quarentena.
- Não sobrescrever intenção pendente do ciclo B com recibo/diagnóstico do ciclo A.
- COMMITTED e NOT_COMMITTED não autorizam retry automático.
- Se o DB estiver indisponível, não afirmar que bloqueio ou STOP foram confirmados.

MISSÃO 3 — STOP E MANUTENÇÃO
- Mapear chamadas de retenção do worker nos estados RUNNING, STOPPED, PAUSED e em exceções.
- STOP de negociações não é garantia automática de freeze de manutenção.
- Implementar guardas mínimas que preservem histórico e permitam STOP/diagnóstico seguro.
- Provar que backup sintético sob writers quiescidos não produz snapshot misto.
- Não implantar backup operacional novo nesta fase.

MISSÃO 4 — TESTES REPRODUZÍVEIS
- Testar antes/depois e capturar evidências distintas.
- Executar pytest offline e PostgreSQL 16/18 sintéticos.
- Incluir: no-op, conflito, rollback, restart, falta de ACK, falha da quarentena,
  UNKNOWN, diagnóstico atrasado A/B, conexão perdida, STOP, retenção concorrente,
  restore isolado, writer não cooperante e preservação do broker PAPER.
- Não aceitar testes que passam porque afirmam o defeito antigo.
- Baseline histórica INFRA-1.2.5: 667 casos distintos; não inventar contagem nova.
- Coletar logs, XML, tempos, warnings/skips e checks de diff/compileall/pip.
- Não mascarar falhas e não alterar expectativas para obter verde artificial.

PROIBIÇÕES
- Nenhuma ordem real, corretora real ou credencial de produção.
- Nenhuma alteração de score, min_signal_score, thresholds, capital, ticket,
  max_open_positions, perfil, provider ou estratégia.
- BOS/H1/H4/Fibonacci/webhook/shadow continuam somente estudo.
- Não alterar main, não criar PR, não fazer merge.
- Não tocar Railway, banco remoto, volume, variáveis remotas, migração ou deploy.
- Não contratar serviços pagos nem iniciar validação de 72 horas.
- Não usar gh CLI, push manual ou workaround diante de bloqueio do GitHub.

ENTREGA
Novo commit somente na branch exclusiva.
Criar docs/retomada/ATLAS_INFRA_1_2_7_RETENCAO_QUARENTENA.md.
Incluir SHA inicial/final, diff, testes antes/depois, matrizes P0/P1,
provas PostgreSQL 16/18, desempenho STOP, pendências e decisão.
Deixar 72 horas BLOQUEADAS até nova revisão Atlas.
```

## 4. Testes de aceitação essenciais — sem autorização automática por PASS

| ID | Objetivo | Bloqueio em caso de falha |
| --- | --- | --- |
| I127-01 | Retenção sem item não muda payload/seq/recibo/manifesto | Crítico |
| I127-02 | Conflito dedup recusa apagamento e faz rollback | Crítico |
| I127-03 | Divergência posterior ao COMMIT produz quarentena durável | Crítico |
| I127-04 | Quarentena permanece após restart e conexão separada | Crítico |
| I127-05 | Falha da escrita de quarentena não permite novo trade | Crítico |
| I127-06 | Diagnóstico antigo A não libera gate B | Crítico |
| I127-07 | Intenção UNKNOWN não executa nova BUY/SELL | Crítico |
| I127-08 | Retenção em STOPPED/PAUSED respeita contrato de manutenção | Alto |
| I127-09 | Backup com manutenção concorrente não produz snapshot misto | Alto |
| I127-10 | Falha na retenção preserva todas as provas e STOP | Crítico |
| I127-11 | App e SQL writers têm permissões/versões auditáveis | Alto / fase distinta se precisar DDL |
| I127-12 | Equivalência dos sinais/thresholds e broker PAPER | Crítico |
| I127-13 | PG16 + PG18 + offline + logs reais | Obrigatório para aprovação |
| I127-14 | STOP sob carga medido sem falsear ACK | 72h BLOQUEADAS enquanto acima da meta |

## 5. Decisão Atlas

Esta consolidação **não equivale a execução da correção**. Não avançar para fase visual nem dinheiro real com integridade pendente. Quando o executor voltar, cumprir a missão corretiva em branch a partir da base infra apropriada. Revisão independente e nenhum merge sem autorização específica.
