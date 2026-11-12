# ATLAS — PC-2: Auditoria estática de segurança, parada e recuperação local

**Data:** 12/11/2026.
**Repositório único:** oliveirasamuel03-jpg/robo-investimentos.
**Branch documental:** atlas/pc-2-auditoria-seguranca-recuperacao.
**Base verificada:** atlas/pc-1-plano-local-backup-custos, SHA 6759222afc0167af72af5ae27a2fbbca3fb42a37.
**Natureza:** SECURITY_REVIEW / PLANNING_ONLY / PAPER_ONLY / NO_PRODUCTION_DEPLOY.

**Nenhum teste foi executado neste estudo.** Revisão de fonte e metadados do repositório conectado; sem abrir arquivos potencialmente sensíveis de storage, instalar dependências, abrir conexões PostgreSQL, iniciar web/worker, produzir backup, acessar Railway ou fazer alteração em módulos operacionais. Achados estáticos devem ser reproduzidos com dados sintéticos antes de afirmar impacto observado.

## 1. Veredito e escopo

**APROVADA A DOCUMENTAÇÃO DA AUDITORIA, NÃO A PRONTIDÃO OPERACIONAL.**

Existem controles relevantes: autenticação obrigatória, verificação bcrypt, sessões com expiração e consulta da conta persistida, guardas de START/STOP/UNKNOWN_COMMIT, transações PostgreSQL, backups descartáveis com isolamento e scripts de teste locais. Contudo, a retenção financeira pode reescrever e consolidar evidências e a reconciliação pode devolver bloqueio não persistido quando a intenção anterior já está encerrada (INFRA-1.2.6). O STOP sob 5.000 eventos continua acima da meta de homologação em amostras relevantes. Esses bloqueios são anteriores a qualquer decisão de funcionamento 72h ou compra de hardware.

### Escopo inspecionado

`.gitignore`, `.env.example`, `core/auth/security.py`, `core/auth/session.py`, `core/auth/users_store.py`, `core/persistence.py`, `core/commit_protocol.py`, `core/retention.py`, `core/stop_control.py`, `scripts/local_web.py`, `scripts/check_postgres_local.py`, `scripts/local_synthetic_backup.py`, `workers/trader_worker.py`, `tests/test_infra126_postgres.py`, `tests/test_infra12_retention_postgres.py`, diretórios GitHub sob `storage/` (somente nomes/metadados). Também foram considerados os planos INFRA-1.2.6, INFRA-1.2.7 e PC-1.

Esta auditoria **não afirma** que há credenciais expostas, que dados pessoais estejam em logs, que funções sensíveis foram exploradas ou que a máquina do usuário foi comprometida. Tais conclusões exigiriam inspeção autorizada, preferencialmente automatizada, sem divulgar segredos.

## 2. Achados de maior prioridade

### PC2-P0-01 — CRÍTICO — divergência financeira pode não ativar quarentena persistida

Referências: `tests/test_infra126_postgres.py:36-56`, `core/commit_protocol.py:97-115,235-339`, relatório INFRA-1.2.6.

A retenção, mesmo com zero registros elegíveis para arquivo, pode mudar a identidade/hashes do manifesto. A reconciliação subsequente informa `UNKNOWN_COMMIT` e `blocked=True`, mas, quando `active_cycle_id` já foi liberado, o gate persistido pode permanecer aberto. O teste sintético existente chama `assert_execution_allowed()` com sucesso após esse retorno, **sem executar nova ordem**.

Exigência de correção: detector de divergência atual deve estabelecer barreira financeira durável, distinta da intenção histórica ou equivalente formalmente provado; se a gravação ou leitura falhar, nenhuma solicitação financeira pode prosseguir. Provar readback por conexão independente, restart, concorrência, identidade e bloqueio sem retry.

### PC2-P0-02 — CRÍTICO / DEPENDÊNCIA — retenção pode destruir/reescrever provas financeiras

Referências: `core/retention.py:165-214`, `tests/test_infra126_postgres.py:36-75`.

A função de arquivamento substitui a tabela runtime mesmo quando nada é arquivado e usa `_dedupe_rows(...keep='last')` em dados que podem ser distintos financeiramente. Uma linha com conteúdo conflitante pode desaparecer, e campos de identidade como `cycle_id` podem ser removidos na reconstrução tabular, invalidando recibos.

Exigência de correção: no-op real que preserva bytes/seq/cycle_id/manifesto; dados conflitantes devem causar falha e rollback integral, nunca descarte silencioso. Arquivamento real requer contrato explícito de manutenção que preserve compatibilidade com recibos.

### PC2-P1-03 — ALTO — manutenção de retenção é invocada também com bot STOPPED/PAUSED

Referências: `workers/trader_worker.py:1873-1897`, `workers/trader_worker.py:1917-1920,1969-1976`, `core/retention.py:535-549,552-624`.

No caminho `bot_status != RUNNING`, o worker reconhece STOP e ainda chama `_run_daily_retention_maintenance()`. Na rota de erro, essa função também é chamada. `should_run_retention_job()` decide por enabled/intervalo/last_run, **não pela confirmação de um freeze de manutenção financeira**.

Não confundir: STOP de execução de trades não foi necessariamente projetado como STOP de todas as escritas administrativas. No entanto, qualquer procedimento futuro de backup que assuma “STOPPED = ledger imutável” **é insuficiente sem congelamento verificável dos escritores de manutenção**. O arquivo sintético de backup utiliza lock comum e fonte STOP; isso não converte a rotina geral em política de backup operacional. O risco se amplifica pelos defeitos atuais da retenção.

Exigência: definir separação clara entre STOP de trades e MAINTENANCE_FENCE; validar que o backup do futuro PC não concorre com arquivamento/renumeração/alterações no histórico; bloquear manutenção financeira destrutiva em estado incompatível, com log e sem alteração parcial. Preservar heartbeat/STOP administrativo necessário para diagnóstico, e não quebrar recuperação de UNKNOWN.

### PC2-P1-04 — ALTO — arquivos históricos e artefatos de validação em árvore versionada

Referências: `.gitignore`, diretórios GitHub `storage/archive/`, `storage/tmp_real_env_validation/`, `storage/tmp_real_env_validation_hardening/` e diretórios `storage/archive/runtime_resets/`.

O `.gitignore` protege `storage/runtime/*`, `storage/cache/*`, `storage/reports/*` e certos formatos sob a raiz de storage, mas **não cobre recursivamente** todos os subdiretórios `storage/archive/*` e `storage/tmp_*/`. Na branch consultada existem artefatos versionados de testes legados, incluindo nomes de capturas de tela, arquivos de logs e relatórios de validação, além de diretórios antigos de resets.

**Não houve leitura dos conteúdos desses artefatos nesta auditoria.** Não há prova de vazamento de segredo nem de dados pessoais reais. `.gitignore` também não desversiona arquivos já rastreados.

Exigência: antes de qualquer publicação, varredura local/segura de PII, credenciais e URLs; inventário com classificação `SYNTHETIC / SANITIZED / UNKNOWN / SENSITIVE`, sem expor conteúdo. Só depois propor regras `.gitignore` mais amplas, migração de evidências históricas para local apropriado e plano de preservação. **Não apagar, reescrever histórico Git ou fazer push forçado nesta fase.** Rotacionar eventuais credenciais somente se a auditoria futura comprovar exposição e após autorização explícita.

### PC2-P1-05 — ALTO — ferramenta sintética de backup não é runbook operacional

Referência: `scripts/local_synthetic_backup.py:1-35,58-161`.

A função valida origem local, container Docker pertencente ao runner e banco sintético; exige STOP, registra inventário e executa `pg_dump/pg_restore` em destino recém-criado. É proteção adequada ao ambiente de teste, mas não fornece rota aprovada para um banco PAPER real/futuro, backup de arquivos externos ao banco, cópia fora do PC, retenção criptografada, gestão de segredos ou recuperação completa após perda de disco.

Exigência: nova fase para desenho, implementação e testes de backup operacional, em **banco sintético local descartável** primeiro. Fixar consistência PG+filesystem, roles, STOP/MAINTENANCE_FENCE, hashes e restore STOPPED/UNKNOWN preservados. Backup íntegro não é prova de estratégia rentável, nem reparo de recibo divergente.

### PC2-P1-06 — ALTO — acesso SQL direto não cumpre guardas do runtime

Referências: `tests/test_infra126_postgres.py:147-165`, protótipo de papéis mínimos de estudo, `core/persistence.py:125-164`.

Um escritor SQL com UPDATE concedido sobre tabelas sensíveis pode evitar o contrato Python e locks advisory voluntários. O protótipo `infra126_writer_contract_local.sql` prova restrições numa tabela de marcadores sintética, **não** nas tabelas financeiras reais.

Exigência: papel de aplicação sem acesso administrativo, compatibilidade de versões/escritores, travas transacionais e teste PG local com conexão independente. Não alterar privilégios remotos ou fazer migração sem fase explícita.

## 3. Observações de segurança secundárias

### PC2-P2-07 — configuração de exemplo contém ramo legado de autenticação

`.env.example` contém `AUTH_REQUIRED=false`; `core/auth/session.py:40-45` retorna `True` incondicionalmente, portanto esse exemplo **não comprova bypass de autenticação**. Retirar ambiguidade da documentação no momento adequado; manter login e session RBAC.

### PC2-P2-08 — criação implícita de schema e efeitos de leitura

`core/persistence.py:248-284` contém `_ensure_db_schema()` com `CREATE TABLE/INDEX IF NOT EXISTS`; aplicações que abram armazenamento via APIs de alto nível podem inicializar schema no destino se a guarda permitir. Não executar “smoke test de leitura” num banco operacional/antigo esperando ausência de escritas. Scripts de teste `scripts/check_postgres_local.py` e `scripts/local_web.py` isolam ambientes e não devem ser apontados para banco remoto.

### PC2-P2-09 — portabilidade e isolamento

`core/persistence.py:7,84-104` depende de `fcntl.flock`; Windows Python nativo não está homologado. Preferir Linux/WSL2 (sujeitos a teste), um worker, app em loopback e DB local não exposto. `scripts/local_web.py` limpa o ambiente, usa storage temporário e restringe rede no processo do demo; **não foi validado como isolamento de segurança geral nem launcher de produção**.

### PC2-P2-10 — saúde operacional, integridade e disponibilidade distintas

`core/production_monitor.py` mostra heartbeat/worker/feed/broker, mas não prova ledger íntegro; `core/commit_protocol.py` pode ter gate aberto sem prova de histórico íntegro. A UI-2/UI-3 devem diferenciar `HEALTHY`, `INTEGRITY_UNVERIFIED`, `STOP_PERSISTED`, `STOP_ACKNOWLEDGED`, `UNKNOWN_COMMIT`, `POST_COMMIT_DIAGNOSTIC_FAILURE`, `BROKER_PAPER_EFFECTIVE` e `FEED_LIVE_OPERATIONAL`. Indicador verde isolado jamais autoriza trade.

## 4. Avaliação de autenticação e privilégios

**Controles vistos no código (não testados nesta sessão):**
- `core/auth/security.py`: hashing bcrypt e verificação da senha, falha fechada se bcrypt ausente.
- `core/auth/users_store.py`: persistência de hash, roles limitadas a admin/user, rejeição de corrupção e duplicatas, proteção do último admin habilitado.
- `core/auth/session.py`: login obrigatório, expiração e consulta à conta persistida (incluindo conta desabilitada); ausência de banco resulta em logout, não acesso anônimo.
- `core/auth/guards.py`: separação require_auth/require_admin; comandos operacionais com admin.
- `scripts/local_web.py`: demo com admin sintético e socket local, sem `.env` e sem worker.
- Política do PC futuro ainda exige proteção de dispositivo, disco/backup cifrados, senhas robustas, monitoramento de conta e rede privada. Não foi feito teste de pentest, reset de senha ou autorização API.

**Pendente:** validação dinâmica de RBAC, autenticação com sessão expirada, revogação durante uso, usuários desabilitados, possíveis chamadas indiretas e confirmação de que o broker efetivo nunca sai do PAPER.

## 5. Contrato futuro de congelamento e recuperação

Desenho conceitual, **não API pronta**:

```
NORMAL_PAPER_STUDY
   -> STOP_REQUESTED
   -> STOP_PERSISTED
   -> STOP_ACKNOWLEDGED (worker)
   -> FINANCIAL_WRITERS_QUIESCED (inclui manutenção/retenção)
   -> CONSISTENT_BACKUP (PG + arquivos relevantes)
   -> RESTORE_IN_ISOLATION
   -> INTEGRITY_REVIEW / UNKNOWN_PRESERVED
   -> STOPPED_QUARANTINED
```

STOP_ACKNOWLEDGED por si só não prova ausência de outras escritas administrativas. Um fence de backup deve impedir, registrar e aguardar escritores cooperantes; SQL privilegiado continua fora da prova até controle de roles. Falha em qualquer transição não autoriza backup marcado como completo nem START automático. Nada de replay BUY/SELL após restore.

**Requisitos não negociáveis de recuperação futura:**
- Identificador/hash da origem e SHA do software gravados sem expor credenciais.
- Origem sem operações financeiras, sem manutenção mutável e sem processos de versão antiga.
- Captura consistente de PostgreSQL (tabelas, índice, sequência, gate, recibos, STOP, ledger) e arquivos fora do banco quando estes forem parte do estado financeiro.
- Checksum de artefato protegido e inventário; cópia em mídia separada; restore em ambiente *novo* sem worker; sem sobrescrever banco existente.
- Comparar before/after sob fence; falhas devem deixar destino STOPPED/quarentena e a prova de UNKNOWN intacta.
- RTO/RPO só definidos após testes de tamanhos, política de backup e durabilidade. Sem declarar restore operacional concluído com evidência exclusivamente sintética.
- Política de backup não pode depender de credenciais de admin do app nem conceder privilégios genéricos ao worker.

## 6. Plano corretivo e testes que deverão ser executados em Codex

**Prioridade 1 — INFRA-1.2.7:** corrigir retenção com no-op real, preservação de identidade e rejeição de conteúdo conflitante; fechar gate de quarentena persistente após divergência atual. Adicionar explicitamente teste de `STOPPED/PAUSED` enquanto retenção estaria elegível e prova de ausência de writes financeiros inesperados no procedimento de backup.

**Prioridade 2 — migração/ACL e backup planejados separadamente:** contrato de escritores de menor privilégio + protótipo em PG16/18 descartável. Depois planejar backup operacional local isolado e restauração sem reiniciar worker. Nunca mexer em banco remoto ou histórico legível sem autorização de fase.

**Prioridade 3 — segurança de artefatos versionados:** inventário de `storage/tmp_*` e `storage/archive/runtime_resets`, varredura segura de segredos e PII, separação de evidência pública de artefatos locais, e `.gitignore` com exceções explícitas para `.gitkeep` e evidências sanitizadas. Sem exclusão/reescrita histórica automática.

**Prioridade 4 — operação e UI:** apenas após integridade e retomada seguras, integrar painéis UI-2/UI-3, observando tempos/custos da interface, permissão admin, indicador financeiro não homologado e acessibilidade móvel em ambiente local.

### Matriz de aceitação PC-2 (futura execução; todos PENDENTES)

| ID | Cenário local sintético | Critério objetivo |
| --- | --- | --- |
| PC2-T01 | Worker STOPPED e retenção elegível | Nenhuma alteração de evidência financeira não autorizada; estado de manutenção explícito |
| PC2-T02 | Worker PAUSED e retenção elegível | Não presumir freeze financeiro; manutenção somente com contrato aprovado |
| PC2-T03 | Gate UNKNOWN e retenção | Recibos/ledger inalterados, STOP disponível, zero retry |
| PC2-T04 | Retenção de no-op | Identidade/bytes/seq/manifesto iguais antes e depois |
| PC2-T05 | Relatórios conflitantes | Falha com rollback e evidências preservadas |
| PC2-T06 | Divergência detectada e novo processo | Quarentena durável confirmada; nova execução recusada |
| PC2-T07 | Falha na gravação da quarentena | Fail-closed no mesmo processo e após reinício seguro |
| PC2-T08 | Backup durante manutenção concorrente | Rejeitar ou quiescer com prova, sem snapshot misto |
| PC2-T09 | Backup de PG + arquivos auxiliares | Inventário/hashes consistentes; restore isolado sem worker |
| PC2-T10 | Backup restaurado com UNKNOWN | Quarentena e recibo preservados, sem BUY/SELL |
| PC2-T11 | Papel SQL restrito e versão antiga | UPDATE direto bloqueado e versão incompatível recusada |
| PC2-T12 | Login sem usuário/senha, revogado e expirado | Sem acesso indevido a dados/admin |
| PC2-T13 | Artefatos antigos e padrões gitignore | Nenhum segredo/PII reintroduzido, evidências sanitizadas mantidas |
| PC2-T14 | Windows nativo vs Linux/WSL2 | Não anunciar suporte em ambiente com fcntl ausente |
| PC2-T15 | Worker offline e UI health anterior healthy | Indicador STALE/UNKNOWN, não “operando” |
| PC2-T16 | Falha de conexão ao banco | Nenhum START/trade, STOP não falsamente confirmado |
| PC2-T17 | STOP sob 5.000 eventos | Limites de 2 s STOP, 1,5 s lock e demais metas atendidos por medição relevante |
| PC2-T18 | Importação de dados legados | Nenhuma migração implícita nem contador de revisão falsamente confiável |

**Prova exigida:** executar pytest offline e PostgreSQL 16/18 descartáveis, falhas de conexão/PG restart, backup/restore, logs e evidência de diff; contar casos distintos separadamente de repetição por versão. Nenhuma aceitação baseada apenas neste documento.

## 7. Critérios de risco e decisão

| Situação | Decisão |
| --- | --- |
| P0 da retenção/quarentena aberto | **BLOQUEAR continuidade PAPER 72h** |
| STOP acima da meta com 5.000 eventos | **BLOQUEAR homologação longa** |
| Não há restore operacional testado | **BLOQUEAR recuperação como pronta** |
| Faltam provas de backup com writers quiescidos | **BLOQUEAR backup considerado seguro** |
| GitHub contém artefatos legados não sanitizados/verificados | **AUDITORIA PENDENTE**, nenhuma exclusão por impulso |
| Hardware ainda não medido | **ADIAR compra como requisito obrigatório para o robô** |
| Testes sintéticos aprovados e P0 encerrado | Reavaliar fase curta de testes, sem autorização automática para 72h |
| Qualquer sugestão de dinheiro real/alteração de broker/threshold | **PROIBIDO** |

## 8. Entrega e limites da autorização

**Entrega desta PC-2:** registro documental com achados, critérios, dependências e decisão de segurança. Não executar testes nem validar funcionamento real como parte desta escrita. Não baixar/inspecionar conteúdo potencialmente sensível de logs, dumps, arquivos de reset ou capturas para fora do repositório; a auditoria usa apenas metadados de diretório e código-fonte selecionado. Nenhum código operacional, schema, variável, Railway, banco remoto, main, PR, merge, deploy, sinal externo, provider, capital, ticket, score, threshold, min_signal_score ou broker real pode mudar.

A próxima missão efetiva quando Codex/Work estiverem disponíveis deve ser **INFRA-1.2.7**, com o requisito adicional de congelamento de manutenção/retention mesmo após STOP no contexto de backup. Uma futura fase PC-3 de backup operacional deverá ser planejada depois, e não misturada com P0 da integridade.
