# ATLAS — UI-2: especificação de segurança e experiência operacional Streamlit

**Data:** 12/11/2026. **Estado:** SPEC_ONLY / PLANNING_ONLY. **Execução:** nenhuma alteração de runtime nesta entrega.

Repositório: `oliveirasamuel03-jpg/robo-investimentos`.
Branch de trabalho: `atlas/ui-2-especificacao-seguranca`.
Base: `atlas/ui-1-auditoria-estatica`, commit `b7790fe70626a46b2e0c053602f8327825496df2`.
Documento de origem: `docs/retomada/ATLAS_UI_1_AUDITORIA_TELAS_STREAMLIT.md`.

**Invariantes:** PAPER_ONLY, um único worker futuro, nenhum broker real, nenhuma ordem real, nenhum avanço FASE 2.6C, nenhuma alteração de estratégia, score, thresholds, min_signal_score, capital, ticket, max_open_positions, provider ou autoridade shadow sem fase específica. Nada de main, PR, merge, Railway, banco remoto, volume, env remoto, deploy, worker contínuo, serviço pago ou migração nesta fase.

## 1. Decisão de arquitetura UI

A interface **não é fonte de verdade** para integridade financeira, STOP, feed, heartbeat ou broker. Ela somente apresenta fatos confirmados por APIs de leitura existentes e identificados pela origem/horário. Um estado desconhecido não significa sucesso. A aprovação da **INFRA-1.2.7 é pré-requisito para exibir qualquer indicador de integridade geral como “verificada”**; antes disso o texto honesto é “Validação financeira pendente / execução prolongada bloqueada”.

A UI-2 posterior será **somente apresentação e interação**, preservando contratos de persistência. Mudança de permissão PostgreSQL, gate financeiro, stop_control, retenção ou engine pertence a outra fase e não pode ser escondida em um PR visual.

Ordem de aplicação futura: P0 backend/integridade -> revisão Atlas -> UI somente-leitura/mensagens -> AppTest e revisões de acesso -> homologação visual em desktop/celular. A especificação pode ser desenvolvida em paralelo; implementação que afete execução aguarda backend corrigido.

## 2. Hierarquia de informação

### 2.1 Página inicial: painel de situação, não executor secundário

Abertura do app deve mostrar: modo autorizado PAPER, estado de controle com nível de confirmação, estado financeiro (somente se validável), saúde da infraestrutura, horário da última observação e ações de navegação. O botão `Rodar ciclo` atualmente em `app.py:33-43` gera um segundo ponto de execução manual sem o mesmo feedback financeiro das páginas Trader/Controle. **Opção preferida:** retirar o botão da Home em futura fase de UI, mantendo um link para Trader/Controle. Alternativa somente se a auditoria aprovar: reutilizar um único controlador auditável com recibos, permissões e feedback idênticos. Não criar qualquer gatilho financeiro novo.

### 2.2 Trader: operação e observabilidade com distinção de autoridade

Topo: aviso de PAPER obrigatório; estado STOP (pedido/persistido/ACK) e bloqueio financeiro; worker e feed, cada um com fonte própria. Depois: gráficos e ordens PAPER. A seguir: análises shadow/diagnósticos, visualmente marcadas `SOMENTE ESTUDO — SEM AUTORIDADE DE TRADE`; parâmetros sensíveis apenas com permissão e fase explicitamente autorizada.

### 2.3 Controle do Bot: comandos isolados dos estudos

Topo compacto com fatos confirmados e ações explícitas, mantendo STOP acessível mesmo com UNKNOWN_COMMIT. Secção separada com histórico/diagnósticos extensos e detalhes de broker. Um clique de STOP não deve ser renderizado como ACK; retorno `PERSISTED` significa apenas solicitação gravada. `START` exige estado persistido/versão/revisão conforme backend, não apenas o snapshot desenhado na tela.

### 2.4 Histórico: fatos imutáveis e lacunas declaradas

Ordens, trades e relatórios precisam deixar claro o que é ledger confirmado, tabela de apresentação, relatório semanal, arquivo de retenção e resultado não reconciliado. A frase atual em `pages/5_Historico.py:45-49` que garante dados antigos “arquivados sem apagar histórico” deve ser substituída após revisão por formulação que **não prometa integridade não demonstrada**. Não eliminar históricos nem corrigir conflitos via interface.

### 2.5 Login e sessão

Autenticação permanece obrigatória em qualquer ambiente, conforme `core/auth/session.py:40-45`. Somente administradores podem executar comandos sensíveis. O ramo `auth_required() == False` no Login é legado e não deve ser reativado. Evitar expor credenciais, caminhos internos, stack traces ou segredos em erros exibidos ao operador.

## 3. Contratos de exibição de estado

| Indicador | Exibição segura | Fonte exigida / exceção |
| --- | --- | --- |
| Regime permitido | `PAPER OBRIGATÓRIO` | Política do projeto, não prova de conexão |
| Broker efetivo | `PAPER confirmado / NÃO CONFIRMADO / DIVERGENTE` | Diagnóstico do broker com timestamp; divergência é bloqueio, nunca verde |
| Controle STOP | `RECEIVED / PERSISTED / ACKNOWLEDGED / UNCONFIRMED` | Recibo persistido e confirmação do worker; `STOPPED` isolado não é ACK |
| Bot | `RUNNING / PAUSED / STOPPED / DESCONHECIDO` | Estado do bot com horário; `RUNNING` não comprova worker vivo |
| Resultado financeiro | `COMMITTED / NOT_COMMITTED / UNKNOWN_COMMIT / POST_COMMIT_DIAGNOSTIC_FAILURE` | Recibo de ciclo e resultado de verificação explícita |
| Bloqueio financeiro | `PENDENTE / BLOQUEADO / DIVERGÊNCIA / INDISPONÍVEL` | Gate e quarentena verificáveis no banco; ausência de active_cycle_id **não** é certificação do ledger |
| Reconciliação | `RESULTADO VERIFICADO` e `LIBERAÇÃO REGISTRADA` independentes | `resolution_recorded` + `blocked`; recibo histórico não prova estado atual |
| Heartbeat | `Há N segundos` ou `SEM MEDIÇÃO` | Timestamp não nulo UTC; nunca renderizar `None` como `0` |
| Feed operacional | `LIVE / DELAYED / FALLBACK / UNKNOWN` | Fonte worker, separada da cotação/gráfico |
| Feed de gráfico | Rótulo de fonte visual independente | `LIVE` no gráfico não significa feed confiável do worker |
| Trava diária | `LIMITE NÃO ACIONADO / ACIONADO` | Não equivale a liberação geral para entrar |
| Saúde geral | `BLOQUEADA / DESCONHECIDA / OBSERVAÇÃO / VERIFICADA` | Uma agregação conservadora só depois de um contrato completo e testado; não inferir pela ausência de erro |

**Regra de prioridade:** se qualquer fonte necessária falhar ou for stale, não exibir `APTO PARA OPERAR`. Durante INFRA-1.2.6/1.2.7, o dashboard não pode declarar integridade financeira verificada.

## 4. Catálogo de mensagens aprovadas como especificação

Cópias propostas, não implementadas:

- **STOP RECEIVED:** “Solicitação recebida; persistência ainda não confirmada. Não considere o robô parado.”
- **STOP PERSISTED:** “Parada registrada. A confirmação do worker ainda está pendente; um commit anterior não é cancelado.”
- **STOP ACKNOWLEDGED:** “Parada reconhecida pelo worker. Identificador e horário confirmados: {id}/{UTC}.”
- **STOP UNCONFIRMED:** “Não foi possível verificar a parada. Consulte o estado persistido e investigue a conexão.”
- **UNKNOWN_COMMIT:** “Resultado financeiro desconhecido. Novos ciclos bloqueados até reconciliação explícita. Não repita BUY/SELL.”
- **COMMITTED:** “Transação PAPER confirmada no PostgreSQL; o resultado exibido pertence ao ciclo {id}.”
- **NOT_COMMITTED:** “Ausência da transação comprovada; isso não autoriza repetição automática.”
- **POST_COMMIT_DIAGNOSTIC_FAILURE:** “Transação PAPER confirmada, diagnóstico posterior falhou. Não execute novamente.”
- **VERIFIED_BUT_NOT_RELEASED:** “Resultado financeiro verificado, mas liberação do bloqueio não confirmada. Execução permanece restrita.”
- **INTEGRITY_DIVERGENCE:** “Divergência financeira identificada; quarentena e investigação necessárias. Não use reset para liberar.”
- **INTEGRITY_NOT_HOMOLOGATED:** “Integridade financeira ainda não homologada para operação contínua.”
- **HEARTBEAT_MISSING:** “Sem medição de heartbeat. Não é possível confirmar worker online.”
- **DAILY_LOSS_FREE:** “Limite diário não acionado. Outros bloqueios podem impedir entradas.”
- **PAPER_POLICY:** “Regime autorizado: PAPER. Nenhuma ordem real é permitida.”
- **RETENTION_UNDER_REVIEW:** “A política de retenção ainda está sob revisão de integridade; não presuma que este histórico esteja reconciliado.”

Mensagens nunca expõem DSN, token, SQL interno, tracebacks, hashes sensíveis ou login de terceiros. Não criar sucesso otimista antes de readback.

## 5. Priorização e escopo de UI-2 posterior

**P0 (depende da INFRA-1.2.7):**
1. Separar indicador financeiro de gate e integridade efetivamente validada.
2. Normalizar feedback de `resolution_recorded` para TODOS os outcomes, especialmente NOT_COMMITTED.
3. Remover a promessa de retenção íntegra antes da correção correspondente.
4. Não apresentar aprovação geral se STOP, gate, feed, broker ou worker não puderem ser confirmados.

**P1 (implementação visual separada):**
1. Remover execução manual duplicada da Home ou centralizar explicitamente sem novos privilégios.
2. Corrigir heartbeat ausente `or 0` em `pages/4_Controle_do_Bot.py:223`.
3. Diferenciar “PAPER por política” de “broker PAPER efetivo”.
4. Concentrar STOP e bloqueios no topo; não misturar diagnóstico shadow com comandos.
5. Substituir aviso de trava diária livre que parece autorização geral de entrada.

**P2 (depois de testes de acessibilidade):**
1. Reduzir complexidade do Trader (>3.500 linhas) e Controle (>2.200), mediante pequenas funções/componentes de UI sem alterar engines.
2. Retirar `BUILD_UI_MARKER_20260422_A` da tela principal e colocar informações técnicas em expander.
3. Revisar layout em celular, contrastes, rótulos, rolagem, tempo de render e estado do operador após rerun.
4. Atualização de saúde/validação durante render em `pages/4_Controle_do_Bot.py:175-186` exige medir efeitos laterais antes de simplificar.

**Restrições de UX:** `Reset geral`, `Resetar módulo trader`, troca de perfil/ticket/max_open_positions/provider e `START` só podem permanecer acionáveis em fase autorizada, com revisão de estado, confirmação e testes de RBAC. Nenhuma mudança de regra operacional entra como refactor visual.

## 6. Matriz de testes futuros de interface

| ID | Cenário de AppTest/integração | Critério |
| --- | --- | --- |
| UI2-01 | Home com admin e com usuário comum | Nenhuma ação manual insegura; navegação correta |
| UI2-02 | `UNKNOWN_COMMIT` persistido | Banner bloqueado; sem incentivo a retry financeiro; STOP visível |
| UI2-03 | `NOT_COMMITTED` com `resolution_recorded=False` | Resultado e liberação distinguíveis, sem falso sucesso |
| UI2-04 | `COMMITTED` com falha diagnóstica | Sem atribuir rollback ou repetir trade |
| UI2-05 | STOP received/persisted/ACK/timeout | Mensagens distintas; sem ACK falso |
| UI2-06 | STOP acionado após render da página | Feedback de recibo imediato e estado reconsultável; sem stale success |
| UI2-07 | Heartbeat ausente, inválido ou antigo | UNKNOWN/sem medição, nunca “0 s online” |
| UI2-08 | Feed visual LIVE e worker FALLBACK | Nenhum selo de feed operacional confiável |
| UI2-09 | Trava diária não acionada e gate financeiro fechado | Sem “novas entradas liberadas” geral |
| UI2-10 | Broker efetivo desconhecido ou divergente | Sem afirmar PAPER efetivo confirmado |
| UI2-11 | Usuário não admin/expirado/desabilitado | Comandos protegidos e sem escalada de role |
| UI2-12 | Real mode desabilitado | Nenhum caminho UI habilita real; testes de callback e botão |
| UI2-13 | Reset e configuração sensível | Confirmação/cancelamento; sem alteração indevida ou replay após rerun |
| UI2-14 | Histórico com retenção anômala | Não afirmar conservação/integração de dados sem prova |
| UI2-15 | Teste de dupla ação/clique e exceção pós-commit | Nenhum retry automático; recibo idêntico |
| UI2-16 | Diagnóstico shadow e sinal externo | Nunca viram autoridade de trade |
| UI2-17 | Celular vertical e desktop | Informação crítica legível antes da rolagem, sem depender só de cor |
| UI2-18 | Renderização de Controle | Ausência de alterações financeiras ou mudança de ciclo apenas por abrir/recarregar a tela |

Execução real dos testes: **PENDENTE**. A cobertura anterior em `tests/test_ui_mutations.py` não comprova todos os cenários acima. Não inventar métricas de desempenho visual ou acessibilidade.

## 7. Dependências, limites e aceitação

1. Corrigir primeiro P0-01/P1 da INFRA-1.2.6 em fase corretiva com PostgreSQL local (INFRA-1.2.7); isso inclui retenção que destrói identidade e divergência sem quarentena durável.
2. Antes da implementação, revisão de segurança da mudança de interface para garantir que nada muda em broker, score, thresholds, capital, ticket, max_open_positions ou provider. Se qualquer mudança for necessária, separar fase e autorização.
3. Implementação futura somente em branch nova, sem editar main diretamente. Testar no mínimo os fluxos UI2-01 a UI2-18, preservar regressão e coletar evidência AppTest e screenshots em ambiente local próprio.
4. **Critério de bloqueio:** qualquer mensagem que confirme STOP, commit, operação autorizada, integridade ou worker online sem fonte persistida verificável. Erros de acesso são críticos.
5. Não liberar 72 horas, dinheiro real, deploy, Railway, banco remoto ou trabalhador contínuo com base em aparência/copy. A UI nunca pode tornar um backend inseguro seguro.

## 8. Entrega desta fase

Documentação de decisão e plano de implementação apenas. Não foram alterados `app.py`, `pages/`, `core/`, `workers/`, `paper_trading_engine.py`, banco, configuração de trading, testes existentes ou `main`. Sem PR, merge, deploy, conta paga, Railway ou dados reais.

**Próximo executor quando disponível:** Codex para a INFRA-1.2.7, seguido de auditoria e, depois, UI-2 de execução em branch específica. Até lá, manter PAPER e homologação prolongada BLOQUEADA.
