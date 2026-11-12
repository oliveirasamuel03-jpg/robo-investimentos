# ATLAS — UI-1: auditoria estática das telas Streamlit e plano de melhoria

Data: 12/11/2026. Repositório exclusivo: `oliveirasamuel03-jpg/robo-investimentos`.
Branch de documentação: `atlas/ui-1-auditoria-estatica`.
Base inspecionada: `faf3fa26b6d8c9d8edf45e8b8e8177b54dcdad05` (INFRA-1.2.6 com contrato e critérios da INFRA-1.2.7).
Classificação: **READ_ONLY_AUDIT / UI_PLANNING_ONLY / PAPER_ONLY / NO_PRODUCTION_DEPLOY**.
**Nenhuma alteração nas telas, motores, configuração operacional ou infraestrutura foi autorizada por este documento.**

## 1. Parecer executivo

**Auditoria de código concluída com ressalvas; experiência visual NÃO homologada.**
Foram examinados `app.py`, `pages/0_Login.py`, `pages/1_Trader.py`, `pages/4_Controle_do_Bot.py`, `pages/5_Historico.py`, `core/ui_mutations.py`, `core/stop_control.py`, `core/auth/guards.py` e `core/auth/session.py`; também foram consultados os casos existentes em `tests/test_ui_mutations.py`.
Trata-se de **análise estática de fonte GitHub**: não houve inicialização do Streamlit, screenshots, clique em controles, teste responsivo em celular, consulta a banco real ou observação do worker. Todo achado de comportamento em tempo de execução exige validação automatizada futura antes de ser considerado reproduzido.

A estrutura já fornece login, permissões de administrador, painel Trader, Controle Operacional e Histórico. Componentes compartilhados distinguem STOP solicitado, persistido e reconhecido; reconhecem `UNKNOWN_COMMIT` e mensagens sobre a impossibilidade de repetir operações financeiras. O botão de **habilitar** real mode no Trader está desabilitado quando o modo real está desligado, e o callback recusa a habilitação; isso NÃO comprova a inexistência de outras configurações externas, mas indica que esta tela não oferece ativação normal de execução real.

O principal risco de UI não é uma cor ou estilo: é **comunicar “liberado”, “saudável”, “ativo” ou “concluído” sem explicitar o alcance da prova disponível**. A INFRA-1.2.6 identificou uma divergência de histórico cuja barreira financeira não fica duravelmente registrada; nenhum badge visual pode compensar essa falha do backend.

## 2. Inventário das telas

| Arquivo | Função atual | Acesso observado | Nota |
| --- | --- | --- | --- |
| `app.py` | Entrada com links e execução PAPER manual por admin | `require_auth()`, botão admin | Fluxo manual não compartilha as mensagens de resultado das páginas principais |
| `pages/0_Login.py` | Login e bootstrap | Usuário/senha e sessão | `auth_required()` é sempre verdadeiro no backend atual; ramo “modo dev opcional” é legado/inacessível no código estudado |
| `pages/1_Trader.py` | Trader, gráficos, comandos, análises e admin | `require_auth()`, mutações condicionadas a admin | ~3.566 linhas; mistura operação, diagnósticos shadow e configuração |
| `pages/4_Controle_do_Bot.py` | Controle administrativo, broker/feed/worker, diagnósticos e STOP | `require_admin()` | ~2.287 linhas; incorpora muitas seções de auditoria ao lado de comandos |
| `pages/5_Historico.py` | Ordens, relatórios, validação semanal e logs | `require_auth()` | Mensagem atual de retenção precisa ser compatível com o contrato financeiro real |

## 3. Achados priorizados e evidência de código

### UI-01 — ALTA: execução manual na Home sem feedback de commit padronizado
`app.py:29-43` chama `run_paper_cycle()` diretamente e exibe `result['report']`, `result['equity']` e `result['trades']`. Diferentemente de `pages/1_Trader.py:1669-1693` e `pages/4_Controle_do_Bot.py:2239-2268`, não usa `run_ui_mutation` nem `classify_cycle_feedback`. Não presumir que isso contorna a segurança do motor; o problema confirmado é **inconsistência no tratamento visual de sucesso, exceção e confirmação financeira**. Proposta: retirar a ação manual da Home ou encaminhá-la a um único handler PAPER auditável com confirmação de resultado; nenhum novo gatilho.

### UI-02 — ALTA: verde operacional por indicador parcial
`pages/4_Controle_do_Bot.py:243-255` mostra “Trava diária pronta: novas entradas liberadas neste momento” quando apenas o bloqueio **diário** está inativo. Isso NÃO verifica simultaneamente STOP, integridade, `UNKNOWN_COMMIT`, feed, broker ou versão dos escritores. O painel pode induzir um operador a confundir **limite diário livre** com **bot apto a operar**. Proposta: escrever “Limite diário de perda: não acionado; outros bloqueios podem impedir entradas” e reservar um status geral para agregação conservadora baseada em evidências confiáveis. Não construir autorização operacional apenas na UI.

### UI-03 — ALTA: texto de histórico faz afirmação excessiva
`pages/5_Historico.py:45-49` descreve retenção automática como “arquivados sem apagar histórico”. Entretanto, os contraexemplos da INFRA-1.2.6 (`tests/test_infra126_postgres.py:36-75`) descrevem reescrita de seq/payload, eliminação de `cycle_id` e perda de relatório conflitante por deduplicação. Não afirmar preservação completa enquanto as falhas P0/P1 estiverem abertas. Proposta: banner “Retenção em revisão de integridade; histórico exibido não equivale a auditoria aprovada”, condicionado ao estado real, nunca indicando que o problema foi corrigido.

### UI-04 — ALTA: telas de operação com ações de reset e mudança de configuração
`pages/1_Trader.py:3344-3376` inclui gravação de perfil, ticket, max_open_positions, watchlist e reset do Trader. `pages/4_Controle_do_Bot.py:2280-2284` expõe “Reset geral”. A execução é restrita a admin e respeita guardas existentes, mas **não há etapa de confirmação explícita nos trechos inspecionados**. Em fases PAPER de estudo, alterações de parâmetros críticos não são autorizadas pelo plano de projeto sem fase específica. Proposta: mostrar estado da fase/escopo, exigir confirmação intencional, preferir somente leitura ou ações desabilitadas até autorização formal; não alterar parâmetros em UI-1.

### UI-05 — MÉDIA: status de heartbeat pode aparentar zero segundos quando ausente
`pages/4_Controle_do_Bot.py:221-230` usa `production_state.get('heartbeat_age_seconds') or 0`. Quando o valor é `None` ou não registrado, a apresentação `0` pode ser confundida com heartbeat instantâneo. Proposta: exibir “Sem medição” para valor ausente, usar UTC e idade baseada em timestamp efetivo, e sinalizar `STALE/UNKNOWN` sem inventar estado online.

### UI-06 — MÉDIA: tela Controle atualiza estado durante renderização
`pages/4_Controle_do_Bot.py:175-186` chama `update_broker_status(probe_broker_status(...))`, `update_production_status(evaluate_production_health(...))` e `refresh_swing_validation_cycle()` a cada passagem por esse trecho. O código possui **efeitos de escrita/avaliação durante o render**, sem acionamento explícito de botão. Não foi medido custo de requests nem duração de locks. Proposta: estudar separação de leitura de status e atualização explícita com TTL seguro, sem modificar o feed, o worker ou a frequência operacional.

### UI-07 — MÉDIA: fontes de verdade e indicadores pouco diferenciados
`pages/1_Trader.py:1593-1612` exibe “PAPER Ativo” fixo, independentemente da verificação atual do broker. O painel Controle traz “Modo produção” e indicador de saúde (`pages/4_Controle_do_Bot.py:204-239`), enquanto os detalhes sobre modo efetivo do broker ficam em expander (`2188-2200`). Proposta: separar três coisas: **regime autorizado (PAPER obrigatório)**, **modo efetivo observado (com fonte/horário)** e **prontidão financeira (somente depois de backend verificado)**. Divergência deve ser crítica; indisponibilidade deve ser UNKNOWN, não verde.

### UI-08 — MÉDIA: STOP/reconciliação dependem de recarregamento para nova leitura
`pages/1_Trader.py:1637-1666` e `pages/4_Controle_do_Bot.py:2211-2216` rendem badges antes dos comandos; alguns botões disparam `st.rerun()`, mas STOP não força no mesmo trecho a atualização de todos os indicadores já renderizados. Não afirmar que a parada falhou; o problema potencial é **estado visual antigo no restante da interação**. Proposta: apresentar recibo do comando imediatamente, botão “Atualizar estado confirmado”, distinguir request/PERSISTED/ACK e testar reruns com Streamlit AppTest.

### UI-09 — MÉDIA: feedback de reconciliação pode omitir que a liberação não foi registrada
`core/ui_mutations.py:182-215` verifica `resolution_recorded` explicitamente para `COMMITTED`, mas o ramo `NOT_COMMITTED` (`210-212`) não apresenta essa mesma distinção. Quando há evidência positiva porém falha na gravação da resolução (`core/commit_protocol.py:331-339`), o operador precisa ler claramente que **a liberação não foi confirmada**, mesmo que o resultado financeiro seja conhecido. Proposta: sempre exibir separadamente `verified_outcome`, `resolution_recorded` e `blocked`. Nenhum texto de UI pode liberar a barreira.

### UI-10 — BAIXA: resíduos de diagnóstico e complexidade de navegação
`pages/4_Controle_do_Bot.py:171-173` mostra `BUILD_UI_MARKER_20260422_A`; várias seções de estudo/sinais/swing estão na mesma tela dos comandos (mais de 2.000 linhas). O Trader também excede 3.500 linhas. Proposta: informação de build em expander técnico; seção operacional curta no topo, diagnóstico em abas/expanders; sem remover métricas de auditoria nem transformar shadow em gatilho.

### UI-11 — BAIXA: ramo de autenticação dev obsoleto no Login
`pages/0_Login.py:27-33` apresenta caminho de “autenticação opcional” se `not auth_required()`, mas `core/auth/session.py:40-45` sempre retorna `True` no código inspecionado. Isso é **código legado não alcançável**, não uma brecha de login comprovada. Proposta: remover o ramo morto em fase de UI futura, preservando obrigatoriedade de login e os testes de acesso.

### UI-12 — LIMITAÇÃO: ausência de homologação visual/uso móvel
Sem execução Streamlit, screenshots e navegação no Android/iOS, não há prova sobre responsividade, contraste, carga de páginas, comportamento do cache, mensagens completas ou compatibilidade dos componentes. Ficam PENDENTES até ambiente local seguro. Nenhuma auditoria estática deve alegar aprovação visual.

## 4. Proteções existentes que devem permanecer intactas

- `core/auth/guards.py:27-44`: `require_auth()` e `require_admin()`. Não retirar autenticação nem ampliar permissões por facilidade visual.
- `core/auth/session.py:40-45`: autenticação obrigatória inclusive para ambiente local; não habilitar dev-bypass.
- `core/ui_mutations.py:26-66`: erros de conflito sem retry, mensagens cautelosas, sem imprimir exceções com segredos.
- `core/ui_mutations.py:79-135`: STOP recebido ≠ STOP persistido ≠ STOP reconhecido.
- `core/ui_mutations.py:182-215`: botão de reconciliação não opera trades e preserva feedback `UNKNOWN_COMMIT`.
- `pages/1_Trader.py:3524-3565`: botão de habilitar real mode está desabilitado quando o modo real está desligado; callback também impede ativação.
- Preservar sinais shadow/study/diagnostic como informativos, nunca operacionais.

## 5. Proposta de informação principal (somente planejamento)

Priorizar o topo do Trader e do Controle com um quadro legível de evidências, sem comandos novos:

1. **Regime:** PAPER obrigatório (configuração autorizada); nunca confundir com broker efetivo.
2. **Controle:** RUNNING / PAUSED / STOP REQUESTED / STOP PERSISTED / STOP ACKNOWLEDGED / UNKNOWN, sempre com fonte e timestamp.
3. **Integridade:** VERIFIED CURRENT / PENDING INTENT / UNKNOWN COMMIT / QUARANTINE / UNAVAILABLE. Não exibir “VERIFIED CURRENT” se o backend apenas informar ausência de `active_cycle_id`; essa funcionalidade depende de correção da INFRA-1.2.7.
4. **Infraestrutura:** worker heartbeat e idade, feed operacional versus feed de gráfico, provider efetivo, broker efetivo PAPER, banco e versão do escritor (quando efetivamente verificáveis).
5. **Resultado financeiro:** COMMITTED / NOT_COMMITTED / UNKNOWN_COMMIT / POST_COMMIT_DIAGNOSTIC_FAILURE separados do estado STOP; jamais misturar contabilização e diagnóstico.
6. **Pendências:** número e severidade dos bloqueios, último erro conhecido e instrução segura de próxima ação. Sem link, botão ou automação para reprocessar o trade.

Cores apenas como reforço: **UNKNOWN cinza/aviso**, **BLOCKED vermelho**, **PENDING âmbar**, **CONFIRMED** somente com prova. Ausência de erro não é sinal de saúde.

## 6. Plano de execução futura e critérios de aceitação

**UI-1 (esta etapa):** inventário e especificação estática somente. Não modificar `app.py`, `pages/`, `core/`, `workers/` ou testes. Nenhuma liberação operacional.

**Dependência P0 obrigatória:** INFRA-1.2.7 deve eliminar quarentena não durável e retenção destrutiva antes de adicionar badge “integridade verificada”. Testes de UI não substituem os testes PostgreSQL dessa fase.

**UI-2 posterior (nova branch, fase explícita):** implementar apenas mensagens e leitura, com testes AppTest/pytest e sem mudanças em motores/regras:
- T-UI01: Home não apresenta resultado financeiro como confirmado sem protocolo; execução manual única/padronizada ou removida.
- T-UI02: trava diária livre + gate UNKNOWN produz aviso de bloqueio geral, nunca “pode operar”.
- T-UI03: ausência de timestamp de heartbeat mostra “Sem medição”, não “0 segundos”.
- T-UI04: STOP recebido/persistido/ACK demonstram mensagens distintas sob re-render, falha de conexão e restart.
- T-UI05: NOT_COMMITTED provado + resolução não registrada mantém “liberação pendente”.
- T-UI06: usuário comum não vê/administra ações sensíveis; admin só com confirmação e fase permitida.
- T-UI07: retenção não ostenta histórico íntegro quando houve divergência reconhecida ou quando backend não tem prova atual.
- T-UI08: modo real não habilitável; broker efetivo não divergente da política PAPER.
- T-UI09: diagnóstico e shadow continuam sem autoridade de trade; listas extensas legíveis sem apagar observabilidade.
- T-UI10: reload da página de Controle não dispara mutações desnecessárias de estado/feed, ou registra justificativa e teto mensurado para as inevitáveis.
- T-UI11: autenticação obrigatória, sessão expirada, RBAC e usuário desabilitado testados.
- T-UI12: celulares e desktop testados em ambiente local, sem pressupor layout responsivo.
- T-UI13: todos os controles perigosos com teste de cancelamento/duplo clique e sem retry financeiro.

Exigir a suíte de regressão então vigente, sem mascarar skips/warnings, e provas de que `main`, Railway, variáveis, DB remoto, provider, broker e thresholds permanecem inalterados. Não criar PR, merge ou deploy automaticamente; não executar worker contínuo.

## 7. Riscos e impedimentos

**ALTO:** defeito de integridade da INFRA-1.2.6 aberto; execução contínua bloqueada.
**ALTO:** 5.000 eventos ainda excedem metas de STOP/lock; nenhuma mudança visual resolve esse desempenho.
**PENDENTE:** testar efeitos de atualização das telas, limites de requests, tempo de resposta, acessibilidade, uso móvel, testes reais de screenshots.
**PROIBIDO:** usar “apto”, “live”, “saudável” ou “PAPER ativo” como substituto de prova de broker/worker/ledger; reativar real mode; avançar FASE 2.6C; operar dinheiro real.

## 8. Status e próximos passos

**STATUS: AUDITORIA ESTÁTICA DOCUMENTADA; IMPLEMENTAÇÃO UI NÃO INICIADA.**
Primeiro fechar INFRA-1.2.6 com logs e corrigir INFRA-1.2.7 em Codex quando estiver disponível. Em paralelo, a especificação UI-2 pode ser refinada nesta conversa; testes e mudanças operacionais esperam fase explícita.

Esta documentação não aprova 72 horas, não contém alteração do app em execução e não implica custo externo.
