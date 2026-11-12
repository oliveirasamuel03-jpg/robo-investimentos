# ATLAS — UI-3: especificação de monitoramento e diagnóstico PAPER com prioridade móvel

Data: 12/11/2026.
Repositório exclusivo: oliveirasamuel03-jpg/robo-investimentos.
Branch: atlas/ui-3-monitoramento-mobile-spec.
Base: atlas/ui-2-especificacao-seguranca, commit c731999722a4e4bc968b068161c3273a1b8ed15b.

**Status: PLANNING_ONLY / READ_ONLY_DESIGN / PAPER_ONLY / COST_CONTROLLED / NO_PRODUCTION_DEPLOY.**
Este documento **não implementa** um painel, não inicia servidor, worker, rede, e-mail, app móvel, broker ou banco e não autoriza homologação de 72 horas.

## 1. Objetivo e não objetivos

Preparar uma vista simples, confiável, acessível pelo celular *quando existir um ambiente local seguro*, para o operador entender:
1. se o **regime permitido** é PAPER;
2. o **estado efetivo observado** do controle, worker, banco, broker e feed;
3. se uma intenção financeira está **bloqueada, desconhecida ou confirmada**;
4. quando cada evidência foi observada;
5. qual é a **próxima ação de diagnóstico permitida**, sem retries financeiros.

Não objetivos: criar aplicativo nativo, habilitar dinheiro real, rodar worker em celular, abrir porta pública, gerar sinais, alterar estratégia, contratar serviço, medir rentabilidade ou implementar alertas externos nesta fase.

**P0 bloqueante:** a INFRA-1.2.6 revelou perda potencial de provas na retenção e divergência reconciliada sem barreira persistida quando gate já estava livre. Até correção e testes da INFRA-1.2.7, a UI **nunca pode declarar integridade financeira global APROVADA** ou “pronto para operar”, mesmo se o gate existente estiver vazio.

## 2. Evidência de código inspecionada

| Origem | Campos e funções existentes | Limitação |
| --- | --- | --- |
| core/production_monitor.py:37-160 | heartbeat_age_seconds, evaluate_production_health, worker_status, health_level, last_health_at, consecutive_errors, feed_status, broker_status | Produz **saúde operacional**, não autorização financeira. heartbeat ausente retorna None; health=healthy não prova ledger íntegro |
| core/state_store.py:272-299, 780-799 | worker_heartbeat, feed_status UNKNOWN, provider_effective, production.heartbeat_age_seconds=None e production.health_level default | Defaults não são prova da leitura atual ou de worker online |
| workers/trader_worker.py:1848-1973 | worker_loop, atualizações runtime e monitor, gate_status, reconhecimento STOP, logs de falhas | Worker offline pode deixar último heartbeat/health persistido; sem observação atual não inferir que está ativo |
| core/market_data.py:47-50,190-239 | LIVE, DELAYED, FALLBACK, UNKNOWN; snapshot feed_quality e provider_effective | Feed do gráfico pode ser diferente do feed operacional |
| core/broker.py:87-139 | probe_broker_status, status, effective_mode, last_sync_at, execution_enabled, can_submit_orders | Probe não é prova de trade nem licença para mudar broker; não disparar de forma repetida por renderização |
| core/stop_control.py:23-43 | request_id, persisted, acknowledged, cycle_activity | STOP recebido, persistido e ACK são eventos diferentes; só ACK verifica reconhecimento do worker |
| core/commit_protocol.py:97-115,235-339 | gate_status, reconcile, UNKNOWN_COMMIT, active_cycle_id, recibo | Gate aberto por si só não atesta integridade. Falha P0 da INFRA-1.2.6 permanece aberta |
| core/ui_mutations.py:79-135,182-215 | render_stop_status e render_commit_control | Não repetir reconciliação automaticamente; status de liberação precisa considerar resolution_recorded |
| pages/4_Controle_do_Bot.py:175-240 | saúde, heartbeat, broker, worker e renderizações com updates | Exibição atual pode ter efeitos de escrita durante render; novo dashboard deve ser somente leitura |
| pages/5_Historico.py:37-52 | relatórios e mensagens de retenção | Não afirmar arquivo íntegro diante dos contraexemplos da INFRA-1.2.6 |

Todas as referências são observações estáticas do GitHub. **Não houve execução de Streamlit nem medição de latência de celular** nesta fase.

## 3. Modelo de visualização para tela estreita

Largura de referência para projeto: aproximadamente 360–430 px, não tamanho homologado; responder também em tablet e desktop.

Ordem de prioridade de cima para baixo, sem exigir scroll para conhecer bloqueios críticos:
- Barra superior: “Robô Investimento”, subtítulo “MONITORAMENTO PAPER”, momento da leitura com fuso visível e botão **Atualizar leitura** (somente leitura).
- Faixa principal de segurança: “OPERAÇÃO CONTÍNUA NÃO HOMOLOGADA”, quando INFRA-1.2.7/performance/72h pendentes.
- Estado STOP (pedido, persistido, ACK, não confirmado).
- Integridade financeira: gate/receipt em destaque **sem** badge verde de ledger aprovado; estado incerto = bloqueio e orientação para investigação.
- Worker: online reportado, heartbeat há N segundos, última execução e falhas; ausência/stale = “não confirmado”.
- Dados de mercado: feed **operacional do worker**, provider efetivo, último sucesso; feed do gráfico em linha separada.
- Broker: regime permitido PAPER, modo efetivo observado e hora do probe, mais alerta em caso de divergência.
- Retenção e armazenamento: última tentativa, último erro, “auditoria de histórico pendente”, sem validar dados implicitamente.
- Diagnóstico em detalhes expansíveis: últimos eventos com tempo/fonte, IDs mascarados, build/versão/writer apenas se atestado.
- Histórico PAPER resumido só depois dos bloqueios: contagem de trades/eventos e relatório, jamais como indicador de lucro esperado.

Princípios: uma coluna no celular; texto legível sem zoom; ícones com legenda e cor não como único sinal; títulos curtos; cards acessíveis pelo leitor de tela; nenhum painel gigante de estudos acima dos bloqueios; atualização da vista não muda estado financeiro.

### Wireframe textual (não executável)

ROBÔ INVESTIMENTO — MONITORAMENTO PAPER

**PAPER OBRIGATÓRIO | 72H NÃO LIBERADAS**

CONTROLE: STOP PERSISTIDO · ACK PENDENTE
INTEGRIDADE: NÃO HOMOLOGADA · INVESTIGAÇÃO NECESSÁRIA
WORKER: SEM LEITURA ATUAL
FEED OPERACIONAL: DESCONHECIDO
BROKER EFETIVO: NÃO CONFIRMADO
ÚLTIMA VERIFICAÇÃO: [UTC / horário local, com origem]

[ATUALIZAR LEITURA — SEM OPERAR]

Diagnóstico detalhado ▾
Histórico PAPER ▾
Política de acesso e segurança ▾

Valores do wireframe são **exemplos de cópia segura**, não estado lido do aplicativo.

## 4. Contrato de fonte, atualidade e confiança

Cada bloco de estado na futura UI deverá transportar:
- `source`: namespace/protocolo concreto, não uma inferência;
- `observed_at_utc`: timestamp real de leitura;
- `event_at_utc`: timestamp do evento original quando houver;
- `read_status`: OK, MISSING, INVALID, UNAVAILABLE, STALE;
- `value`: enum ou número validado;
- `authority`: DISPLAY_ONLY, CONTROL_RECEIPT, FINANCIAL_EVIDENCE, POLICY;
- `validation_note`: quais verificações NÃO foram realizadas.

Formato acima é **contrato proposto**, não novo schema nem API já implementada. Não gravar esses campos em app_state nem adicionar tabela sem uma fase específica.

**Snapshots inconsistentes:** leituras separadas do bot_state, gate, broker e feed podem representar momentos distintos. A interface não pode compor “tudo aprovado” somando registros de instantes diferentes. Estudar uma leitura consistente/snapshot de leitura somente após a correção do backend, sem usar o protótipo de snapshot transacional rejeitado na INFRA-1.2.5 como prova financeira. Na ausência dessa garantia, mostrar timestamps independentes e estado combinado DESCONHECIDO.

**Sem efeito colateral:** leitura do painel não chama `run_trader_cycle`, `run_paper_cycle`, `start_robot`, `reconcile`, `reset_state`, probe remoto, `update_production_status` ou `refresh_swing_validation_cycle` automaticamente. Não usar funções load que persistam defaults sem verificar; eventual adaptador read-only terá seus próprios testes.

**Validade temporal:** idade de heartbeat é calculada de timestamp UTC; `None`, string inválida e horário futuro suspeito não são idade zero. Estabelecer limites de stale pela frequência real do worker numa fase de homologação, não inventar agora. Renderizar explicitamente “última leitura há N segundos” apenas com tempo computável.

## 5. Máquina de apresentação e precedência conservadora

A apresentação do monitor **não decide trades**. Prioridade visual proposta:
1. DIVERGÊNCIA / política PAPER violada / operação real observada -> **ALERTA CRÍTICO**, não autoriza operar;
2. UNKNOWN_COMMIT ou barreira de integridade durável -> **BLOQUEIO FINANCEIRO**; STOP sempre acessível aos operadores autorizados;
3. Banco, recibo, controle ou timestamp inacessível -> **NÃO CONFIRMADO** e orientação de diagnóstico, nunca “livre”;
4. STOP apenas PERSISTED -> **PARADA REGISTRADA, ACK PENDENTE**;
5. STOP ACKNOWLEDGED -> **PARADA RECONHECIDA** (não significa lucro, feed ou ledger aprovado);
6. Worker heartbeat antigo, feed fallback/delayed, broker não confirmado -> **OBSERVAÇÃO / DEGRADAÇÃO**;
7. Operação normal só pode ser exibida após critérios futuros de backend, prova de integridade e homologação; até lá **MODO DE ESTUDO — NÃO HOMOLOGADO**.

Em particular, `production.health_level == healthy`, `bot_status == RUNNING`, `daily_loss_block_active == false` ou `gate_status.blocked == false` **isoladamente jamais autorizam o selo “pode operar”**.

Não misturar:
- PAPER permitido vs PAPER efetivamente observado;
- worker/status “online” armazenado vs processo vivo;
- feed visual LIVE vs feed operacional LIVE;
- STOPPED vs STOP ACK;
- COMMITTED vs reconciliação registrada/liberação;
- BACKUP restaurado vs integridade reparada;
- saúde técnica vs desempenho lucrativo.

## 6. Alertas e ações

**Fase de implementação futura inicial: READ_ONLY.** Botão de atualizar a leitura não reexecuta consulta ao mercado, trades, reset, STOP, START ou reconciliação. STOP remoto por celular **não está autorizado nesta especificação**. Quando autorizado em fase separada, exigir autenticação admin atual, contexto seguro, mensagem de recibo, ausência de retry automático, proteção contra duplo clique e teste real.

Alertas visíveis locais (sem notificações push, SMS ou e-mail novo nesta fase):
- resultado financeiro UNKNOWN_COMMIT;
- bloqueio/quarentena de integridade;
- STOP aguardando ACK;
- worker/heartbeat não confirmado;
- feed operacional em fallback ou stale;
- broker divergente de PAPER;
- retenção falha, manifestos inconsistentes;
- falhas consecutivas no worker;
- incongruência de versão do writer, quando verificação futura existir.

Para cada alerta: severidade, origem, timestamp, o que foi verificado e próxima ação segura **não financeira**. Não expor senhas/tokens/DSN, IDs de conta completos, dados de outros usuários, traceback ou URLs sensíveis. Não compartilhar links públicos do painel.

## 7. Frequência e controle de custos

Preferir **atualização manual** no primeiro protótipo. Se um dia houver auto-refresh, propor teto de requests/TTL e medição de consumo por sessão antes de ativá-lo. Não executar consultas pesadas de manifesto financeiro em cada rerun, nem iniciar workers/probes/reminders contínuos para a tela.

Arquitetura local futura candidata: Streamlit existente + PostgreSQL local + um worker PAPER + acesso privado autenticado e autorizado ao painel. Não escolher túnel público, hosting, exposição na internet ou acesso ao PC do usuário agora. Acessibilidade pelo celular depende de infraestrutura e revisão de segurança que ainda não existem; este documento não resolve conectividade remota.

Limites de custo devem ser documentados após medir CPU, disco, memória e queries reais. Nada de Railway, serviços pagos, banco externo ou notificações externas por este planejamento.

## 8. Sequência de implementação sugerida

**UI-3.0 (esta entrega):** arquitetura documental, sem arquivos de runtime.
**Dependência obrigatória:** fechamento dos defeitos críticos da INFRA-1.2.6 na fase corretiva INFRA-1.2.7 com provas PG18/16. STOP com 5.000 eventos ainda exige otimização segura separada e auditoria.
**UI-3.1 futura:** funções *puras* de mapeamento do status para apresentação, com testes unitários de origem/tempos/precedência; sem banco.
**UI-3.2 futura:** Streamlit responsivo e read-only, reaproveitando campos já existentes, sem nova infraestrutura; AppTest com RBAC e refresh.
**UI-3.3 futura:** validação manual em celular e desktop em ambiente local seguro; medir render, memória, disco, tempo de consulta, stale/erro e acessibilidade.
**Controle remoto/alertas externos:** projeto e fase distintos, somente se necessário e autorizado; não incluir neste escopo.

## 9. Matriz de aceitação futura (não executada)

| ID | Cenário | Resultado exigido |
| --- | --- | --- |
| MON-01 | Banco indisponível | UNKNOWN e diagnóstico, não saúde verde |
| MON-02 | `production.health_level` healthy com integridade não homologada | Não exibir “pronto” |
| MON-03 | Gate sem intenção ativa com histórico divergente conhecido | Integridade nunca atestada por gate vazio |
| MON-04 | UNKNOWN_COMMIT com STOP possível | Banner de bloqueio; não clicar/repetir BUY/SELL |
| MON-05 | STOP recebido sem persistência | Não afirmar parado |
| MON-06 | STOP PERSISTED sem ACK | Exibir ACK pendente |
| MON-07 | STOP ACKNOWLEDGED com pedido correto | Confirmação e hora com fonte |
| MON-08 | Heartbeat ausente, futuro, inválido ou antigo | Sem medição/STALE, nunca zero artificial |
| MON-09 | Worker “online” persistido, processo caído | Não afirmar processo ativo só pelo campo |
| MON-10 | Feed visual LIVE, worker FALLBACK | Separar fontes; priorizar alerta operacional |
| MON-11 | Provider efetivo desconhecido | UNKNOWN; não presumir Twelve Data |
| MON-12 | Broker PAPER configurado mas efetivo não confirmado | Política e observação em indicadores diferentes |
| MON-13 | Modo real detectado | Alerta crítico e nenhuma ordem real |
| MON-14 | Retenção com conflito histórico | Não afirmar arquivamento íntegro |
| MON-15 | Recibo COMMITTED mas bookkeeping de liberação incerto | Exibir resultado conhecido, liberação não confirmada |
| MON-16 | Duas leituras em épocas diferentes | Não compor aprovação global otimista |
| MON-17 | Recarregar o painel | Nenhum write inesperado, trade, reset, reconciliação ou chamada de feed |
| MON-18 | Usuário comum e admin | RBAC preservado; modo somente leitura na UI-3 inicial |
| MON-19 | Sessão expirada/usuário desabilitado | Acesso negado, sem exibir dados privados |
| MON-20 | 360px–430px e desktop | Falhas e STOP legíveis antes do histórico, sem scroll horizontal |
| MON-21 | Cor desativada/leitor de tela | Estado compreensível por texto |
| MON-22 | Dados obsoletos após reinício do worker | Timestamp da observação e stale visíveis |
| MON-23 | Simulação de múltiplos reruns | Custo de consulta limitado e nenhum efeito financeiro |
| MON-24 | Ausência de dados de trade | Exibição “sem amostra”, sem taxa de acerto inferida |

Testes necessários posteriormente: pytest dos mapeadores, Streamlit AppTest, falhas de banco com PostgreSQL 16/18 descartável para pontos de integração, inspeção visual em dispositivos reais, medição do custo de refresh e revisão Atlas. Resultados **PENDENTES**; não contar como aprovados.

## 10. Decisão de risco e status

**APROVADA APENAS A ESPECIFICAÇÃO.** A página móvel não foi implementada, não há servidor aberto e o acesso remoto não foi projetado em definitivo.

P0 aberto: retenção e quarentena da INFRA-1.2.6. STOP sob carga de 5.000 eventos ainda não atende metas. Não planejar UI verde ou execução 72h sobre backend não homologado.

Proibido: main, merge, PR, Railway, env, banco remoto, migração real, operadora/provedor novo, capital, risco/score/threshold, trades reais, autorização de sinais shadow e qualquer cobrança externa.

**Próximo passo antes de Codex/Work:** preparar mockup textual detalhado ou checklist técnico de implementação, sempre separado do backend. Quando Codex estiver disponível, priorizar primeiro a INFRA-1.2.7 e só depois implementar UI-3 read-only.
