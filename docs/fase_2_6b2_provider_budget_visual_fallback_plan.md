# FASE 2.6B.2 - Provider Budget & Visual Fallback Clarity

## 1. Resumo executivo

Este documento planeja uma futura **FASE 2.6B.2 - Provider Budget & Visual Fallback Clarity**.

Objetivo: melhorar a clareza defensiva sobre consumo de cota, erro 429, provider efetivo, diferenca entre feed operacional do worker e feed visual do grafico, fallback sintetico e risco de interpretacao errada.

Escopo deste documento: **planejamento/documentacao only**.

A futura fase proposta deve permanecer sem autoridade operacional. Ela nao deve alterar estrategia, score, thresholds, `min_signal_score`, broker, provider, capital, ticket, `max_open_positions`, execucao, PnL, historico, posicoes, banco, volume, variaveis de ambiente ou Railway config.

PAPER TRADING permanece obrigatorio. Nenhuma ordem real deve ser enviada.

## 2. Problema observado

A FASE 2.6B.1 foi fechada como **ESTAVEL COM OBSERVACAO IMPORTANTE**.

O sistema permaneceu seguro, mas a operacao evidenciou pontos de clareza que precisam ser melhor documentados e expostos ao usuario:

- consumo de cota Twelve Data pode exceder o limite diario e gerar erro 429;
- Yahoo tambem pode sofrer rate limit;
- o worker pode manter feed operacional LIVE enquanto o grafico visual usa fallback sintetico;
- o provider efetivo operacional pode divergir do provider esperado pelo usuario na leitura visual;
- um preco sintetico ou visualmente inconsistente pode parecer dado estrategico valido;
- a UI precisa deixar claro se a leitura sustenta apenas visualizacao, diagnostico ou decisao operacional.

O risco principal nao e operacional imediato, pois o sistema manteve PAPER, bloqueios e ausencia de ordem real. O risco e de interpretacao: confundir fallback visual ou provider alternativo com confirmacao estrategica.

A FASE 2.6B.2 deve distinguir explicitamente dois casos:

- fallback operacional do worker, quando o ciclo do worker depende de dado sintetico; nesse caso, a leitura estrategica deve ser considerada nao confiavel;
- fallback visual do grafico/Trader, quando apenas a camada visual cai em fallback; nesse caso, o alerta deve ser destacado sem confundir com o feed operacional do worker.

Esses casos devem ser exibidos separadamente na UI/relatorio. Nenhum deles autoriza trade, microajuste, mudanca de threshold, score, broker, provider ou avanco para FASE 2.6C.

## 3. Evidencias da FASE 2.6B.1

Evidencias registradas no fechamento 72h+:

- FASE 2.6B.1 fechada como **ESTAVEL COM OBSERVACAO IMPORTANTE**;
- worker online;
- falhas consecutivas: 0;
- broker: PAPER/simulado;
- execucao real habilitada: Nao;
- posicoes abertas: 0;
- PnL diario: R$ 0,00;
- nenhuma ordem real enviada;
- FASE 2.6B permaneceu em **STUDY_ONLY / DIAGNOSTIC_ONLY / SHADOW_ONLY**;
- nenhum microajuste foi aplicado;
- nenhum threshold foi alterado;
- nenhum score real foi alterado.

Evidencias de feed/provider:

- Twelve Data retornou erro 429 por limite diario excedido;
- exemplo registrado: 896 creditos usados para limite de 800;
- Yahoo apareceu como provider efetivo operacional em parte da janela;
- worker manteve feed operacional LIVE;
- feed operacional do worker registrou `market=5`, `cached=0`, `fallback=0`, `unknown=0`;
- grafico/Trader caiu em FALLBACK sintetico para BTC-USD;
- houve preco visual inconsistente, com exemplo BTC-USD aparecendo como 138.48.

Conclusao das evidencias:

O worker permaneceu operacionalmente seguro, mas a camada visual precisa diferenciar com mais clareza o que e feed operacional, o que e provider efetivo, o que e fallback visual e o que nao deve ser usado como sinal.

## 4. Riscos se nao tratar

Riscos de clareza e observabilidade:

- usuario interpretar fallback sintetico como preco de mercado confiavel;
- usuario confundir feed operacional LIVE do worker com feed visual do grafico;
- usuario nao perceber que Twelve Data atingiu limite diario;
- ausencia de alerta visual para erro 429 ou cota elevada;
- diagnosticos estruturais parecerem sustentados por feed visual inconsistente;
- dificuldade para separar provider operacional, provider visual, cache, fallback e unknown;
- dificuldade para auditar consumo diario de provider antes de novo ciclo de validacao.

Riscos que continuam bloqueados pela arquitetura atual:

- fallback visual nao deve virar gatilho operacional;
- Yahoo ou Twelve Data nao devem virar autorizacao de entrada por si so;
- Fibonacci, BOS, pivo, candle, 4H, 1H ou contexto externo nao devem virar gatilho;
- qualquer melhoria visual nao deve habilitar trade real;
- qualquer melhoria de orcamento nao deve alterar estrategia ou threshold.

## 5. Escopo permitido da futura fase

A futura FASE 2.6B.2 pode estudar e documentar melhorias como:

- painel de consumo estimado de cota Twelve Data por dia;
- indicacao visual de risco de erro 429;
- separacao explicita entre provider operacional do worker e provider visual do grafico;
- label de feed operacional: LIVE, DELAYED, CACHED, FALLBACK ou UNKNOWN;
- label de feed visual: provider, cache, fallback sintetico ou unavailable;
- aviso quando grafico/Trader estiver usando fallback sintetico;
- aviso quando o preco visual nao deve ser usado para leitura estrategica;
- resumo de orcamento diario estimado por provider;
- contadores diagnosticos de chamadas, falhas, fallback visual e fallback operacional;
- logs e cards de observabilidade sem impacto em decisao;
- mensagens defensivas para evitar leitura incorreta do usuario.

Todo item permitido deve permanecer em modo de observabilidade, clareza ou controle de cota.

## 6. Escopo proibido

Permanece proibido:

- alterar estrategia;
- alterar score real;
- alterar thresholds;
- alterar `min_signal_score`;
- alterar broker;
- alterar provider como autoridade operacional;
- alterar capital;
- alterar ticket;
- alterar `max_open_positions`;
- alterar execucao;
- alterar ordem paper oficial;
- habilitar ordem real;
- alterar PnL, historico ou posicoes;
- alterar banco, volume, variaveis de ambiente ou Railway config;
- fazer redeploy como parte deste planejamento;
- transformar fallback em autorizacao operacional;
- transformar Yahoo ou Twelve Data em sinal de entrada;
- transformar Fibonacci, BOS, pivo, candle, H1, H4 ou Multi-TF em gatilho de entrada;
- avancar para FASE 2.6C.

## 7. Proposta de melhorias visuais e de observabilidade

Melhorias propostas para estudo futuro:

- Card "Provider operacional do worker": provider efetivo, status, timestamp e qualidade.
- Card "Provider visual do grafico": provider usado na renderizacao, status e origem do preco.
- Card "Cota Twelve Data": consumo estimado, limite diario conhecido, percentual usado e risco de 429.
- Badge "Fallback visual ativo": exibido quando grafico/Trader usa fallback sintetico.
- Badge "Feed operacional limpo": exibido quando worker tem feed LIVE sem fallback operacional.
- Aviso defensivo: "Preco visual em fallback sintetico; nao usar como confirmacao estrategica."
- Tabela curta por ativo com colunas `symbol`, `worker_feed_status`, `visual_feed_status`, `provider_effective`, `fallback_visual`, `fallback_operational`, `last_success`.
- Log diagnostico para diferenciar `provider_budget`, `visual_fallback`, `worker_feed` e `chart_feed`.
- Resumo diario no email indicando se houve 429, rate limit, fallback visual ou divergencia worker/grafico.

Todas as melhorias devem ser informativas. Nenhuma delas deve alterar decisao, score, threshold, broker, provider operacional, execucao ou criacao de ordens.

## 8. Criterios de sucesso

A futura fase deve ser considerada bem-sucedida se:

- UI diferenciar claramente feed operacional do worker e feed visual do grafico;
- erro 429 da Twelve Data for apresentado como evento de cota, nao como bloqueio estrategico automatico;
- fallback visual for rotulado de forma evidente;
- provider efetivo operacional for visivel sem confundir autoridade de decisao;
- usuario conseguir identificar se o preco mostrado e live, cached, fallback sintetico ou unknown;
- logs permitirem busca por eventos de cota e fallback visual;
- email diario registrar cota/fallback visual quando relevante;
- PAPER TRADING permanecer obrigatorio;
- nenhuma ordem real for enviada;
- nenhuma decisao operacional for alterada;
- FASE 2.6C continuar bloqueada.

## 9. Criterios de bloqueio

A futura fase deve ser bloqueada se qualquer proposta:

- alterar estrategia;
- reduzir threshold ou `min_signal_score`;
- alterar score real;
- alterar broker;
- alterar provider como autoridade operacional;
- alterar capital, ticket ou `max_open_positions`;
- alterar execucao;
- criar ordem paper oficial;
- habilitar ordem real;
- alterar PnL, historico ou posicoes;
- mexer em banco, volume, variaveis de ambiente ou Railway config;
- usar fallback, Fibonacci, BOS, pivo, candle, H1, H4 ou Multi-TF como gatilho operacional;
- recomendar avancar para FASE 2.6C sem nova validacao.

## 10. Confirmacao de que a fase nao altera operacao

A FASE 2.6B.2 proposta deve ser apenas de observabilidade, clareza visual e controle de cota.

Confirmacoes obrigatorias:

- `PAPER_TRADING` permanece obrigatorio;
- dinheiro real permanece bloqueado;
- ordens reais permanecem desabilitadas;
- ordem paper oficial nao deve ser alterada;
- estrategia real nao deve ser alterada;
- score real nao deve ser alterado;
- thresholds nao devem ser alterados;
- broker nao deve ser alterado;
- provider nao deve ser alterado como autoridade operacional;
- capital, ticket e `max_open_positions` nao devem ser alterados;
- PnL, historico e posicoes nao devem ser alterados.

## 11. Confirmacao de que FASE 2.6C continua bloqueada

Este plano nao autoriza a FASE 2.6C.

A FASE 2.6C continua bloqueada ate que exista nova decisao explicita, nova branch, nova revisao de seguranca e nova aprovacao. O objetivo da futura FASE 2.6B.2 e apenas reduzir ambiguidade visual e melhorar observabilidade de provider/cota/fallback.

Nao ha autorizacao para:

- aplicar microajuste;
- alterar threshold;
- alterar score;
- alterar perfil;
- operar dinheiro real;
- transformar diagnostico shadow em decisao operacional.

Decisao deste documento: planejar uma fase futura defensiva de clareza e observabilidade, mantendo o sistema seguro em PAPER e sem mudanca operacional.
