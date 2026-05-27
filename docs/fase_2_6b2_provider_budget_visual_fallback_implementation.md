# FASE 2.6B.2 - Provider Budget & Visual Fallback Clarity

## Resumo executivo

A FASE 2.6B.2 adiciona uma camada defensiva de observabilidade para provider budget, cache, risco de 429 e escopo de fallback visual/operacional. A fase e `OBSERVABILITY_ONLY` / `DIAGNOSTIC_ONLY` / `SHADOW_ONLY` e nao altera decisao operacional.

## Escopo implementado

- Consolida um bloco `state["provider_budget_visual_fallback"]`.
- Separa feed operacional do worker e feed visual do grafico/Trader.
- Distingue fallback sintetico operacional do worker de fallback apenas visual.
- Exibe status de cota diaria estimada, risco de limite por minuto, 429 observado, cache hits/misses e chamadas estimadas.
- Exibe `daily_budget_limit`, `minute_limit`, `daily_budget_source` e `minute_limit_source`.
- Usa defaults conservadores de observabilidade para Twelve Data (`800` creditos/dia e `8` chamadas/min) quando nao houver medicao real no runtime.
- Marca limites sem consumo real como `configured` ou `estimated`, nunca como medicao real.
- Adiciona logs auditaveis `provider_budget_visual_fallback_*` no worker.
- Adiciona leitura curta no relatorio diario por email.
- Adiciona blocos visuais no Trader e Controle do Bot.

## Fontes de budget

- `measured`: valor observado no runtime ou diagnostico do provider.
- `configured`: valor recebido por configuracao/runtime, sem alterar variavel de ambiente nesta fase.
- `estimated`: default defensivo usado apenas para clareza quando o provider efetivo e Twelve Data e nao ha medicao real.
- `unknown`: sem base suficiente para exibir limite ou consumo.

Essas fontes sao somente informativas. Elas nao reduzem threshold, nao mudam score, nao trocam provider, nao alteram broker e nao autorizam trade.

## Garantias de seguranca

- PAPER TRADING permanece obrigatorio.
- Nenhuma ordem real e autorizada.
- Nenhuma ordem paper oficial e criada por esta camada.
- Score real, `min_signal_score`, thresholds, broker, provider, capital, ticket e `max_open_positions` nao sao alterados.
- Fibonacci, BOS, pivo, H1, H4, webhook, diagnosticos shadow e fallback nao viram gatilho operacional.
- FASE 2.6C continua bloqueada.

## Leitura de fallback

Fallback operacional do worker significa que o ciclo do worker dependeu de dado sintetico. Nesse caso, a leitura estrategica deve ser considerada nao confiavel.

Fallback visual do grafico/Trader significa que apenas a camada visual caiu em fallback. Esse caso deve ser mostrado separadamente e nao deve ser confundido com o feed operacional do worker.

Nenhum dos dois casos autoriza trade, microajuste, mudanca de threshold, mudanca de score, mudanca de broker, mudanca de provider ou avancar para FASE 2.6C.

## Logs para procurar

- `[provider_budget_visual_fallback_summary]`
- `[provider_budget_visual_fallback_budget]`
- `[provider_budget_visual_fallback_scope]`
- `[provider_budget_visual_fallback_cache]`
- `[provider_budget_visual_fallback_safety]`

## Validacao esperada

- `git diff --check`
- `git diff --cached --check`
- `py_compile` nos arquivos Python alterados
- `pytest tests/test_provider_budget_visual_fallback.py`
- Testes relevantes de state/feed se alterados

## Decisao final

Esta implementacao e somente observabilidade/clareza. Ela melhora a leitura de budget/cache/fallback sem mudar estrategia, score, thresholds, broker, provider ou execucao.
