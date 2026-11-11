# Mapa de concorrência ANTES das correções INFRA-1.2.1

Base inspecionada: 5d7165217e42ff5121b5444f956d45a877118734.
Branch de trabalho: atlas/infra-1-2-1-parada-segura.
Mapa registrado antes da alteração funcional dos ciclos.

1. run_trader_cycle usa @transactional_persistence: adquire o advisory lock
   atlas:persistence-unit antes de ler estado/risco/broker/STOPPED.
2. run_paper_cycle decorado aninha savepoint na mesma conexão. Dentro dessa
   unidade executa load_market_data_result/fetch_market_data_map e, mais tarde,
   fetch_multi_timeframe_intraday_data. Enriquecimento/estratégia/diagnósticos,
   trades, relatórios, estado PAPER, ordens e espelho de posições seguem sob lock.
3. Duração do lock = duração completa de feed + cálculo + persistência + leitura
   de relatório. Não existe limite superior da duração do feed imposto pelo lock.
   Writers concorrentes têm lock_timeout PostgreSQL de 5s por aquisição.
4. STOP não tem API nem recibo: páginas alteram bot_status no dict e salvam todo
   bot_state. O mesmo global lock protege save/update, heartbeat, auth, logs,
   retenção e reset. Durante feed lento STOP pode falhar antes de persistir.
5. Worker lê bot_status antes de chamar o ciclo; não revalida STOP ao final da
   aquisição. Não há request_id, persisted_at ou reconhecimento do worker.
6. Trader UI oferece iniciar/pausar sem Parar dedicado; Controle usa snapshot
   do topo para salvar status/modo. Configuração/admin/reset têm catches genéricos
   ou nenhum catch; sucesso local não distingue intenção de confirmação do worker.
7. reset_state substitui o documento por DEFAULT, inclusive STOPPED. CAS protege
   snapshot antigo, mas reset explícito genérico ainda remove a intenção parada.
8. Interrupção sob PostgreSQL reverte ledger da INFRA-1.2; arquivos locais só
   asseguram atomicidade por arquivo. Chave trade_id não tem índice único.

Reprodução determinística: worker da versão base foi bloqueado por Event dentro
fake feed; STOP real recebeu LockNotAvailable enquanto o Event permanecia fechado.
Estado permaneceu RUNNING, nenhum trade foi confirmado; feed liberado terminou
com exceção sintética e rollback. Resultado em infra121-legacy-stop.json/txt.
Nenhum sleep arbitrário é usado para comprovar a ordem; o timeout é do PostgreSQL.
