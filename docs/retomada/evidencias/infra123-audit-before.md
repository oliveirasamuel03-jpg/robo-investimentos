# Auditoria inicial INFRA-1.2.3

HEAD remoto consultado antes da branch: e62a3a2be39d70782ce07d50e59027fc2c32e886,
igual à referência conhecida. Repositório exclusivo robo-investimentos.

Riscos por comprovar na base: falhas anteriores são injeções Python nas fronteiras
transacionais, sem perda real de TCP; faltam crash do postmaster, durabilidade de
cluster, backup completo de gate/recibos/STOP/usuários e restauração em quarentena;
manifestos são O(histórico), sem orçamento de latência/memória medido. Runtime
local em arquivos continua sem transação multi-arquivo; exactly-once externo,
falha de energia/host e operação 72h não foram validados.

Falha concreta reproduzida antes de correção: após A e B confirmados, diagnóstico
atrasado de A podia reabrir gate.active_cycle_id=A quando já estava vazio e
last_cycle_id=B. Recibo histórico A era comparado ao manifesto atual de B sem
qualificação histórica. Novos testes executados com código da base: 2 failed,
1 passed; evidência infra123-identity-before.txt. O controle financeiro corrente
não pode ser reordenado por diagnósticos antigos.

O runner anterior usava tmpfs: não serve para alegar durabilidade após reinício
do contêiner. A homologação desta fase usará volume anônimo próprio e porta
loopback estável, removidos apenas com o contêiner etiquetado do teste. Nenhum
volume/banco antigo pode ser alterado. Backup deve comparar snapshot consistente
antes de persistir STOP de quarentena; não ativar worker no restore.
