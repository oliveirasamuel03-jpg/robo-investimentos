# ATLAS — PC-1: Plano de computador, instalação local, backup, recuperação e custos

**Data de elaboração:** 12/11/2026.
**Repositório exclusivo:** `oliveirasamuel03-jpg/robo-investimentos`.
**Branch:** `atlas/pc-1-plano-local-backup-custos`.
**Base conferida:** `atlas/ui-3-monitoramento-mobile-spec`, SHA `06ec2702911eb772031ab27eefa661f03e81e481`.
**Natureza:** `PLANNING_ONLY / PAPER_ONLY / COST_CONTROLLED / NO_PRODUCTION_DEPLOY`.

**Esta etapa não compra computador, não instala nada, não altera código, parâmetros financeiros, banco, Railway, volume ou variáveis, não cria PR/merge, não inicia worker e não autoriza 72 horas.**

## 1. Parecer executivo

**Recomendação:** adiar a compra até resolver as pendências financeiras da INFRA-1.2.6/1.2.7, confirmar STOP sob carga e medir CPU/RAM/disco em testes curtos. O projeto já tem Python 3.12, Streamlit, worker e PostgreSQL implementados; não necessita de GPU dedicada e pode ser preparado para uma máquina local de custo controlado. Um PC não corrige inconsistências de ledger.

**Ponto crítico de portabilidade:** `core/persistence.py` importa `fcntl`, módulo de bloqueio de arquivos típico de Linux/Unix. O Windows nativo não deve ser homologado com base apenas na presença de `run_web.bat` e `run_worker.bat`. Priorizar **Linux nativo** ou **Windows 11 + WSL2 (Linux dentro do Windows)**, após testes. Os scripts .bat usam um caminho local do runtime Codex como preferência e fallback para `python`; isso não é uma instalação portátil nem evidência de suporte Windows puro. Não alterar arquivos de execução nesta fase.

**Outro bloqueio:** `scripts/local_synthetic_backup.py` valida deliberadamente Docker, loopback `127.0.0.1`, rótulo de contêiner e credenciais/nomes de base de testes. **É um ensaio de backup/restauração sintético, não um procedimento de backup de dados operacionais.** Uma rotina separada, homologada e autorizada será necessária para o futuro PC.

## 2. Evidência técnica no GitHub

- `.python-version`: `3.12`; `requirements.txt`: `streamlit==1.44.1`, `pandas==2.2.3`, `numpy==2.2.4`, `psycopg[binary]==3.2.6` e demais dependências. Essa é a **base do projeto**, não recomendação de versão mais recente.
- `requirements-dev.txt`: pytest 8.3.5, com outras dependências referenciadas; `requirements-dev.lock` disponível para investigação de reprodutibilidade.
- `README.md`: separa web/worker e prevê persistência local e `DATABASE_URL`; **arquivos locais não são substituto de transação PostgreSQL** para longa homologação.
- `core/persistence.py:7`: `import fcntl`; `core/persistence.py:125-164`: unidade transacional PostgreSQL e aviso de atomicidade apenas por arquivo quando local.
- `.env.example`: `APP_ENV=development`, `BROKER_PROVIDER=paper`, `BROKER_MODE=paper`, `PRODUCTION_MODE=false`, `DATABASE_URL` vazio, flags de sinal externo em modo auditoria, credenciais não preenchidas. O antigo `AUTH_REQUIRED=false` é exemplo legado: `core/auth/session.py:40-45` exige login no backend atual. **Não copiar configuração sem revisão de segurança**.
- `scripts/sql/new_history_schema.sql`: só prepara schema de banco **novo e vazio**, e recusa relações existentes. **Não é migration nem restauração**.
- `scripts/local_synthetic_backup.py`: backup com fonte em STOP, snapshot consistente, `pg_dump`/restore, comparação de inventário, sequência/índices e quarentena, restrito a dados fictícios.
- `core/retention.py`: limpeza/arquivamento ainda sob revisão; `tests/test_infra126_postgres.py` relata modificação de manifesto e perda potencial de provas. Não ativar retenção automática no futuro computador antes da correção.
- `core/production_monitor.py`: mede heartbeat e saúde operacional, mas não prova integridade do ledger.
- `pages/1_Trader.py` e `pages/4_Controle_do_Bot.py`: ações do operador exigem cautela. Home mantém atalho de ciclo manual que será revisado em UI-2.

## 3. Hardware: recomendações preliminares, não requisitos homologados

| Perfil | CPU | RAM | Disco principal | Uso pretendido | Decisão |
| --- | --- | --- | --- | --- | --- |
| Referência econômica | CPU moderna com 4 núcleos reais, 64-bit | 16 GB | SSD 512 GB | Streamlit + 1 worker PAPER + 1 PostgreSQL + navegador, dados históricos modestos | **Ponto de partida para orçamento, sem garantia de desempenho** |
| Margem para testes exigentes | CPU moderna 6–8 núcleos | 32 GB | SSD NVMe 1 TB | PostgreSQL descartável adicional, Docker, testes concorrentes, histórico maior | Comprar apenas se benchmark justificar |
| Computador atual/reaproveitado | Depende de inventário real | Preferir 16 GB | SSD saudável e espaço livre monitorado | Teste de capacidade com zero compra inicial | **Prioridade antes de adquirir outro PC** |

Não é necessário GPU dedicada nem computador gamer. Mini PC ou desktop pode ser mais econômico para funcionamento prolongado; notebook serve em homologação curta, observando refrigeração, suspensão e alimentação. Comparar garantia, eficiência, reparabilidade, conectividade e ruído. Um notebook usado pode ter bateria deteriorada; preferir avaliação de saúde.

**Capacidade final obrigatoriamente medida:** pico de RAM web + worker + PostgreSQL; uso de CPU, filas, disco, crescimento mensal de PostgreSQL/archives e STOP/lock com 5.000 eventos. A meta técnica ainda não foi cumprida no código estudado. Não concluir que 16 GB resolve o problema de lock, que é arquitetural.

### Plataforma e sistema operacional

- **Linux nativo 64-bit**: candidato de menor atrito com `fcntl`, `venv`, `psycopg` e scripts Bash. Não foi homologado neste computador.
- **Windows 11 + WSL2 Ubuntu**: alternativa quando Windows for necessário; virtualização compatível, WSL2, integração dos caminhos de arquivos, permissões e rede local precisam ser testadas. Instalar o app/venv **dentro do filesystem Linux do WSL2**, evitar execução simultânea de versões Windows/Linux sobre o mesmo storage.
- **Windows Python nativo**: **não aprovado** por conta da importação `fcntl`. Não recomendar usar .bat como solução; exigir ajuste de portabilidade em fase específica e regressão antes de considerar suporte nativo.

Referência oficial para WSL2: https://learn.microsoft.com/pt-br/windows/wsl/install .

## 4. Arquitetura local alvo — somente proposta

```
                 [PC LOCAL — CONEXÕES PRIVADAS]
  navegador no próprio PC -> Streamlit 127.0.0.1
                               | leitura/controle autenticado
                               v
  um processo Web --------> PostgreSQL 127.0.0.1
  um processo Worker PAPER -> PostgreSQL 127.0.0.1
                               |
                          estado canônico
                               |
             backup consistente sob STOP + verificação
                               v
                  mídia separada/offline
```

Sem exposição de 8501 ou 5432 em IP público, sem port-forward no roteador, sem Cloudflare Tunnel, VPN, serviço de hospedagem ou acesso remoto nesta fase. Monitorar pelo celular **não** requer acesso remoto agora; a arquitetura UI-3 é apenas especificação. Futuro acesso móvel exigirá fase de autenticação, TLS e rede privada devidamente auditada; não ativar acesso pela internet apenas porque o app abre no localhost.

Um worker apenas; não rodar instâncias antigas ou usuários com credenciais de DML incompatíveis. O computador não deve iniciar o worker automaticamente antes da homologação; `STOPPED` e gate de integridade precisam ser inspecionados.

## 5. Instalação planejada — roteiro para o dia do computador

Nenhum passo abaixo foi executado agora. **Não usar automaticamente comandos contra dados existentes**.

### Etapa A — Preparação do host

1. Inventariar CPU/RAM/SSD, versões SO, virtualização, espaço livre, energia estável, internet e acesso de administrador local. Validar integridade do disco e atualizações de segurança.
2. Escolher Linux ou Windows+WSL2; definir usuário do sistema operacional sem privilégios administrativos para executar app e worker.
3. Atualizar o sistema, instalar Python 3.12 (conforme `.python-version`), Git, cliente PostgreSQL compatível e dependências com versões auditadas. Docker só para testes locais descartáveis e se a máquina suportar.
4. Usar **uma branch candidata explicitamente revisada**; nunca `main` automaticamente. Conferir SHA, diff, hashes de dependências, licenças e `pip check`.
5. Separar diretório de código, `venv`, arquivos temporários, logs e diretório de backup. Não versionar `.env`, logs pessoais nem dumps.

### Etapa B — Dados e autenticação

1. Criar banco local novo/descartável, com porta restrita a loopback e roles com privilégio mínimo **após** definir o contrato de escritores da INFRA-1.2.7; **não criar banco ou roles nesta etapa de planejamento**.
2. Conferir compatibilidade PostgreSQL 16 e 18; escolher versão alvo somente após homologação da branch candidata. `DATABASE_URL` deve apontar exclusivamente para localhost do ambiente de teste, nunca Railway ou conta antiga.
3. Configurar login administrativo com segredo forte local; validar usuários, RBAC e timeout de sessão. Não publicar credenciais ou DSN no chat ou repositório.
4. Preservar `BROKER_PROVIDER=paper`, `BROKER_MODE=paper`; nenhuma credencial de corretora real; flags de execução real, sinais externos, emails e serviços de rede continuam desativados. Não mudar provider Twelve Data nem parâmetros financeiros.
5. Não importar dados legados sem projeto de migração/validação próprio. **Banco novo ≠ recuperação de histórico antigo.**

### Etapa C — Testes curtos antes do worker

1. `python -m compileall`, `python -m pip check`, `git diff --check`, pytest offline; PG16/18 somente em bases locais descartáveis e controle de volumes.
2. Reproduzir e corrigir P0 da retenção/quarentena na INFRA-1.2.7; confirmar backups com dados sintéticos e STOP/UNKNOWN.
3. Iniciar apenas Streamlit em loopback para teste de login/interface; não iniciar worker/execução manual por conveniência.
4. Com a fase de testes autorizada e o ambiente isolado, iniciar **um** worker PAPER para janelas curtas. Confirmar heartbeat, broker PAPER, feed e fonte de dados. Abortar em qualquer divergência.
5. Coletar recursos/latência, testar STOP, integridade, recuperação de falhas e equivalência de relatório. A homologação de 72 horas só pode ser considerada após novos critérios e revisão Atlas independente.

## 6. Backup futuro: protocolo necessário, não existente em produção

**Estratégia candidata, ainda sem implementação:** backup PostgreSQL consistente com fonte em STOP, versão dos escritores congelada, gate/recibos registrados e arquivos auxiliares preservados. `pg_dump` representa o estado do banco num snapshot consistente, porém **não captura automaticamente arquivos do sistema fora do banco**; a captura coordenada do storage e catálogos precisa de contrato explícito. Não usar backup como forma de remover divergências financeiras. A ferramenta PostgreSQL é descrita em https://www.postgresql.org/docs/18/app-pgdump.html e https://www.postgresql.org/docs/18/backup-dump.html.

### 6.1 Componentes mínimos de cópia

- Banco PostgreSQL: app_state, app_rows, sequences, índices, controle STOP, gate UNKNOWN, intenções e recibos, saldo, posições, trades, relatórios e contas sintéticas de teste.
- Diretórios relevantes sob `ROBO_STORAGE_DIR`, se usados: runtime, archives, weekly reports, relatórios e outros arquivos não derivados do banco. Identificar responsabilidades por namespace e por versão.
- Metadados fora do banco: commit/tag SHA exato, dependências, schema/role **sem segredos**, momento UTC, tamanho, inventário e hashes de arquivos, localização de mídias e regras de retenção.
- Credenciais e chaves: backup criptografado com gestão separada; **não exportar segredos em logs ou GitHub**, não testar restauração com credenciais reais desnecessárias.

### 6.2 Política candidata (a aprovar após teste)

1. Antes de cada backup: confirmar STOP persistido e, quando aplicável, reconhecido, e garantir ausência de unidade financeira em andamento. Se STOP/estado falhar, backup não é marcado como válido.
2. Adquirir fence/congelamento de escritores relevante; detectar `UNKNOWN_COMMIT` e **preservá-lo**, nunca liberá-lo automaticamente para obter um backup “limpo”.
3. Capturar estado PostgreSQL e storage consistente entre si, mantendo identificadores/épocas. Criar arquivo temporário, verificar código de saída e integridade de transferência; hash e inventário assinados ou em storage protegido são desejáveis, não provas independentes de integridade do ledger.
4. Manter ao menos duas cópias em mídias distintas; uma delas fora do disco do PC e preferencialmente desconectada e cifrada. O método `3-2-1` é boa orientação futura, mas pode ter custo adicional e não foi imposto.
5. Definir retenção de backups apenas depois de medir volume real e política de privacidade. **Nunca confundir limpeza de backup com retenção dos registros financeiros originais**.
6. Medir duração/tamanho e restaurar periodicamente em **destino novo, isolado e sem worker**. Conferir linhas, índices, sequences, hashes, STOP, recibos, `UNKNOWN_COMMIT` e o relatório de divergências.
7. Restaurado = **STOPPED/QUARANTINE**. Nenhum START automático, reenvio de BUY/SELL ou reconciliação que apague pendências.

**Atenção:** não adotar `scripts/local_synthetic_backup.py` para banco operacional; seu contrato de isolamento e teste é deliberadamente restrito. Backups regulares para o futuro computador requerem nova função e testes sem alterar esse guard.

## 7. Recuperação e cenários de falha (plano)

| Falha | Ação segura | Evidência para retorno |
| --- | --- | --- |
| Queda de energia/PC | Não auto-iniciar worker na volta; inspeção local do PG, gate e STOP | Arquivos, ledger e controle íntegros; sem replay financeiro |
| PostgreSQL indisponível | Falhar fechado; UI deve exibir “NÃO CONFIRMADO” | Banco recuperado, saúde e identidade verificadas |
| COMMIT com resposta perdida | Preservar intenção/UNKNOWN, sem retry | Reconciliação positiva e durável sob fence, sem BUY/SELL |
| Backup inconsistente | Não usar para restaurar; manter cópias anteriores | Comparação de inventário e restauração isolada aprovada |
| Disco quase cheio | Suspender estudos pesados/worker com segurança, coletar diagnóstico | Espaço seguro restaurado sem apagar evidências |
| Corrupção de recibo/trade | Quarentena persistente / STOP quando possível | Integridade reavaliada, nunca corrigir via reset automático |
| Worker duplicado/antigo | Parar tentativa de homologação e isolar versões | Um worker compatível; sem DML externo bypass |
| Perda de internet/Twelve Data | Feed UNKNOWN/DELAYED/FALLBACK, sem ampliar entradas | Fonte/horário/escopo verificados em PAPER |
| Celular sem acesso ao PC | Nenhuma alteração remota; operador não presume STOP | Consulta local/autenticada possível em fase futura |

**Objetivos de recuperação candidatos, não garantias:** tempo de restauração (RTO) e perda máxima de dados recuperáveis (RPO) só podem ser definidos depois do tamanho e frequência de backup, da durabilidade local e dos testes de recuperação. Não declarar “backup diário seguro” sem restore testado.

## 8. Orçamento e custo recorrente

**Compra não autorizada agora.** Preferência é aproveitar computador existente, pesquisar equipamento usado com garantia, ou um mini PC/desktop eficiente após medir consumo.

### 8.1 CAPEX — aquisições futuras condicionais

| Item | Observação |
| --- | --- |
| Computador | Orçamento PENDENTE de cotação contemporânea ao momento da compra; CPU 4 núcleos, 16 GB RAM, SSD 512 GB como referência preliminar |
| Disco separado para backup | Avaliar após medição do banco; por exemplo SSD externo USB 1 TB ou HD externo adequado, nunca apenas partição do mesmo SSD interno |
| Nobreak/UPS | Opcional para análise posterior de risco de energia; não substitui backup nem corrupção do banco |
| Acessórios/monitor | Não presumir necessidade; celular não substitui setup do host |

**Preço de referência, não cotação:** pesquisa pública em novembro de 2026 encontrou um SSD externo SanDisk 1 TB em torno de R$ 926–945 em comparadores; lojas, frete, garantia e capacidade real devem ser reconfirmados antes da compra. A referência não é recomendação comercial nem compra aprovada:
https://fit.melhorprecoagora.com.br/produto/informatica-sandisk-sdssde301t00-1024gb
https://sobusca.com.br/produto/ssd-externo-portatil-1tb-interface-usb-3-2-sandisk-sdssde30-1t00-veloc-mlb19885932

Não foram verificadas propostas suficientemente comparáveis para **preço do PC** nesta sessão; não apresentar preço de máquina inventado ou desatualizado.

### 8.2 OPEX — operação local

- **Railway/hosting:** R$ 0 de mensalidade prevista *se não for usado*, sem reiniciar qualquer projeto existente; não é custo total zero.
- **PostgreSQL, Streamlit, Python, Git:** software utilizável sem assinatura; conferir licença de SO, máquina e ferramentas no momento da aquisição.
- **Energia:** fórmula mensal `(potência média em watts / 1000) x horas/dia x dias x tarifa efetiva R$/kWh`. Exemplo ilustrativo a 24h x 30 dias: 15 W médios = 10,8 kWh; 50 W = 36 kWh, **sem incluir tarifa local, modem e perdas adicionais**. Medir potência real do conjunto, nunca prometer valor mensal fixo.
- **Internet/Twelve Data:** verificar franquia de API e limites contratuais do plano disponível quando o worker for autorizado. Desabilitar loops, probes e emails não essenciais no teste de custo. Não comprar plano pago sem fase.
- **Backups:** custo da mídia, substituições periódicas, eventual criptografia/cópia externa; não ligar plano cloud automaticamente.
- **Manutenção:** atualizações, falhas e substituição de hardware; observar desgaste/temperatura e sistema suspenso.

**Teto de custo:** nenhuma cobrança recorrente externa sem aprovação prévia. Antes de compra, cotar conjunto mínimo + backup, e medir custo da energia e volume de dados em janela curta. Não habilitar auto-renovação de serviços pagos.

## 9. Checklist de aceitação para aquisição e PC

**Antes de comprar:**
- [ ] INFRA-1.2.7 fechou P0/P1 (retenção/quarentena) com evidências PG16/18?
- [ ] STOP/lock com 5.000 eventos atingiu metas ou foi documentado plano realista aprovado?
- [ ] Banco alvo e escritor/role compatíveis foram definidos? Sem dependência de Railway?
- [ ] Benchmark curto mediu picos de RAM/CPU/SSD e custo por consulta?
- [ ] Há acesso físico ao PC e possibilidade de testar Linux/WSL2?
- [ ] O preço total e o plano de backup cabem no orçamento sem assinatura externa?
- [ ] Existe política de recuperação pós-queda de energia e proteção de segredos?

**Antes de 72h de PAPER contínuo:**
- [ ] Um worker e web compartilham o mesmo PostgreSQL local autenticado?
- [ ] STOP persistido e ACK demonstrados, sem falso sucesso?
- [ ] UNKNOWN, integridade e retenção são fail-closed mesmo após restart?
- [ ] Restore completo validado em outra base com rollback/isolamento?
- [ ] Alertas e UI não mentem sobre feed, broker e heartbeat?
- [ ] CPU/RAM/disco/temperatura e energia medidos?
- [ ] Plano de falha, suspensão/hibernação e reconexão documentados?
- [ ] Revisão Atlas independente da versão candidata autorizou 72h?
- [ ] Nenhuma ordem real; PAPER e parâmetros protegidos?

Qualquer checkbox pendente é BLOQUEIO ou PENDÊNCIA, não aprovação presumida.

## 10. Execução futura e separação de fases

PC-1 é **apenas documentação**. INFRA-1.2.7 é prioridade quando Codex/Work retornarem; UI-3 será implementada só depois de backends verificados. O projeto de **backup local operacional** requer autorização e testes específicos; não adaptar script sintético para acesso real. A compra e configuração física aguardam avaliação econômica posterior.

A documentação não mudou main, não criou PR, não realizou merge, não fez deploy, não acessou Railway, banco remoto, variáveis, storage existente, nem executou Python/PostgreSQL no computador do usuário. A assinatura dos testes permanece **PENDENTE**.

**Decisão Atlas: aperfeiçoar primeiro a prova financeira e o STOP, comprar depois — nunca o contrário.**
