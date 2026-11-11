# ATLAS — acesso à recuperação e controles financeiros

Data: 11/11/2026. PAPER_ONLY / NO_PRODUCTION_DEPLOY.
Branch: `atlas/infra-1-1-persistencia-segura`.
Base auditada: `9e05cb2c43d2199904c94f26fb790882896cb57c`.

## Evidências e limites

| Item | Resultado |
|---|---|
| Dossiê histórico | Lido `ATLAS_ROBO_INVESTIMENTO_DOSSIE_MIGRACAO.md`, registro de 10/11/2026; inventário, não backup |
| Conta antiga | Consulta somente de leitura ao projeto `390990a3-4ac3-407e-a07f-a589a050b19a`, ambiente `7449c165-a1ff-4380-a55c-813d9f6611fa`, recusada: `You don't have the required role (viewer) on this resource.` |
| Banco antigo | Nenhuma conexão, exportação ou restauração executada; backup recuperável continua NÃO VERIFICADO |
| Versão histórica | Imagem informada no dossiê: `ghcr.io/railwayapp-templates/postgres-ssl:18`; versão efetiva do servidor não confirmada |
| Teste anterior | 349 testes offline e 19 verificações PostgreSQL 16.10 aprovados na base acima; não demonstram compatibilidade com PostgreSQL 18 nem recuperação do histórico |
| Conta nova | Conector lista o workspace autorizado `4e937e0d-7701-4de4-8d18-b1c61bd5b47d` e projeto `perceptive-integrity` |
| Infraestrutura nova | Último deploy `821211f6-531b-4f5a-a25c-2ae80dbb6d29` CRASHED; nove alterações STAGED preservadas; sem worker/PostgreSQL/volume/bucket provisionados nesse projeto |
| Painel financeiro | Conector não expõe ferramentas de billing/limites. Navegador sem sessão apresenta Login e 404 na página do projeto; consumo/saldo/limites não consultáveis nesta sessão |
| Valores financeiros | Trial, crédito US$ 5 e consumo anterior US$ 0 são informações do usuário, não medição atual independente |

Nenhuma tentativa de contornar permissões, reativar serviços antigos ou resolver cobrança por contratação. Nenhuma configuração financeira alterada. Nenhum serviço externo criado, deploy, restart ou aceite de staged executado. Isso não comprova que a fatura total da conta seja zero.

## Recuperação: sequência condicionada ao acesso legítimo

1. Obter acesso autorizado de leitura ao inventário/backups antigos, sem reativar serviços ou pagar cobranças automaticamente. Se isso exigir custo, parar essa etapa e solicitar autorização específica.
2. Confirmar versão efetiva, backup disponível, data, tamanho, integridade e origem. Volume de 5 GB por si só não comprova backup.
3. Usar cliente de exportação compatível com a versão efetiva (não presumir que o `pg_dump` 16 local atende uma origem 18). Registrar versões do cliente, origem e destino. Não planejar downgrade implícito para 16.
4. Validar primeiro uma cópia em ambiente local isolado compatível, sem provider/worker e sem sobrescrever histórico. Conferir tabelas, contagens, timestamps, estado, posições, usuários e sequência, protegendo senhas e hashes.
5. Somente após evidências completas propor a recuperação remota, ainda sujeita a autorização para escrita/deploy. O script `audit_database_readonly.py` não recupera backups e não deve ser executado contra endpoint inacessível ou exigir ativação.

Se não houver histórico recuperável, banco novo depende de decisão explícita do usuário e será documentado como início de histórico novo, nunca como migração bem-sucedida.

## Financeiro e homologação

Ainda é necessário acesso autenticado ao billing para confirmar plano, crédito restante, consumo, ciclo e limites disponíveis gratuitamente. Preparação desejada: Compute Hard Limit no máximo US$ 10, Agent no menor valor permitido e desativado; nenhum upgrade, cartão ou crédito. Compute não é teto de assinatura, impostos, Agent ou fatura total. Se qualquer limite exigir contratação, registrar a limitação sem contratar.

O staged ainda aponta para INFRA-0, que não inclui as correções INFRA-1/1.1. Não aceitar esse conjunto como versão homologada. Antes de preparar um novo destino, revisar deploy automático/preview e branch aprovada; não criar PR nem aplicar staged nesta etapa.

## Validação desta atualização

Apenas documentação foi alterada; `git diff --check` executado. Não foram repetidos testes de código sem mudanças. Os resultados anteriores permanecem rastreáveis à base e aos arquivos de evidência INFRA-1.1. Homologação externa continua bloqueada por acesso financeiro, recuperação/decisão de histórico e configuração de deploy ainda não aprovada. Não avançar para 2.6C ou dinheiro real.
