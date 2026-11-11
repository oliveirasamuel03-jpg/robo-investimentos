# Banco novo — preparação sem provisionamento

11/11/2026. PAPER_ONLY / NO_PRODUCTION_DEPLOY.
Solicitação: preparar banco novo. Isso inicia histórico do zero; não recupera
operações, usuários, relatórios ou estado do banco antigo. Volume antigo preservado.

## Entrega

`scripts/sql/new_history_schema.sql` reproduz o esquema de `core/persistence.py`:
`app_state`, `app_rows`, sequência de IDs e índice por namespace/seq.
Uma transação cria a estrutura sem inserir usuários, ordens, capital ou parâmetros.
Recusa esquema public com tabelas, views ou sequências existentes; não apaga nem
substitui dados. Não é ferramenta de migração nem de reexecução idempotente.
A proteção pressupõe banco dedicado e nenhum escritor concorrente durante o preparo.

## Execução futura, condicionada à autorização

1. Escolher destino PostgreSQL e comprovar custos/limites. Nenhum destino remoto
   foi criado ou escolhido automaticamente. Não usar o endpoint antigo.
2. Confirmar identidade, nome do banco, versão, ownership e isolamento do destino
   vazio. PostgreSQL local 16.10 é o ambiente de teste; outras versões exigem validação.
3. Manter web/worker desligados. Após autorização específica para escrita remota,
   um operador aplica o SQL com cliente que pare ao primeiro erro, em conexão
   protegida. Não incluir DSN, senha ou dump no GitHub.
4. Validar tabelas vazias, índice e sequência antes de iniciar a aplicação.
   Se houver erro, interromper; não contornar a recusa com DROP/TRUNCATE.
5. Preparar AUTH_REQUIRED=true, BROKER_MODE=paper, BROKER_PROVIDER=paper.
   Criar administrador por bootstrap de servidor com segredo protegido; jamais
   cadastro público ou acesso anônimo. Remover segredo de bootstrap após validação.
6. Antes da primeira abertura, revisar os loaders: o app pode importar arquivos
   locais/defaults para namespaces ausentes. Usar armazenamento local vazio,
   sem CSV/JSON históricos ou fixtures; revisar explicitamente qualquer seed.
7. Homologar web autenticada e conferir estado STOPPED e flag real desativada.
   Não habilitar worker para preencher histórico. Não criar operações fictícias
   no banco remoto para aparentar uma migração bem-sucedida.
8. Registrar marco de início do novo histórico fora das tabelas operacionais;
   definir backup e restaurar uma cópia em ambiente isolado antes do worker PAPER.

## Teste gratuito

O harness `scripts/check_postgres_local.py` cria e remove somente cluster local
temporário com dados sintéticos. Agora valida o SQL, tabelas inicialmente vazias
e recusa de segunda inicialização preservando um registro. Não aceita DSN remoto.

```sh
.venv/bin/python scripts/check_postgres_local.py --pg-bin /caminho/local/postgresql/bin --transport loopback
.venv/bin/python scripts/test_offline.py
```

## Evidência executada em 11/11/2026

- PostgreSQL 16.10 local: 21 verificações aprovadas, incluindo os dois cenários
  novos de inicialização; cluster encerrado e dados temporários removidos pelo harness.
- Suíte offline: 349 passed, 42 warnings, 5.51s; 38 tentativas de rede/processo
  bloqueadas. Os avisos são mantidos na saída, não tratados como aprovação adicional.
- `git diff --check`: aprovado. Evidência completa da suíte em
  `evidencias/banco-novo-pytest.txt`.
- Dump sintético descartável SHA-256:
  `39dfa222193bf27a60a6b233ca5f2372e767dcfab2b02e598724a54ec05cff66`.
  Não contém dados antigos nem representa backup de usuário.

## Bloqueios operacionais

Provisionamento, migrations remotas, deploy e worker não autorizados nesta entrega.
Falha de auth, real mode, importação involuntária de histórico, custo desconhecido,
estado inconsistente ou teste reprovado bloqueiam a homologação. Um banco vazio
não torna o robô pronto para operar. Rollback de preparação é interromper antes
do deploy e preservar evidências; nunca excluir automaticamente dados ou recursos.
