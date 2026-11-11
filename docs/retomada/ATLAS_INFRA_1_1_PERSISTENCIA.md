# ATLAS — INFRA-1.1: persistência e recuperação local

Data de referência do usuário: 10/11/2026 (America/Sao_Paulo).
PAPER_ONLY / COST_CONTROLLED / NO_PRODUCTION_DEPLOY.
Repositório exclusivo: `oliveirasamuel03-jpg/robo-investimentos`.

## Decisão e rastreabilidade

Correções e testes locais concluídos. Publicação e operação externa continuam
bloqueadas. Esta fase não recuperou o PostgreSQL antigo.

- Base: INFRA-1 `c3d9b75b2d8f7d26bd1c367b02ea3f5518072df7`.
- Branch: `atlas/infra-1-1-persistencia-segura`, filha da INFRA-1.
- Commit final: consultar o commit que introduz este relatório e suas evidências.
- Main não alterada. Nenhum PR, merge, deploy ou alteração operacional.
- Resultado do baseline antes das mudanças: 331 testes aprovados.

## Correções implementadas

| Problema | Comportamento desta branch | Evidência |
|---|---|---|
| DELETE seguido de INSERT sob autocommit podia deixar histórico apagado/parcial | `replace_json_rows` executa toda a substituição em `conn.transaction()` | Falhas reais de serialização e constraint SQL revertem DELETE e INSERT |
| JSON corrompido era interpretado como vazio/default | `PersistenceDataError` impede continuar; arquivo/registro fica preservado | JSON truncado, tipo errado, linha inválida e bootstrap com store corrompido |
| Linhas JSON inválidas do banco eram descartadas silenciosamente | Leitura rejeita a corrupção explicitamente | Registro inválido permanece no PostgreSQL e audit detecta corrupção |
| Escrita direta truncava arquivo antes de terminar | JSON/CSV passam por arquivo temporário no mesmo diretório, flush/fsync e replace atômico | Falha simulada de fsync/replace/serialização preserva arquivo anterior |
| `limit=0` local retornava todo o histórico | Retorna lista vazia sem conectar/criar schema; negativo rejeitado | Testes locais e PostgreSQL |

Arquivos de produção alterados: somente `core/persistence.py`.
Nenhuma estratégia, score, threshold, capital, ticket, max_open_positions,
provider de mercado ou autoridade shadow foi alterada.

**Mudança de comportamento intencional:** dados JSON existentes e inválidos agora
podem impedir abertura da página/ciclo. Isso é bloqueio de integridade, não motivo
para apagar o arquivo, substituir por default ou relaxar autenticação. Recuperar
cópia íntegra após investigação. Arquivo ausente continua tendo o default previsto.

## Testes executados

| Verificação | Resultado |
|---|---|
| Suíte completa offline | **349 passed, 0 failed, 42 warnings, 7,51 s** |
| Testes adicionados de persistência | 18 casos, incluídos nos 349 |
| Integração PostgreSQL real local | **19 verificações aprovadas** |
| Compileall dos arquivos relevantes | Aprovado |
| pip check | Sem dependências quebradas |
| git diff --check | Aprovado antes do commit |

Evidências: [suíte offline](evidencias/infra11-pytest.txt) e
[integração PostgreSQL](evidencias/infra11-postgres.json).

O runner offline bloqueou 38 tentativas de rede/processo, sem acesso a providers.
Os warnings preexistentes de datetime/pandas permanecem. O runner de integração
é separado: permite processos PostgreSQL locais e conexão autenticada em loopback,
não aceita DSN remoto e remove credenciais herdadas antes de iniciar.

As 19 verificações cobrem listener privado, inventário read-only sem criação de
schema, roundtrip de estado JSON, rollback por dois tipos de falha, preservação
de outro namespace, commit bem-sucedido, limites, corrupção, usuários com bcrypt,
estado STOPPED, flags PAPER e restauração exata de ambas as tabelas. A sequência
de IDs também avançou corretamente depois da restauração.

## PostgreSQL local: execução de fato

Nenhum PostgreSQL/Docker estava instalado. A instalação por apt não estava
disponível ao usuário local. Foram baixados fontes oficiais e verificados os
checksums publicados, sem instalar serviços de sistema ou usar privilégios elevados.

PostgreSQL 17.6 não compilou por ausência de bison. PostgreSQL **16.10** compilou
em diretório temporário do workspace. Essa versão é bancada de compatibilidade,
não recomendação de versão para produção nem afirmação de compatibilidade com
a versão desconhecida do banco antigo.

O primeiro teste de socket Unix foi bloqueado pelo ambiente (`Operation not
permitted`). O teste concluído usou somente `127.0.0.1`, porta efêmera, SCRAM-SHA-256,
senha aleatória temporária e nenhum listener externo. Não houve exposição pública.

O harness criou cluster próprio, schema e dados fictícios; injetou falhas; executou
pg_dump; restaurou em segundo banco descartável com transação única; comparou
conteúdo, timestamps, IDs e sequência. **Cluster encerrado e diretório temporário
descartado ao final.** Não há serviço contínuo nem backup real do usuário neste ensaio.

SHA-256 do dump fictício desta execução:
`506ef94893f361c6499b2a42999e42ce7bec4a26a798dcc9f4342abade7b66a6`.
O dump não é publicado e foi descartado com o cluster. O hash é evidência do ensaio,
não um identificador de backup recuperável do banco antigo.

## Ferramentas preparadas e execução reproduzível

```bash
.venv/bin/python scripts/test_offline.py -q
.venv/bin/python scripts/check_postgres_local.py --pg-bin /CAMINHO/POSTGRES/bin --transport loopback
```

O segundo comando exige binários PostgreSQL previamente instalados de fonte
confiável, recusa `.env`, não aceita banco existente nem DSN externo e encerra
seu próprio cluster. A opção padrão é socket Unix; `loopback` serve para ambientes
como este que não permitem esse socket. Não executar como root.

`scripts/audit_database_readonly.py` está preparado para uma futura inspeção
autorizada. Exige `ATLAS_AUDIT_DATABASE_URL`, não usa `DATABASE_URL` implicitamente
e não carrega `.env` nem módulos do app. Configura conexão read-only e usa transação
REPEATABLE READ/read-only, timeout SQL de 30s por instrução e leitura em lotes de 256.
Retorna versão, colunas, contagens e número de payloads inválidos, sem valores de
usuários, histórico, hashes de senha ou credenciais. Não exporta nem restaura dados.

**A ferramenta de auditoria não foi executada contra o banco antigo.**
Não inserir DSN em código, relatório, linha de comando ou chat. Usar ambiente
protegido quando houver acesso legítimo, custo conhecido e autorização aplicável.

## Limites importantes ainda abertos

1. Atomicidade de arquivo não resolve duas operações read-modify-write concorrentes.
   Web/worker não devem compartilhar arquivos como substituto do banco.
2. A transação impede substituição parcial, mas não define arbitragem entre dois
   escritores concorrentes para o mesmo namespace. Não habilitar múltiplos workers.
3. `_ensure_db_schema` e importação automática de arquivos locais continuam existindo
   nos loaders do app. Não usar o app para inspeção read-only de origem.
4. CSV malformado ainda pode ser interpretado como vazio por `read_table`; a nova
   validação estrita cobre JSON e as gravações atômicas cobrem JSON/CSV.
5. Validação da estrutura de domínio dentro de cada objeto JSON, corrupção de
   valores financeiros, schema incompatível, concorrência e grande volume exigem
   testes adicionais. JSON sintaticamente válido não significa estado operacional válido.
6. fsync do arquivo e rename não equivalem a garantia universal de durabilidade
   após queda de energia; não foi testado crash físico, fsync do diretório ou rede de arquivos.
7. Não homologados: versão do PostgreSQL antigo, carga prolongada, memória, providers
   LIVE, credenciais reais, rede Railway ou restauração de backup autêntico.

## Railway, custos e proibições preservadas

Leitura final confirmou `perceptive-integrity` com deploy CRASHED
`821211f6-531b-4f5a-a25c-2ae80dbb6d29` e nove mudanças STAGED no patch
`56534ac3-4dc4-4220-8fbf-f7f7b12bd393`. Source staged ainda aponta para INFRA-0,
não para INFRA-1.1. Nenhuma configuração Railway foi escrita nesta fase.

Nenhum recurso pago, plano, Agent, volume, bucket ou banco remoto contratado.
Não foi acionado gasto de provisionamento externo. Saldo/crédito/fatura atuais
não certificados; os limites financeiros permanecem pendentes. Nenhuma ordem real.

## Próximo passo e critérios de bloqueio

Antes de qualquer homologação externa: revisão de segurança desta branch,
confirmação financeira, controle de autodeploy/previews e acesso legítimo a backup.
Se houver backup: verificar versão/checksum, restaurar em ambiente isolado autorizado
e comparar histórico/usuários/posições. Se não houver: decisão expressa sobre
histórico novo. Banco novo jamais deve ser descrito como migração do antigo.

Continuar bloqueado com corrupção, restore sem comparação, autenticação ausente,
flags reais, dois escritores sem coordenação, custos desconhecidos ou auto-preview
não verificado. Não criar PR até resolver risco de acionamento automático.

Rollback: revert somente na branch, com revisão. Não reintroduzir perda silenciosa
de dados como solução de disponibilidade. Não há rollback remoto a fazer, pois
esta fase não alterou a infraestrutura.

## Referências técnicas utilizadas

- [Transações explícitas em conexões autocommit — Psycopg](https://www.psycopg.org/psycopg3/docs/basic/transactions.html)
- [Fontes oficiais PostgreSQL 16.10](https://ftp.postgresql.org/pub/source/v16.10/)
- [Obtenção do código-fonte PostgreSQL](https://www.postgresql.org/docs/17/install-getsource.html)
